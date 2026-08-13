# Playbook: train SMP cho một motion mới hoặc một robot mới

Tài liệu này đúc kết từ một lần train **đã chạy thành công và đã kiểm chứng bằng video**: humanoid
native của MimicKit học động tác spinkick từ một clip duy nhất, bằng SMP (Score-Matching Motion
Priors), trên Kaggle với 2× Tesla T4.

Mục đích: lần sau đổi sang motion khác, hoặc sang robot khác (VR M3.1), thì biết **đổi cái gì, mong
đợi số liệu ra sao, và dừng ở đâu**. Các cell Kaggle cụ thể nằm ở
[`HUONG_DAN_SMP_KAGGLE.md`](HUONG_DAN_SMP_KAGGLE.md); file này nói về *quy trình* và *cách đọc kết
quả*, không lặp lại lệnh.

---

## 1. Bằng chứng: kết quả đã đạt được

Hai session nối tiếp nhau, session sau load checkpoint của session trước qua `--model_file`.

| | Session 1 (`mbi5axg4`) | Session 2 (`m6rv7ht3`) |
|---|---|---|
| Iterations | 915 | 5 340 |
| Samples | 60.0 M | 350.0 M |
| Thời gian | 1 h 27 m | 7 h 27 m |
| Throughput | 11 546 /s | 13 058 /s |
| `Sds_Loss_Mean` | 2.41 → 0.93 | 0.58 → **0.186** |
| `Train_Episode_Length` | 17.5 → 41.7 | 24.2 → **298.3** / 300 |
| `Test_Episode_Length` | 18.2 → 53.9 | 54.7 → **300** (kịch trần) |
| `Ep_Len_Frac` | 0.14 | **0.994** |
| `Timeout_Frac` | ~0 | **0.94** |
| `Critic_Loss` | 0.06 → 0.66 | 0.85 → **0.098** |

**Tổng chi phí thực tế: 410 M samples, ~9 giờ trên 2× T4** — nhưng 100 M cuối là thừa. Lần sau đặt
**320 M và chạy một session duy nhất** (~6 h 35 m); xem [§6.1](#61-gọn-trong-một-session--cách-nên-làm).

Bước ngoặt xảy ra ở khoảng iteration 400–600 của session 2 (tức ~90 M samples tích luỹ):
`Fail_Frac` rơi từ 1.0 xuống 0.22, `Ep_Len_Frac` nhảy từ 0.18 lên 0.79. Trước mốc đó robot ngã ở
mọi episode; sau mốc đó gần như không ngã nữa. Đừng bỏ cuộc trước mốc này.

### Kiểm chứng bằng video

`policy_final.mp4` (301 frame, 10.03 s): humanoid đứng vững ở **mọi frame**, lặp chu kỳ
đá–hồi phục–lấy đà 5–6 lần trong 10 giây. Soi kỹ một chu kỳ thấy đủ bốn pha: đá chân duỗi cao với
thân nghiêng ngược lấy momen → tiếp đất gối khuỵu hấp thụ → về tư thế đứng trung tính ổn định →
khuỵu gối lấy đà cho cú tiếp theo. Thân có xoay yaw thật giữa các cú.

Khác biệt còn lại so với clip gốc, nhất quán với việc `Sds_Loss_Mean` dừng ở 0.186 chứ không về 0:

- Clip gốc giữ tay guard sát đầu; policy hay dang tay ngang để giữ thăng bằng — chiến lược do vật lý
  ép ra, không có trong data.
- Cú đá của policy thấp hơn, ít duỗi hết chân, thân nghiêng nông hơn.
- Clip gốc dài 1.3 s một cú; policy lặp tuần hoàn vô hạn. **Đây là đúng bản chất SMP**, không phải
  lỗi: SMP không tracking theo phase, nó chỉ ép phân phối chuyển động khớp với prior.

Đây là mức "đạt" hợp lý cho một clip đơn. Đừng kỳ vọng trùng khít từng frame.

---

## 2. Pipeline năm giai đoạn

```
clip motion (.pkl)  ──►  [0] kiểm khả thi (5 giây, §4.1)
                                │  13/27 khớp vượt mô-men?  ──► giãn clip trước
                                ▼
                         [1] train prior TinyMDM  ──►  smp_prior_<name>.pt
                                                              │
                                                              ▼
                         [2] train policy PPO + SMP reward  ──►  model.pt
                                                              │
                                                              ▼
                         [3] rollout + render  ──►  policy_final.mp4
                                                              │
                                                              ▼
                         [4] đọc metric + xem video  ──►  đạt / train tiếp
```

**[0] Kiểm khả thi** — `tools/retime_motion.py --report`. Miễn phí và bắt buộc với clip retarget:
retarget khớp tư thế mà không biết gì về giới hạn actuator. Bỏ qua bước này đã tốn 216 M sample một
lần rồi. Chi tiết ở §4.1.

**[1] Prior** — `tools/diffusion_model/train_tinymdm.py --cfg_path <cfg> --out_dir <dir>`.
Mô hình khuếch tán DiT nhỏ (2 layer, 4 head) học phân phối các đoạn chuyển động 10 bước từ clip.
50 000 iteration, **chạy một GPU duy nhất** — script là vòng lặp một tiến trình, không có
`torch.distributed`, nên GPU thứ hai nằm không ở giai đoạn này.

**[2] Policy** — `mimickit/run.py`. Reward = `exp(-sds_loss_norm × sds_loss_scale)`, tức
"đoạn chuyển động vừa sinh ra có giống thứ prior đã học không". Không có task reward
(`task_reward_weight: 0.0`). Đây là giai đoạn đắt nhất, và là chỗ dùng được nhiều GPU qua `--devices`.

**[3] Video** — `kaggle/make_videos.py`, gọi lần lượt `tools/play_policy_to_mp4.py` (roll out policy
trong sim, ghi lại state) rồi `tools/render_robot_video.py` (MuJoCo EGL + ffmpeg). Tách rời khỏi
simulator nên không phụ thuộc GL stack lúc train.

**[4] Đánh giá** — mục 5 và 7 bên dưới.

---

## 3. Đổi gì khi sang robot khác hoặc motion khác

Đây là phần cốt lõi. Bảng dưới liệt kê **mọi** chỗ phải sửa, lấy cặp humanoid/spinkick và
M3.1/zombie làm ví dụ đối chiếu.

| File | Khoá | Humanoid | VR M3.1 |
|---|---|---|---|
| `data/envs/smp_<robot>_env.yaml` | `char_file` | `humanoid/humanoid.xml` | `vr_m3_1/vr_m3_1.xml` |
| | `motion_file` | `humanoid_spinkick.pkl` | `vr_m3_1_long_zombie_fixed.pkl` |
| | `init_pose` | 34 số (28 dof) | **33 số (27 dof)** |
| | `key_bodies` | `head, *_hand, *_foot` | `head_pitch_link, *_wrist_pitch_link, *_ankle_roll_link` |
| | `contact_bodies` | `right_foot, left_foot` | `*_ankle_roll_link` |
| `tools/diffusion_model/config/tinymdm_<robot>.yaml` | `env_config` | trỏ về env ở trên | trỏ về env ở trên |
| | `motion_file` | **phải trùng** env | **phải trùng** env |
| `data/agents/smp_<robot>_agent.yaml` | `smp_prior_cfg` | file tinymdm ở trên | file tinymdm ở trên |
| | `smp_prior_model` | `.pt` do bước [1] sinh | `.pt` do bước [1] sinh |
| `args/smp_<robot>_args.txt` | `--env_config` / `--agent_config` | hai file trên | hai file trên |

### Bốn cái bẫy khi đổi robot

**`init_pose` sai độ dài = mọi khớp lệch, không có lỗi báo.** Định dạng là
`[root_pos(3), root_rot_expmap(3), dof_pos(N)]`. Humanoid 28 dof → 34 số; M3.1 27 dof → 33 số. Copy
file humanoid sang rồi chỉ sửa vài số là hỏng âm thầm. **Đếm trước khi sửa.**

**Chiều cao root trong `init_pose` phải đo, không đoán.** M3.1 dùng 0.854 m — vị trí mà đỉnh
va chạm thấp nhất chạm z = 0 khi chân duỗi thẳng. Đặt thấp quá thì robot xuyên sàn ngay frame đầu,
cao quá thì rơi tự do vào đầu episode.

**`contact_bodies` đọc ngược với trực giác.** Đây là danh sách các body **được phép** chạm đất.
`deepmimic_env.py:744-749` xoá lực tiếp xúc của những body trong danh sách rồi kết thúc episode nếu
*bất kỳ thứ gì khác* đang chạm:

```python
masked_contact_buf[:, contact_body_ids, :] = 0
has_fallen = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
```

Danh sách rỗng ⇒ không gì được chạm đất ⇒ mọi episode chết ở frame một. Nới theo motion, đừng thu
hẹp: cartwheel / vault → thêm cổ tay; crawl / roll / getup → thêm thân và cẳng tay.

**Kiểm tra clip gốc có tự vi phạm `contact_bodies` không.** Với M3.1/zombie đã kiểm: qua toàn bộ
1098 frame, body gần sàn nhất ngoài bàn chân là đầu gối ở +5.3 cm. Nếu chính clip tham chiếu chạm
sàn bằng body không nằm trong danh sách thì policy không bao giờ học được — nó bị phạt vì bắt chước
đúng.

---

## 4. Sáu assert phải qua trước khi train chạy

`SMPAgent._check_prior_env_config()` (`mimickit/learning/smp_agent.py:85`) đối chiếu env của prior
với env của RL và **assert ngay lúc khởi động**. Sai một cái là dừng, may mắn là dừng sớm chứ không
train sáu tiếng rồi mới biết:

| Khoá | Yêu cầu |
|---|---|
| `global_obs` | prior == env |
| `root_height_obs` | prior == env |
| `enable_tar_obs` | prior == env (phải `False` cho SMP thuần) |
| `num_disc_obs_steps` | prior == env |
| `disc_dof_vel_obs` | prior == env |
| `key_bodies` | **số lượng** phải bằng nhau |
| `control_freq` | prior config == `1 / engine.timestep` |

Cách an toàn: prior config trỏ thẳng `env_config:` vào đúng file env mà RL sẽ dùng. Lúc đó năm khoá
đầu tự khớp, chỉ còn `control_freq` phải tự đặt bằng `control_freq` trong
`data/engines/newton_engine.yaml` (hiện là 30).

Một assert **không** được kiểm: số dof. Prior học trên 28 dof của humanoid không thể điều khiển robot
27 dof, và không có gì chặn bạn trỏ nhầm.

---

## 4.1 Kiểm clip có khả thi với robot không — làm trước khi tốn GPU

Miễn phí, mất 5 giây, và nó là nguyên nhân gốc của thất bại đắt nhất tới giờ.

```bash
python tools/retime_motion.py --input <clip>.pkl --char_file <robot>.xml --report
```

**Retarget chỉ khớp tư thế.** Không bước nào trong pipeline retarget biết giới hạn actuator, nên clip
nhìn từng khung thì đúng vẫn có thể đòi mô-men phần cứng không sinh nổi. Script tính quán tính hợp
thành quanh từng trục khớp từ MJCF, lấy `τ = I_eff · q̈` trên toàn clip, so với `actuatorfrcrange`.

Với `vr_m3_1_humanoid_spinkick.pkl` trên M3.1: **13/27 khớp vượt giới hạn**, `right_hip_roll` 3.5×,
`waist_yaw` 3.8×, `left_shoulder_pitch` 11.8×. Không phải nhiễu vi phân số — chỉ 0.9–3.8 % năng
lượng clip nằm trên 8 Hz.

Dấu hiệu nhận ra khi đã lỡ train: **`Sds_Loss` tốt mà vẫn ngã.**

| | humanoid @ 52 M | M3.1 @ 216 M |
|---|---|---|
| `Ep_Len_Frac` | **0.965** | 0.628 |
| `Sds_Loss_Mean` | 0.321 | **0.260** |

M3.1 bám tư thế tốt hơn humanoid mà vẫn đổ, vì nó luôn trễ nhịp và trễ nhịp lúc trụ một chân thì ngã.
Bài toán không nằm ở imitation, và thêm sample không mua được thứ actuator không có.

**Cách sửa: giãn clip, đừng đụng gains.** τ tỉ lệ `1/s²` nên `s = sqrt(vượt)`:

```bash
python tools/retime_motion.py --input <clip>.pkl --output <clip>_slow2.pkl \
    --char_file <robot>.xml --factor 2.0
```

`--auto` chọn `s` từ nhóm khớp còn lại sau `--exclude` (mặc định bỏ qua tay — vai bão hòa không làm
robot ngã, hông bão hòa thì có). Với M3.1 spinkick, 2.0× đưa toàn bộ chân và thắt lưng vào giới hạn;
tuân thủ cả tay cần 3.44×, chậm tới mức không còn ra cú đá.

Việc này **không đổi một byte vật lý nào**, và đó là điểm mấu chốt. `stiffness` / `damping` trong MJCF
là kp/kd của vòng PD — `newton_engine.py:837-848` copy chúng vào `joint_target_ke` / `joint_target_kd`
— và với robot thật chúng chép từ constants của phần cứng (M3.1: `vr_m3_1_constants.py` của mjlab).
Chỉnh chúng cho dễ train là làm hỏng tính hợp lệ sim-to-real và phải được đội phần cứng duyệt. Giãn
clip thì không: robot y nguyên, chỉ bảo nó làm chậm lại. Lấy mẫu lại theo thời gian cũng không sinh
tư thế mới nào ngoài bao lồi của clip gốc, nên các kiểm tra hình học (độ hở mặt đất, `contact_bodies`)
vẫn giữ nguyên kết quả.

Sau khi giãn thì **phải train lại prior** — prior học nhịp của clip; dùng prior cũ là chấm điểm nhịp
mới bằng thước cũ.

Ba lưu ý khi so A/B:

- `Ep_Len_Frac` so trực tiếp được giữa hai clip dài ngắn khác nhau, vì `amp_env.py:252` ép
  `motion_len_term` False — episode kết thúc do ngã hoặc hết `episode_length`, không theo độ dài clip.
- Sàng lọc ở **30 M sample** (~40 phút) là đủ để phân biệt, đừng chạy full ngay.
- Dưới 50 M thì `sds_normalizer_samples` chưa kích hoạt, nên A/B vẫn hợp lệ dù baseline có set nó.

---

## 5. Đọc metric: cái nào tin được, cái nào không

### Bỏ qua hoàn toàn

**`Train_Return` và `Test_Return` luôn bằng 0.** Không phải lỗi. Env `amp`/`smp` không bao giờ ghi
vào `_reward_buf` (`amp_env.py:280`, `_update_reward()` rỗng); reward SMP được tiêm vào experience
buffer *sau* rollout, ở `smp_agent.py:174`. Return tracker chỉ nhìn env reward, nên nó thấy 0.

**`Smp_Reward_Mean` giảm dần không có nghĩa là tệ đi.** Ở run vừa rồi reward đi từ 0.25 xuống 0.094
trong khi policy tốt lên rõ rệt. Lý do: `DiffNormalizer` chia SDS loss cho trung bình trị tuyệt đối
tích luỹ của chính nó, và `sds_normalizer_samples` không được set trong agent yaml nên mặc định là
`np.inf` (`smp_agent.py:28`) — normalizer cập nhật mãi mãi. Mẫu số giảm theo tử số:

| | đầu run | cuối run |
|---|---|---|
| `Sds_Loss_Mean` (thô) | 0.58 | 0.186 |
| `Sds_Norm_Scale` (mẫu số) | 1.33 | 0.426 |
| `Sds_Loss_Norm_Mean` (tỉ số) | 0.432 | 0.435 |
| `Smp_Reward_Mean` | 0.252 | 0.094 |

Reward là đại lượng **tương đối theo thời gian thực**, không so sánh được giữa các iteration, cũng
không so được giữa hai run kể cả của cùng một robot.

**Cách xử lý đúng là đọc `Sds_Loss_Mean` chứ không phải đóng băng normalizer.** `Sds_Loss_Mean` vốn
đã là đại lượng tuyệt đối và so được giữa các run.

Đừng set `sds_normalizer_samples`. Bản trước của tài liệu này khuyên đặt 50 M cho reward dễ đọc, và
run M3.1 đã trả giá cho lời khuyên đó. Mẫu số co lại không phải phiền toái — nó là **một curriculum
tự động**: khi loss giảm, mẫu số giảm theo, nên độ nhạy của reward với phần sai số còn lại cứ tăng
dần, càng về cuối càng soi kỹ. Humanoid cưỡi curriculum đó suốt 350 M sample
(`Sds_Norm_Scale` 1.33 → 0.426). M3.1 đóng băng ở 0.869 từ mốc 50 M, và hệ quả:

```
iter 2100   137.6 M   Sds 0.2582   <- chạm đáy
iter 3300   216.3 M   Sds 0.2597   <- 80 M sau, không nhúc nhích
```

`Ep_Len_Frac` vẫn bò lên trong 80 M đó, nên nhìn qua tưởng còn tiến bộ. Thực chất policy đã ngừng
học bắt chước từ 140 M và chỉ còn mua thêm thăng bằng.

### Tin được

| Metric | Nghĩa | Đích |
|---|---|---|
| `Sds_Loss_Mean` | Khoảng cách thô tới prior. **Thước đo tiến bộ chính.** | giảm đơn điệu rồi phẳng |
| `Ep_Len_Frac` | Độ dài episode / trần. Đo trực tiếp "có trụ được không". | > 0.9 |
| `Timeout_Frac` | Tỉ lệ episode kết thúc do hết giờ thay vì ngã. | > 0.9 |
| `Fail_Frac` | Tỉ lệ ngã. Nhiễu khi số termination mỗi iter nhỏ. | gần 0 |
| `Test_Episode_Length` | Chạy ở chế độ deterministic. | = trần (300) |
| `Critic_Loss` | Value function có fit nổi không. | giảm về < 0.2 |
| `Clip_Frac` | Bước cập nhật có quá lớn không. | 0.1 – 0.25 |

`Ep_Len_Frac`, `Fail_Frac`, `Timeout_Frac`, `Terminations` là các metric được thêm vào
`base_agent._calc_termination_info()` sau session 1, chính vì session 1 không có cách nào phân biệt
"episode ngắn do ngã" với "episode ngắn do hết giờ".

### Mốc dừng

Train tiếp khi `Sds_Loss_Mean` còn giảm rõ. Dừng khi nó phẳng: 1 500 iteration cuối của session 2
chỉ đổi được 0.188 → 0.186, tức ~100 M samples cho 1 % cải thiện. Lúc đó nếu chất lượng vẫn chưa đạt
thì vấn đề nằm ở prior, ở reward scale, hoặc ở chính clip — không phải ở số lượng sample.

**Trường hợp nguy hiểm: `Sds_Loss_Mean` phẳng trong khi `Ep_Len_Frac` vẫn tăng.** Đây không phải hội
tụ, mà là hai mục tiêu đã tách nhau — policy ngừng học bắt chước và chỉ còn học không ngã. M3.1 chạy
80 M sample trong trạng thái này. Dừng lại và hỏi tại sao nó không bắt chước tốt hơn được nữa; câu
trả lời thường là clip đòi hỏi thứ robot không làm nổi (§4.1), chứ không phải cần thêm sample.

---

## 6. Ngân sách

| Giai đoạn | Chi phí đo được |
|---|---|
| Prior TinyMDM, 50 k iter | 1 GPU, vài chục phút cho clip đơn |
| Policy tới khi hết ngã | ~90 M samples |
| Policy tới khi hội tụ | ~310 M samples ≈ 6 h 35 m trên 2× T4 (humanoid; §6.1) |
| Rollout + render 3 video | ~2–3 phút |

Throughput thực đo: **13 058 samples/s** với `--num_envs 1024 --devices cuda:0 cuda:1`
(1024 là **mỗi GPU**, tổng 2048 env). Kaggle giới hạn 9 h interactive / 12 h batch.

Mọi con số ở mục này là của **humanoid native**. Sang robot khác phải đo lại — xem [§6.1b](#61b-ngân-sách-phụ-thuộc-robot--đo-lại-đừng-chép).

### 6.1 Gọn trong MỘT session — cách nên làm

Lần đầu chạy hai session vì không biết cần bao nhiêu. Giờ đã biết, nên **không cần chia nữa**.

Con số 410 M ở trên là tổng cộng thực tế, nhưng nó **thừa**. Nhìn lại quỹ đạo `Sds_Loss_Mean`:

| Samples tích luỹ | `Sds_Loss_Mean` |
|---|---|
| 60 M (hết session 1) | 0.93 |
| 132 M | 0.270 |
| 204 M | 0.211 |
| **309 M** | **0.188** |
| 410 M (hết session 2) | 0.186 |

100 M samples cuối chỉ đổi được 0.002 — khoảng 2 giờ GPU cho 1 % cải thiện. **Điểm hội tụ thực tế
là ~310 M**, tức **6 h 35 m** ở throughput đo được.

```
--max_samples 320000000
```

Cộng setup ~5 phút, prior (nếu phải train) ~30–40 phút, render video ~3 phút:

| Kịch bản | Tổng thời gian | Chạy được ở đâu |
|---|---|---|
| Prior có sẵn (humanoid/spinkick) | **~6 h 45 m** | interactive (9 h) hoặc batch |
| Phải train prior (robot mới, ví dụ M3.1) | **~7 h 30 m** | interactive vẫn kịp, batch an tâm hơn |
| Muốn dư dả, để `--max_samples 400000000` | ~8 h 40 m | **chỉ batch** (12 h) |

**Dùng batch mode cho chắc.** Interactive 9 h nghe thì đủ, nhưng nó phụ thuộc tab trình duyệt còn
mở; mất mạng giữa chừng là mất session. Save Version → **Save & Run All (Commit)** chạy headless tới
12 h, `/kaggle/working` được giữ nguyên thành Output của version đó. Nhớ đính kèm cả Secrets lẫn
dataset data pack cho version, và chọn accelerator T4 x2 **trước khi** commit.

**Bị cắt giữa chừng vẫn không mất gì.** `base_agent.py:456` ghi đè `model.pt` mỗi
`iters_per_output` = 100 iteration, tức mỗi ~6.5 M samples ≈ **8 phút**. Thêm
`--save_int_models true` thì mỗi mốc đó còn được giữ lại một bản riêng trong `int_models/`
(27 MB × ~49 bản ≈ 1.3 GB, thoải mái trong hạn 20 GB của `/kaggle/working`). Session chết ở phút
thứ 400 thì bạn vẫn có checkpoint của phút 392.

Chạy một session thì **bỏ hẳn** hai cell nối session: không cần
`wandb_upload.py --download` ở đầu, không cần `--model_file`. Thứ tự cell rút gọn còn:

```
secrets → clone → setup.sh → prepare_data.py → [train prior] → smoke test
        → train policy (--max_samples 320000000) → make_videos.py → upload model.pt
```

### 6.1b Ngân sách phụ thuộc robot — và ĐỪNG ngoại suy từ vài mốc đầu

310 M là con số của **humanoid**, không chuyển sang robot khác được:

| | Humanoid | VR M3.1 (clip gốc) |
|---|---|---|
| Throughput | 13 058 /s | **8 500–8 700 /s** (~65 %) |
| Giải xong thăng bằng (`Ep_Len_Frac` > 0.96) | **52 M** | không đạt trong 216 M |
| `Sds_Loss` chạm đáy | ~310 M (0.186) | 137 M (0.258), rồi phẳng |
| Hội tụ | 350 M / 7 h 45 m | không hội tụ — clip bất khả thi (§4.1) |

**Bài học đắt nhất của dự án này nằm ở ô "không đạt".** Từ bốn mốc đầu của M3.1 tôi kết luận nó
"hiệu quả gấp đôi humanoid" rồi chốt ngân sách 200 M. Cả hai đều sai. Hai đường cong có **hình dạng
khác hẳn nhau**:

- Humanoid **bò rồi bật**: lẹt đẹt dưới 0.15 tới 52 M rồi dựng gần như thẳng đứng lên 0.96.
- M3.1 **lên đều rồi tà dần**: tăng tuyến tính tới ~150 M rồi chậm lại, tiệm cận dưới 0.85.

Ba, bốn điểm đầu tiên **không phân biệt được hai hình dạng đó**. Ngoại suy tuyến tính từ chúng cho ra
số sai theo cả hai chiều, và mỗi lần sai là vài tiếng GPU.

Heuristic "nhân mốc bước ngoặt với 3.5" ở bản trước của tài liệu này là sản phẩm của đúng sai lầm ấy
— **đừng dùng.**

**Cách làm đúng cho một robot mới:**

1. Chạy `retime_motion.py --report` trước (§4.1). Miễn phí, và loại luôn khả năng clip bất khả thi.
2. Chạy 30 M sàng lọc. Mục tiêu là phát hiện lỗi cấu hình và so A/B, **không phải** để ước ngân sách.
3. Chỉ ước ngân sách sau khi `Ep_Len_Frac` đã vượt điểm bật (> 0.9). Trước đó thì con số duy nhất
   trung thực là "chưa biết".
4. Theo dõi `Sds_Loss_Mean` để quyết định dừng. Nếu nó phẳng mà `Ep_Len_Frac` còn tăng, xem §5 —
   đó là dấu hiệu bỏ tiền mua nhầm thứ.

### 6.2 Khi nào vẫn nên chia hai session

- **Robot/motion hoàn toàn mới, chưa biết ngân sách.** Chạy 60–100 M trước, xem `Sds_Loss_Mean` và
  `Ep_Len_Frac` có đi đúng hướng không rồi mới cam kết cả ngày GPU. Sai config thì mất 1 h chứ không
  mất 8 h.
- **Clip dài.** Zombie 1098 frame nhiều khả năng cần hơn 310 M; nếu vượt ~560 M thì 12 h batch cũng
  không đủ và buộc phải nối.

Cách nối: upload `model.pt` lên WandB Artifact ở cuối session, session sau kéo về bằng
`kaggle/wandb_upload.py --download <artifact>:latest --dest <dir>` rồi
`--model_file <dir>/model.pt`. Đặt `WANDB_NAME` khác nhau cho mỗi session để dễ đối chiếu.

Lưu ý khi nối: `--max_samples` **đếm lại từ 0** ở session mới, không cộng dồn. `_sample_count` là
biến python thường (`base_agent.py:34`), không nằm trong state dict, nên `load()` không khôi phục nó.
Muốn tổng 320 M qua hai session thì đặt `--max_samples` cho từng session, không phải cho tổng.

Hệ quả phụ: `normalizer_samples: 100000000` cũng tính lại từ đầu, tức obs normalizer tiếp tục cập
nhật thêm 100 M samples nữa ở session sau. Vô hại, nhưng nhớ là vậy khi so hai đường cong.
Ngược lại, trọng số của `DiffNormalizer` (`_mean_abs`, `_count`) **được** khôi phục — chúng khai báo
là `nn.Parameter` (`diff_normalizer.py:66-67`) nên nằm trong checkpoint.

---

## 7. Verify bằng video

`kaggle/make_videos.py` sinh ba file. **Chỉ hai trong ba là dùng được:**

| File | Dùng được | Là gì |
|---|---|---|
| `reference_data.mp4` | ✅ | Clip gốc từ data. Ground truth. |
| `policy_final.mp4` | ✅ | Policy chạy trong physics sim. Cái cần đánh giá. |
| `reference_sim_final.mp4` | ❌ **hỏng** | Đứng yên một tư thế suốt cả clip. |

`reference_sim_final.mp4` hỏng vì `amp_env.py:186`:

```python
def _update_ref_motion(self):
    if (self._enable_ref_char()):
        super()._update_ref_motion()
```

`_enable_ref_char()` = `self._visualize and self._visualize_ref_char` (`deepmimic_env.py:138`), mà
`play_policy_to_mp4.py` chạy headless nên `visualize=False`. Buffer `_ref_root_pos` / `_ref_dof_pos`
không bao giờ được cập nhật sau reset, script đọc đúng các buffer đó và ghi ra một pose lặp 301 lần.
**Đừng dùng file này để so sánh và đừng upload nó.** Sửa được bằng cách cho
`play_policy_to_mp4.py` tự lấy frame từ `env._motion_lib.calc_motion_frame()` theo thời gian
episode, thay vì đọc buffer bị gate.

### Dấu hiệu đó là vật lý thật, không phải playback

`policy_final.mp4` là đầu ra của simulator (MuJoCo-Warp, 240 Hz vật lý / 30 Hz điều khiển, 8 bước
tích phân mỗi action, torque chặn bởi `actuatorfrcrange`, ma sát Coulomb μ = 1.0). Nhìn vào video,
những thứ sau chỉ có ở chuyển động do vật lý sinh ra:

- Có pha bay, thân theo quỹ đạo parabol, không lơ lửng giữa chừng.
- Tay vung phản pha với thân khi xoay (bảo toàn momen động lượng).
- Tiếp đất có hấp thụ: gối khuỵu rồi mới đứng thẳng.
- Có run và trôi, tư thế không lặp lại y hệt.

Ngược lại `reference_data.mp4` là kinematic thuần — nó có thể xuyên đất và xoay không cần momen.
Nếu `policy_final.mp4` cũng làm được mấy chuyện đó thì mới là có vấn đề.

---

## 8. Cạm bẫy hạ tầng đã trả giá

Mỗi dòng dưới đây là một lần chạy hỏng trên Kaggle.

**File chưa commit thì bản clone không có.** `tools/render_robot_video.py` và
`tools/play_policy_to_mp4.py` từng nằm untracked, nên `make_videos.py` chết với
`Errno 2: No such file or directory` — sau khi đã train xong 7 tiếng. Trước mỗi lần chạy Kaggle:
`git status --short --untracked-files=all` và kiểm xem thứ mình sắp dùng có trong `git ls-files` không.

**Clone lại thì phải chạy lại `prepare_data.py`.** Script tạo symlink *bên trong* `data/` của repo,
nên xoá repo là mất theo. Triệu chứng: `FileNotFoundError` ở một motion file, xuất hiện muộn hơn
nhiều so với nguyên nhân.

**Kaggle không đảm bảo loại GPU.** Notebook cấp P100 (sm_60) thì PyTorch trong image không có kernel
tương thích (`no kernel image is available for execution on the device`) vì nó chỉ build cho sm_70+.
Chọn **T4 x2** (sm_75). Kiểm ngay ở cell đầu, đừng để chết ở cell rollout.

**Quay video trong lúc train với nhiều GPU từng làm crash.** `Logger._mp_aggregate` all-reduce mọi
log entry dưới dạng float64, và object `Video` không phải số. Đã sửa bằng cách lọc theo
`isinstance(..., numbers.Number)`, nhưng đường an toàn vẫn là `--video false` khi train rồi render
offline sau.

**`warp-lang` ghim ở 1.15.0.** Bản 1.16.0 làm hỏng kernel `J_kj` của mujoco-warp.

**IPython `!` magic phải chiếm trọn một dòng.** `os.chdir(x); !git pull` là `SyntaxError`.

---

## 9. Checklist trước khi bấm train

1. `retime_motion.py --report` đã chạy chưa, và clip có nằm trong giới hạn mô-men không? (§4.1)
   Nếu đã giãn clip: prior có được train lại trên clip mới không?
2. `git status --short --untracked-files=all` — mọi file sắp dùng đã được commit chưa?
3. Accelerator là T4 x2 chưa? (`nvidia-smi` ở cell đầu)
4. Số dof trong `init_pose` có khớp robot không? Đếm, đừng đoán.
5. Chiều cao root có phải đo từ MJCF không?
6. `contact_bodies` có phù hợp với motion không? Clip gốc có tự vi phạm nó không?
7. `motion_file` trong env config và trong prior config có **trùng nhau** không?
8. `smp_prior_model` có trỏ đúng file `.pt` mà bước [1] vừa sinh không?
9. Smoke test 2 phút (`--max_samples 200000`) trước khi chạy full.
10. `WANDB_NAME` đã đặt riêng cho session này chưa?

---

## 10. Trạng thái hiện tại cho VR M3.1

Hướng dẫn từng cell: [HUONG_DAN_SMP_M3_SPINKICK.md](HUONG_DAN_SMP_M3_SPINKICK.md).

**Đã xong.** Toàn bộ cấu hình spinkick đã commit và đã chạy thật:

```
data/envs/smp_vr_m3_1_spinkick_env.yaml          data/envs/smp_vr_m3_1_spinkick_slow2_env.yaml
data/agents/smp_vr_m3_1_spinkick_agent.yaml      data/agents/smp_vr_m3_1_spinkick_slow2_agent.yaml
args/smp_vr_m3_1_spinkick_kaggle_args.txt        args/smp_vr_m3_1_spinkick_slow2_kaggle_args.txt
tools/diffusion_model/config/tinymdm_vr_m3_1_spinkick.yaml    (+ bản _slow2)
kaggle/make_m3_dataset.sh                        kaggle/checkpoint_watchdog.py
tools/retime_motion.py
```

Cấu hình đúng, hạ tầng chạy được, prior train được, pipeline thông từ đầu tới video.

**Kết quả clip gốc: không hội tụ, và đã biết vì sao.** Run 216 M sample dừng ở `Ep_Len_Frac` 0.628
với `Sds_Loss` phẳng từ 137 M. Nguyên nhân là clip retarget đòi mô-men vượt giới hạn actuator ở
13/27 khớp (§4.1) — không phải lỗi cấu hình, không phải thiếu sample. **Đừng chạy lại clip gốc với
`--max_samples` cao hơn.**

**Việc tiếp theo:** thí nghiệm sàng lọc 30 M với clip giãn 2.0× (`*_slow2`), ~1 h 15 m GPU kể cả
prior. Baseline để so là `Ep_Len_Frac` **0.10 tại 30 M**; trên 0.30 thì chạy full, dưới 0.15 thì nhịp
không phải nút thắt.

**Nếu 2.0× không đủ:** phương án còn lại là nâng damping gối (kd 10 → 25, đưa ζ từ 0.64 lên ~1.6).
Đây là chỉnh tham số điều khiển của phần cứng thật, **phải được đội robot duyệt** và phải đồng bộ
ngược về `vr_m3_1_constants.py`, nếu không policy sẽ không transfer. Bảng mô-men ở §4.1 là lý lẽ bằng
số để mở cuộc trao đổi đó.

**Motion tiếp theo (zombie):** clip dài 1098 frame (~18 s) so với spinkick 78 frame (~1.3 s). Phân
phối chuyển động rộng hơn nhiều nên prior khó hơn. Chạy `retime_motion.py --report` trước tiên —
zombie chậm hơn spinkick nhiều nên nhiều khả năng khả thi sẵn, nhưng đừng đoán.
