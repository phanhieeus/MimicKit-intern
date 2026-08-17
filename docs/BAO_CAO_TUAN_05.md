# Báo cáo tuần 5 — MimicKit / SMP trên robot VR M3.1

**Người thực hiện:** phanhieeus
**Khoảng thời gian:** 12–14/08/2026 (phần train tập trung), tổng hợp 17/08/2026
**Repo:** `MimicKit-intern`, nhánh `main`
**WandB:** `phanhieupkkq-vietnam-national-university-hanoi/mimickit-smp`

---

## 1. Mục tiêu tuần

Đưa phương pháp **SMP (Score-Matching Motion Priors)** của MimicKit từ nhân vật
`humanoid` mẫu sang robot thật **VR M3.1 (27 DOF)**, chạy được trên Kaggle, và trả
lời câu hỏi: *M3.1 có bắt chước được motion người không, và nếu không thì vướng ở đâu.*

Kết quả ngắn gọn: **có, nhưng chỉ với clip không có pha bay và đủ ngắn.** Một trong
ba clip đã học thành công; hai clip còn lại thất bại vì hai nguyên nhân khác hẳn nhau,
và cả hai nguyên nhân đều đã được định lượng chứ không phải phỏng đoán.

---

## 2. Tìm hiểu MimicKit

### 2.1 Cấu trúc kế thừa của môi trường

```
CharEnv  →  DeepMimicEnv  →  AMPEnv  →  SMPEnv
```

Đây là chi tiết quan trọng nhất và cũng dễ hiểu nhầm nhất: **SMP dùng lại toàn bộ mã
kết thúc episode của DeepMimic**, dù về mặt thuật toán nó không liên quan gì tới
DeepMimic.

- `amp_env.py:249` — `AMPEnv._update_done` gọi thẳng `deepmimic_env.compute_done`
  nhưng ép `motion_len_term=False`. Hệ quả: episode chỉ kết thúc khi **ngã** hoặc
  **hết 10 giây**, không bao giờ kết thúc khi clip chạy hết. Với clip 1.3 s thì một
  episode 10 s tương đương 7.7 chu kỳ.
- `deepmimic_env.py:743-749` — phát hiện ngã bằng lực tiếp xúc: `|force| > 0.1 N`
  trên bất kỳ body nào **không** nằm trong `contact_bodies`. Đây là ngưỡng rất nhạy,
  chạm nhẹ đầu gối xuống đất là fail.
- `amp_env.py:186` — `_update_ref_motion` bị chặn sau `_enable_ref_char()`, tức là
  `_visualize and _visualize_ref_char`. Khi chạy headless (mọi rollout trên Kaggle),
  buffer của nhân vật tham chiếu **không bao giờ được cập nhật**.

Điểm cuối gây ra một lỗi thật: video `reference_sim_*.mp4` xuất ra luôn là một tư thế
đóng băng, và mất cả một lượt render để tạo ra video của một bức tượng. Đã sửa ở
commit `9e06a72` (`tools/play_policy_to_mp4.py` từ chối ghi reference khi headless).

### 2.2 Vật lý và tham số PD

`newton_engine.py:837-848` chép trực tiếp `stiffness` / `damping` trong MJCF sang
`joint_target_ke` / `joint_target_kd`. Nghĩa là **gain trong file MJCF chính là gain
của bộ điều khiển PD**, không có lớp chuyển đổi nào ở giữa. Với M3.1 các giá trị này
được lấy nguyên từ `vr_m3_1_constants.py` của mjlab, nên chúng là số liệu vật lý thật
của robot — không được chỉnh để cho dễ train.

`newton_engine.py:1005` chỉ thêm một mặt phẳng nền phẳng. **MimicKit không có
heightfield/terrain**; chỉ hai cấu hình dùng `static_objects` (`vault_box.xml`,
`climbing_box.xml`). Đây là câu trả lời cho câu hỏi "data gốc có terrain không": không có,
chỉ có motion clip trên nền phẳng.

### 2.3 Thứ tự ưu tiên tham số

`run.py:24` nạp command line **trước**, rồi mới tới arg file; `load_args` chỉ ghi
`if curr_key not in self._table`. Vậy **cờ trên command line thắng** giá trị trong
file `.txt`. Điều này cần thiết để ghi đè `--out_dir` trên Kaggle mà không phải sửa file.

---

## 3. Phương pháp SMP

SMP gồm **hai giai đoạn tách rời**:

### Giai đoạn 1 — Huấn luyện prior (học có giám sát)

Một mô hình khuếch tán **TinyMDM** được train trên chính clip mục tiêu. Nó học phân
phối của các đoạn chuyển động hợp lệ. Đây là học có giám sát thuần túy, không có mô
phỏng vật lý, chạy vài chục phút.

Đầu ra: `output/smp_prior_<clip>/model.pt` (~22.7 MB) kèm 16 sample GIF để kiểm tra
mắt thường xem prior có học đúng không.

### Giai đoạn 2 — PPO với phần thưởng từ prior

Chính sách điều khiển được train bằng PPO, phần thưởng là:

```
reward = exp(-sds_loss_norm × sds_loss_scale)      # sds_loss_scale = 6
```

`sds_loss` là **Score Distillation Sampling loss**: prior chấm điểm xem đoạn chuyển
động mà chính sách vừa tạo ra có giống phân phối nó đã học không. Loss thấp → phần
thưởng cao.

### 3.1 Quan sát của discriminator

Cửa sổ quan sát là `num_disc_obs_steps = 10` bước, cách nhau `1/control_freq`, tức
**0.333 giây thời gian thực** — không phải 10 khung hình của clip. Với
`disc_dof_vel_obs: False`, vận tốc khớp chỉ vào gián tiếp qua sai khác vị trí giữa các
bước.

### 3.2 Vì sao SMP có thể sập mode (mode collapse)

Đây là phát hiện trung tâm của tuần. Phần thưởng SMP **không phạt việc đứng yên**.
Nếu clip khó, phép tính lợi ích như sau (số thật từ run spinkick M3.1):

| Chiến lược | Reward/bước | Số bước sống | Tổng return |
|---|---|---|---|
| Đứng thăng bằng, không đá | 0.29 | 300 (hết giờ) | **87** |
| Đá rồi ngã | 0.45 | 40 | **18** |

Đứng yên thắng gấp **5 lần**. PPO tìm ra điều này rất nhanh và không bao giờ rời khỏi
đó. Ở mốc 26 M sample, chính sách đã chạy 218 000 episode trung bình ~30 bước cho một
clip dài 38 bước — **chưa một lần nào hoàn thành trọn một chu kỳ động tác**.

Hệ quả thực tiễn: **`Sds_Loss` và `Ep_Len_Frac` không phân biệt được bắt chước với sập
mode.** Run spinkick M3.1 kết thúc với `Ep_Len_Frac = 0.827` và `Sds_Loss = 0.170`
(thấp hơn cả humanoid đã hội tụ) trong khi nó chỉ đứng một chân và không hề đá.

---

## 4. Công cụ đã xây dựng

Toàn bộ nằm trong `tools/` và `kaggle/`, đã commit.

### 4.1 `tools/motion_quality.py` — chấm điểm rollout

Ra đời vì các đường cong training không nói được sự thật (mục 3.2). Hai chỉ số cốt lõi:

```python
fidelity = dist.min(axis=1).mean()   # tư thế policy → tư thế tham chiếu gần nhất
coverage = dist.min(axis=0).mean()   # tư thế tham chiếu → tư thế policy gần nhất
```

**Cả hai là khoảng cách — thấp mới tốt.** `--min_coverage 0.55` là ngưỡng **trần**:
`coverage > 0.55` ⟹ báo MODE COLLAPSE.

Sự bất đối xứng chính là điểm mấu chốt: một chính sách giữ nguyên một tư thế đạt
`fidelity` gần như hoàn hảo (tư thế đó *có* nằm trong clip) nhưng `coverage` rất tệ
(phần lớn clip không bao giờ được ghé tới). Đây đúng là dấu vân tay của sập mode.

Các chỉ số phụ: `periodicity` (≥ 0.45), `tempo` (1.0 ± 0.3), `speed_amplitude` (> 0.5),
`burstiness` (so với clip gốc — cú đá thì "bựt", rung lắc thì phẳng).

Một chi tiết kỹ thuật đã phải sửa: periodicity lấy **đỉnh trên một dải độ trễ**, không
phải một độ trễ chính xác. Clip 78 khung @60 fps ứng với 38.5 khung @30 fps; một chính
sách bắt chước tốt chấm **−0.01** ở lag 38 nhưng **0.86** ở lag 39.

### 4.2 `tools/retime_motion.py` — kiểm tra khả thi trước khi train

Ba việc: kiểm tra mô-men khớp có vượt giới hạn không, phát hiện **pha bay**, và
`--scan` để xếp hạng cả thư viện clip.

```python
implied_jump = 9.81 * seconds ** 2 / 8.0
```

Ngưỡng: nhảy suy ra > 1.0 m ⟹ clip thực chất là địa hình (`TERRAIN?`), > 0.08 m ⟹ có
nhảy thật (`JUMPS`). Phân biệt hai thứ này là commit `7f348bd`.

### 4.3 `tools/csv_to_motion.py` — chuyển CSV đã retarget thành clip MimicKit

Ánh xạ khớp theo tên với bảng bí danh (`_knee_joint` ↔ `_knee_pitch_joint`, …), tự dò
đơn vị và thứ tự Euler (chấm điểm 12 quy ước bằng FK theo tiêu chí bàn chân chạm sàn).
Có `--limit_fix {report,shift,clamp}`, `--ground_offset`, `--fold_waist`.

Căn nền dùng **điểm thấp nhất của geom va chạm**, không dùng gốc body — sai lệch giữa
hai cách là ~6 cm, đủ để chân lún xuống đất.

### 4.4 Hạ tầng Kaggle

- `kaggle/prior_cache.py` — cache prior thành WandB artifact, vân tay băm từ **nội dung**
  config + bytes của clip (không phải đường dẫn), nên đổi chỗ file không làm hỏng cache.
- `kaggle/checkpoint_watchdog.py` — cứ 20 phút đẩy `model.pt` lên WandB. Đây là lưới an
  toàn duy nhất: `/kaggle/working` chỉ tồn tại khi session kết thúc sạch.
- `kaggle/make_videos.py` — rollout → render → upload → chấm điểm, mỗi khâu đều kiểm tra
  và fail thẳng thay vì bỏ qua âm thầm.

---

## 5. Kết quả huấn luyện

### 5.1 Các clip đã dùng

| Clip | Khung hình | FPS | Thời lượng | Ghi chú |
|---|---|---|---|---|
| `vr_m3_1_humanoid_spinkick` | 78 | 60 | 1.30 s | có pha bay 0.47 s |
| `vr_m3_1_humanoid_zombie_walk` | 102 | 60 | 1.70 s | không pha bay |
| `vr_m3_1_dance_what` | 513 | 30 | 17.10 s | từ CSV retarget, loop |

### 5.2 Chỉ số training

Cấu hình chung: 1024 env × 2 GPU T4, `steps_per_iter 32` ⟹ 65 536 sample/iteration,
`episode_length 10 s`, `sds_loss_scale 6`.

| Run | ID | Samples | Giờ | `Ep_Len_Frac` | `Fail_Frac` | `Sds_Loss` | Test ep |
|---|---|---|---|---|---|---|---|
| humanoid spinkick (đối chứng) | `m6rv7ht3` | 350 M | 7.45 | 0.994 | 0.063 | 0.186 | 300.0 |
| **M3.1 spinkick** | `wjw4wwo3` | 320 M | 9.94 | 0.827 | 0.708 | 0.170 | 252.9 |
| M3.1 spinkick chậm 2× | `gb40wf35` | 30 M | 0.97 | 0.114 | **1.000** | 0.668 | 36.5 |
| **M3.1 zombie_walk** | `fdktxeb7` | 60 M | 1.83 | 0.878 | 0.426 | 0.230 | 296.9 |
| **M3.1 dance_what** | `261u7tka` | 120 M | 4.07 | 0.732 | 0.391 | 0.152 | 226.7 |

Tổng thời gian GPU của tuần: **≈ 26 giờ** (chưa kể prior).

### 5.3 Chấm điểm chất lượng — đây mới là bảng quyết định

| Chỉ số | Ngưỡng | spinkick 320M | zombie_walk | dance_what |
|---|---|---|---|---|
| **Coverage** | ≤ 0.55 | 0.893 ✗ | **0.526 ✓** | 0.853 ✗ |
| Fidelity | — | 0.080 | 0.486 | 0.163 |
| Periodicity | ≥ 0.45 | 0.135 ✗ | **0.561 ✓** | 0.002 ✗ |
| Tempo | 1.0 ± 0.3 | 0.184 ✗ | 0.580 ✗ | 0.201 ✗ |
| Speed amplitude | > 0.5 | 0.251 ✗ | **0.916 ✓** | 0.092 ✗ |
| Burstiness (vs clip) | khớp | 0.297 / 0.420 | **0.232 / 0.231** | 0.055 / 0.355 |
| **Verdict** | | sập mode | **học được** | sập mode |

Đọc bảng này:

- **spinkick 320 M**: `Fidelity 0.080` cực tốt đi kèm `Coverage 0.893` cực tệ — chữ ký
  sách giáo khoa của sập mode. Nó đứng đúng một tư thế lấy từ clip và không đi đâu cả.
  Ghi nhớ: run này có `Sds_Loss` **thấp hơn** cả humanoid đã hội tụ.
- **zombie_walk**: đạt 3/4 tiêu chí chính. Burstiness 0.232 so với clip gốc 0.231 —
  khớp gần như tuyệt đối, nghĩa là nhịp bên trong mỗi chu kỳ đúng. Trượt duy nhất ở
  **tempo 0.58**: policy đi chậm hơn clip 42 %, tức nó kéo giãn đều toàn bộ động tác
  chứ không làm sai động tác.
- **dance_what**: `Periodicity 0.002` và `Speed_Amplitude 0.092` — hầu như đứng yên.
  Sập mode nặng hơn cả spinkick.

**Kết luận: 1/3 clip thành công.** `zombie_walk` là chính sách M3.1 tốt nhất có được
cho tới nay, và là bằng chứng rằng đường ống SMP → M3.1 hoạt động.

---

## 6. Phân tích hai thất bại

### 6.1 spinkick — clip vượt quá khả năng vật lý của robot

Đo bằng `tools/retime_motion.py`:

- Pha bay **0.47 s** (31/78 khung hình, hai chân đều rời đất)
- Gốc thân nâng **40 cm**, chân đá lên **1.8 m**
- Suy ra cú nhảy **27 cm** cho robot **57 kg** vốn đã võng gối **24°** khi đứng yên

Đây không phải vấn đề của thuật toán học. Robot không nhảy được 27 cm.

**Vì sao làm chậm lại không cứu được:** thời gian bay `t` cho chiều cao `g·t²/8`. Kéo
dài clip hệ số `s` thì đòi chiều cao nhảy gấp **s²**. Chậm 2× ⟹ cần nhảy **108 cm**.
Đúng như dự đoán, run `slow2` **tệ hơn** bản gốc: `Fail_Frac = 1.000` liên tục.

Điểm đáng chú ý ở `slow2`: `Coverage 0.386` — **tốt hơn cả zombie_walk**. Nó *có* thử
làm đúng động tác, chỉ là ngã mọi lần. Nhưng `Burstiness 4.19` so với clip gốc `0.42`
cho thấy chuyển động giật cục gấp 10 lần bình thường: đó là robot đang giãy, không phải
đang đá. Tôi đã dừng run này sớm dựa trên `Sds_Loss` mà không xem chỉ số chất lượng —
một sai sót về quy trình, xem mục 8.

### 6.2 dance_what — clip quá dài so với ngân sách khám phá

Clip dài **17.1 giây = 512 bước**, trong khi episode tối đa chỉ 10 giây = 300 bước.
Chính sách **về nguyên tắc không thể** hoàn thành một chu kỳ trong một episode.

Ở mốc 32.8 M sample, độ dài episode train là 52.9 bước = 1.76 s trên tổng 17.1 s. Dấu
hiệu ban đầu có vẻ khả quan (`Fail_Frac` rời 1.0 ở 32.8 M, sớm hơn spinkick vốn giữ 1.0
tới 46 M; test ep dài gấp 2.5 lần train ep). Nhưng đến 120 M nó vẫn hội tụ về đứng yên.

Chẩn đoán: với clip 17 s, tín hiệu phần thưởng ở giai đoạn cuối động tác **không bao
giờ tới được** chính sách, nên nó tối ưu cho đoạn duy nhất nó chạm tới — và đoạn đó là
"đứng yên cho hết giờ".

**Hướng xử lý (chưa làm):** cắt clip xuống **một chu kỳ nhảy** rồi train lại. Không
phải tăng ngân sách sample — 120 M đã dư.

### 6.3 Xử lý dữ liệu retarget cho dance_what

Ba vấn đề đã sửa trên clip từ CSV:

1. **Ánh xạ khớp** — ban đầu tôi kết luận CSV không dành cho M3.1. Sai; sau khi đổi tên
   theo bảng bí danh thì **27/27 khớp ánh xạ được**.
2. **Giới hạn khớp vai** — thay vì cắt (`clamp`) làm mất biên độ, đã **dịch** cả dải
   khớp vào trong giới hạn (`--limit_fix shift`).
3. **Chân lún đất** — do căn theo gốc body cổ chân thay vì đế giày. Sửa xong, offset
   đổi từ −4.3 cm thành **+3.4 cm**.

**Hai DOF thắt lưng bị thiếu — không khôi phục được.** Phép gập toán học
`root' = root @ C`, `H' = Cᵀ @ H` chính xác về **hướng**, nhưng trục hông của M3.1 cách
nhau **0, 6 và 29 cm**, nên phép quay bù giữ được hướng chân mà không giữ được vị trí
bàn chân — chân lơ lửng 8 cm. Muốn làm đúng phải giải IK toàn chân. Đã ghi lại giới hạn
này ở commit `f957768` thay vì để lại một tính năng nhìn thì chạy mà thực ra sai.

---

## 7. Vị trí lưu trữ hiện tại

Đã tải về máy (`output/`, đã gitignore):

```
output/smp_m3_zombie_walk/
├── model.pt                   28.8 MB   checkpoint v4 (OrderedDict, 121 tensor,
│                                        10 020 151 tham số, obs 229-d, action 27-d)
└── videos/  policy_final.mp4, reference_data.mp4

output/smp_prior_vr_m3_1_zombie_walk/
├── model.pt                   22.7 MB
├── tinymdm_vr_m3_1_zombie_walk.yaml
└── samples/                   16 GIF + 16 pkl
```

**Cảnh báo:** checkpoint v4 được ghi lúc 01:53 Z còn train kết thúc ~02:04 Z, nên nó
**sớm hơn bản cuối ~11 phút (≈ 6 M sample)**. `make_videos.py` chỉ upload mp4 chứ không
upload checkpoint, nên `model.pt` cuối cùng chỉ còn trong output của notebook Kaggle.
Đây là một lỗ hổng cần vá.

Trên WandB: 5 phiên bản checkpoint (`smp_m3_zombie_walk_ckpt_model:v0…v4`), prior
(`smp_prior_vr_m3_1_zombie_walk:v0`), video và các run chấm điểm.

---

## 8. Bài học rút ra

### Về kỹ thuật

1. **Kiểm tra khả thi vật lý trước khi train, không phải sau.** Một lệnh
   `retime_motion.py --scan` mất vài phút và đã có thể tiết kiệm ~11 giờ GPU của
   spinkick + slow2.
2. **Không tin đường cong training với SMP.** `Sds_Loss` và `Ep_Len_Frac` đều thưởng
   cho việc đứng yên. Chỉ `Coverage` phân biệt được.
3. **Độ dài clip phải nằm gọn trong `episode_length`.** Đây là ràng buộc cứng, không
   phải chuyện điều chỉnh siêu tham số.
4. Triage thư viện clip: 267 clip → 98 clip tiếp đất hoàn toàn → 24 clip đồng thời nằm
   trong giới hạn mô-men chân. `zombie_walk` xếp hạng 2 (không pha bay, 0.28× giới hạn
   mô-men).

### Về quy trình — những chỗ tôi đã làm sai

Ghi lại đầy đủ vì chúng ảnh hưởng tới kết luận:

- **Dừng `slow2` sớm dựa trên `Sds_Loss`** mà không chạy chấm điểm chất lượng. Coverage
  của nó là 0.386, tốt hơn zombie_walk. Kết luận "slow2 không cải thiện gì" là sai.
- **Đọc ngược chiều `Coverage`** trong một lần báo cáo, dù chính tôi viết metric đó.
- **So sánh M3.1 với humanoid khi hai bên dùng prior khác nhau** — không so được.
- **Kết luận "21× sàn nghĩa là không học được"**: humanoid thành công ở **34.4×** còn
  M3 thất bại ở **21.5×**, tức con số này không dự báo được gì.
- **Đổ lỗi cho khoảng hở ống chân**: đo lại thì zombie_walk +6.3 cm và spinkick +6.4 cm
  — giống hệt nhau.
- **Để một lệnh `find /` chạy nền** làm bão hòa I/O ổ NTFS khiến mọi tiến trình Python
  bị timeout, và mất thời gian chẩn đoán nhầm.
- **Một `\` xuống dòng bị thiếu** gây `IndentationError` và giết một batch session sau
  47 phút đã train prior.

Điểm chung của phần lớn các lỗi trên: kết luận trước khi đo. Công cụ ở mục 4 tồn tại
chính là để bắt buộc phải đo.

---

## 9. Việc cho tuần sau

**Ưu tiên cao**

1. Cắt `dance_what` xuống một chu kỳ nhảy (≤ 8 s) và train lại — giả thuyết ở mục 6.2
   cần được kiểm chứng hoặc bác bỏ.
2. Xử lý **tempo 0.58** của zombie_walk. Hai giả thuyết: (a) ngân sách 60 M chưa đủ để
   đạt tốc độ đầy đủ — train tiếp từ checkpoint; (b) mô-men khớp giới hạn tốc độ — đo
   bằng `retime_motion.py` trên chính rollout.
3. Vá lỗ hổng lưu trữ: cho `make_videos.py` upload luôn `model.pt` cuối cùng, để không
   phụ thuộc vào chu kỳ 20 phút của watchdog.

**Ưu tiên trung bình**

4. Thêm trọng lực vào mô hình mô-men của `retime_motion.py` (hiện chỉ tính quán tính).
5. Giải IK toàn chân để `--fold_waist` dùng được thật.
6. Chạy chấm điểm chất lượng cho run humanoid đối chứng, để có một mốc "đạt" thực sự
   thay vì chỉ có các mốc "trượt".

**Việc vặt**

7. `git prune` — `.git/gc.log` và các loose object không tham chiếu.
8. `.venv/bin/python3` là symlink hỏng (reparse tag NTFS); venv nên tạo lại trên ext4.

---

## 10. Phụ lục — các commit trong tuần

| Commit | Nội dung |
|---|---|
| `3f60f73` | Clip spinkick đòi mô-men mà M3.1 không có |
| `951be7a` | Watchdog để batch overrun không làm mất trọng số |
| `ff19f57` | Cache prior thay vì build lại mỗi session |
| `984e3d2` | Không xóa thư mục mà kernel đang đứng trong đó |
| `98f0ee1` | Render mà không ghi vào cây thư mục asset |
| `301e655` | Chấm điểm rollout, vì đường cong training không làm được |
| `432e8e0` | Giữ sample của prior cùng với prior |
| `0d69a06` | Sửa lại cơ chế mà `motion_quality.py` mô tả |
| `99fb2b8` | Cho `Sds_Loss` một cây thước |
| `4031736` | Khoảng cách tới sàn không dự báo được thành công |
| `e926bb0` | Kiểm tra pha bay trước khi retime, và chọn motion không có |
| `d3b85ab` | Tìm data pack theo hình dạng, không theo đường dẫn |
| `9e06a72` | Ngừng render video của một bức tượng |
| `7f348bd` | Phân biệt địa hình với nhảy |
| `0bef83e` | Chuyển CSV animation thành clip MimicKit |
| `a57dbc2` | Căn đế giày xuống sàn, và dịch khớp thay vì cắt |
| `f957768` | Gập thắt lưng vào chậu và hông, và ghi lại vì sao chưa đủ |
| `39157af` | Config cho clip dance |
