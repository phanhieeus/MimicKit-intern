# Train SMP spinkick cho VR M3.1 trên Kaggle

Cùng động tác, cùng phương pháp, cùng hạ tầng như lần chạy humanoid đã thành công — chỉ đổi robot.
Chọn spinkick làm motion đầu tiên cho M3.1 là có chủ ý: đã có một run humanoid hội tụ trên **đúng clip
này** để đối chiếu, nên nếu M3.1 không học được thì biết ngay vấn đề nằm ở robot chứ không ở motion,
prior, hay hyperparameter.

Nền tảng lý thuyết và cách đọc số: [SMP_PLAYBOOK.md](SMP_PLAYBOOK.md).
Bản humanoid: [HUONG_DAN_SMP_KAGGLE.md](HUONG_DAN_SMP_KAGGLE.md).

## Khác gì so với humanoid

| | Humanoid | M3.1 |
|---|---|---|
| Dof | 28 | **27** |
| `init_pose` | 34 số | **33 số**, root z = 0.854 m |
| Data đến từ | data pack 534 MB (Kaggle Dataset) | **Dataset thứ hai, phải tự tạo** |
| Prior | có sẵn `smp_prior_spinkick.pt` | **chưa có, phải train** |
| Clip | 78 frame @ 60 fps | bản retarget 78 frame, **phải giãn 2.0× → 155 frame** (xem Cell 4b) |

Ba dòng in đậm cuối là toàn bộ công việc phát sinh: một Dataset, một lần train prior, và một bước
giãn clip cho vừa khả năng actuator.

---

## Bước 0 — Tạo Kaggle Dataset cho M3.1 (làm ở máy local, một lần)

`data/assets/vr_m3_1` và `data/motions/vr_m3_1` đều bị gitignore (`data/assets/.gitignore` và
`data/motions/.gitignore` đều là `*`). Đúng — 97 MB STL và pickle không nên nằm trong git — nhưng
nghĩa là clone trên Kaggle sẽ có config mà không có data chúng trỏ tới.

```bash
bash kaggle/make_m3_dataset.sh            # chỉ spinkick, ~70 MB
# hoặc: bash kaggle/make_m3_dataset.sh --all-motions   # cả 266 clip, ~97 MB
```

Script dựng `kaggle_dataset_vr_m3_1/` đúng cấu trúc `prepare_data.py` nhận diện được
(có đồng thời `assets/` và `motions/`). Sau đó sửa `dataset-metadata.json` điền username Kaggle của
bạn rồi:

```bash
kaggle datasets create -p kaggle_dataset_vr_m3_1 --dir-mode zip
```

Hoặc upload thủ công ở kaggle.com/datasets/new. Ghi nhớ slug, cell 4 cần nó.

70 MB gần như toàn bộ là mesh STL — bắt buộc, MuJoCo cần chúng cả để dựng model lẫn để render video.
Clip spinkick chỉ 41 KB.

---

## Notebook — 10 cell

Accelerator: **GPU T4 x2**. P100 (sm_60) không chạy được, PyTorch trong image chỉ build cho sm_70+.
Chạy bằng **Save Version → Save & Run All (Commit)** để có 12 h thay vì 9 h.

```python
# Cell 1 — secrets
import os
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["WANDB_API_KEY"] = sec.get_secret("WANDB_API_KEY")
os.environ["GITHUB_TOKEN"]  = sec.get_secret("GITHUB_TOKEN")
os.environ["WANDB_PROJECT"] = "mimickit-smp"
os.environ["WANDB_NAME"]    = "smp_m3_spinkick_slow2"
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Output phải là hai dòng `Tesla T4`. Thấy `P100` thì đổi accelerator rồi restart, đừng chạy tiếp.

```python
# Cell 2 — clone
import os, shutil
REPO = "/kaggle/working/MimicKit-intern"
token = os.environ["GITHUB_TOKEN"]

# Step out of REPO before deleting it. On a re-run the kernel's cwd is still
# inside from the last time this cell ran, and rmtree then pulls the ground out
# from under the process: getcwd() starts failing, git clone dies with
# "Unable to read current working directory", and the cell finally raises
# FileNotFoundError on the os.chdir below -- three errors, one cause.
os.chdir("/kaggle/working")
shutil.rmtree(REPO, ignore_errors=True)
!git clone --depth 1 https://{token}@github.com/phanhieeus/MimicKit-intern.git {REPO}
os.chdir(REPO)
!git log --oneline -1
```

```python
# Cell 3 — dependencies (~4 phút)
!bash kaggle/setup.sh
```

```python
# Cell 4 — data: pack gốc + dataset M3.1
import os, subprocess

# Bắt buộc, phải trước prepare_data.py. link_tree() symlink CẢ thư mục nếu đích
# chưa tồn tại, và đích đó là /kaggle/input chỉ đọc -- Cell 4b sẽ không ghi được
# clip đã giãn vào đấy. Tạo trước thì từng file được symlink riêng, thư mục ghi
# được, và các file thật nằm cạnh symlink không sao cả.
os.makedirs("data/motions/vr_m3_1", exist_ok=True)

for name in sorted(os.listdir("/kaggle/input")):
    path = os.path.join("/kaggle/input", name)
    print("=== {} ===".format(path))
    subprocess.run(["python", "kaggle/prepare_data.py", "--input_root", path])

!ls data/assets/vr_m3_1/ && ls data/motions/vr_m3_1/ && ls data/motions/ | head
```

Phải thấy `vr_m3_1.xml`, thư mục `assets/`, `vr_m3_1_humanoid_spinkick.pkl`, và cả các robot của
pack gốc (`humanoid`, `g1`, `go2`…).

**Vì sao lặp qua từng dataset thay vì gọi hai lần cố định:** `find_source()` duyệt BFS từ
`--input_root` và trả về **pack đầu tiên** tìm thấy rồi dừng. Gọi `prepare_data.py` trần với
`/kaggle/input` chứa hai dataset thì nó chỉ link được một, và không có gì đảm bảo đó là cái nào —
thứ tự phụ thuộc tên thư mục. Ghim `--input_root` vào từng dataset một thì mỗi cái được xử lý đúng
một lần. `link_tree()` merge đệ quy và bỏ qua entry đã tồn tại nên gọi nhiều lần là an toàn; dataset
nào không phải data pack chỉ in ERROR rồi bỏ qua.

```python
# Cell 4b — giãn clip 2.0x cho vừa khả năng actuator (~5 giây)
!python tools/retime_motion.py \
    --input  data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick.pkl \
    --output data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick_slow2.pkl \
    --char_file data/assets/vr_m3_1/vr_m3_1.xml \
    --factor 2.0
```

Clip gốc đòi mô-men vượt giới hạn ở 13/27 khớp — xem
[Clip gốc bất khả thi về động lực học](#clip-gốc-bất-khả-thi-về-động-lực-học--đọc-phần-này-trước-khi-train).
Phải in ra `3 of 27 joints over limit` (chỉ còn nhóm vai) và `155 frames @ 60 fps = 2.567 s`.

Bỏ cell này nếu muốn tái lập baseline clip gốc; khi đó Cell 5–8 dùng bản config không có `_slow2`.

```python
# Cell 5 — prior: lấy từ cache nếu có, train nếu chưa (lần đầu ~35 phút, sau đó ~10 giây)
!python kaggle/prior_cache.py \
    --cfg_path tools/diffusion_model/config/tinymdm_vr_m3_1_spinkick_slow2.yaml \
    --out_dir /kaggle/working/output/smp_prior_vr_m3_1_spinkick_slow2 \
    --project mimickit-smp --device cuda:0

!ls -lh /kaggle/working/output/smp_prior_vr_m3_1_spinkick_slow2/
```

Bắt buộc, không bỏ qua được: prior duy nhất trong data pack là của humanoid 28 dof, không dùng cho
robot 27 dof. Giai đoạn này chỉ dùng một GPU — `train_tinymdm.py` là vòng lặp một tiến trình, không
có DDP.

`prior_cache.py` bọc quanh `train_tinymdm.py` và giải quyết hai chuyện:

**Không mất prior.** 35 phút GPU cho ra một file duy nhất, và trước đây file đó chỉ tồn tại trong
`/kaggle/working` — session chết là mất trắng. Giờ nó được publish thành artifact WandB
(`mimickit-smp/smp_prior_vr_m3_1_spinkick_slow2:latest`), và trong lúc train có watchdog đẩy bản dở
mỗi 5 phút, nên chết ở phút 30 vẫn còn thứ để dùng lại.

**Không đổi hàm reward giữa các run.** Đây mới là lý do chính. Prior **chính là** hàm reward — SMP
tính `exp(-sds_loss × scale)` dựa trên nó. Hai lần train cùng config không cho ra cùng một prior, nên
hai run tự train prior riêng thì **không so được với nhau**. Đúng chuyện đã xảy ra với hai run M3.1:
tại cùng 131.1 M sample, một cái ở `Ep_Len_Frac` 0.470, cái kia 0.288, không có khác biệt nào khác
giải thích được.

Artifact mang theo fingerprint băm từ nội dung config và clip. Prior cache có fingerprint không khớp
là prior cũ — script **từ chối dùng** thay vì im lặng tái sử dụng, vì im lặng đúng là cách tái tạo lại
lỗi trên. `--force_retrain` để train mới đè lên, `--accept_stale` nếu bạn biết mình đang làm gì.

Đường dẫn kết quả giống hệt nhau ở cả hai nhánh (cache hit hay train mới), nên Cell 6 bên dưới không
cần biết nhánh nào đã chạy.

```python
# Cell 6 — trỏ agent config vào prior vừa train
PRIOR = "/kaggle/working/output/smp_prior_vr_m3_1_spinkick_slow2/model.pt"
assert os.path.isfile(PRIOR), "prior chua duoc tao, xem lai Cell 5"

import re, pathlib
p = pathlib.Path("data/agents/smp_vr_m3_1_spinkick_slow2_agent.yaml")
p.write_text(re.sub(r'^smp_prior_model:.*$', f'smp_prior_model: "{PRIOR}"',
                    p.read_text(), flags=re.M))
!grep smp_prior data/agents/smp_vr_m3_1_spinkick_slow2_agent.yaml
```

```python
# Cell 7 — smoke test 2 phút
!python mimickit/run.py \
    --arg_file args/smp_vr_m3_1_spinkick_slow2_kaggle_args.txt \
    --mode train --num_envs 256 --max_samples 200000 \
    --logger txt --out_dir /kaggle/working/output/smoke \
    --devices cuda:0
```

Cell này bắt gần hết lỗi cấu hình trong 2 phút thay vì 7 tiếng: sáu assert của
`_check_prior_env_config`, số dof của `init_pose`, tên link trong `key_bodies` / `contact_bodies`,
đường dẫn prior. Chạy trót lọt vài iteration là qua.

Nhìn `Ep_Len_Frac` ở vài iteration đầu: nếu **đúng bằng 0** thì mọi episode chết ở frame một —
gần như chắc chắn là `init_pose` sai chiều cao hoặc `contact_bodies` sai tên link, chứ không phải
policy dở. Sửa trước khi chạy full.

```python
# Cell 8a — watchdog: đẩy checkpoint lên WandB mỗi 20 phút
import subprocess
wd = subprocess.Popen([
    "python", "kaggle/checkpoint_watchdog.py",
    "--model_file", "/kaggle/working/output/smp_m3_spinkick_slow2/model.pt",
    "--project", "mimickit-smp",
    "--run_name", "smp_m3_spinkick_slow2_ckpt",
    "--interval", "1200",
])
print("watchdog pid", wd.pid)
```

Cell này bảo hiểm cho tình huống hết giờ. `/kaggle/working` chỉ thành Output khi notebook kết thúc
sạch; một version batch chạy quá 12 h bị đánh dấu failed, và output của version failed thì không nên
trông cậy. Metric vẫn an toàn vì đã stream lên WandB, nhưng `model.pt` thì không — `run.py` không
upload gì trong lúc train. Watchdog chạy nền song song với cell train (cell không chờ subprocess đã
detach) và đẩy checkpoint thành artifact version mới mỗi 20 phút.

Mất session vẫn lấy lại được:

```python
!python kaggle/wandb_upload.py --project mimickit-smp \
    --download smp_m3_spinkick_ckpt_model:latest --dest /kaggle/working/recovered
```

```python
# Cell 8 — train policy: sàng lọc 30 M trước (~40 phút)
!python mimickit/run.py \
    --arg_file args/smp_vr_m3_1_spinkick_slow2_kaggle_args.txt \
    --max_samples 30000000
```

**Chạy 30 M trước, đừng đặt thẳng con số lớn.** Baseline clip gốc tại 30 M là `Ep_Len_Frac` **0.10**.
Nếu bản giãn vượt **0.30** thì giả thuyết nhịp đúng, lúc đó mới chạy full; dưới 0.15 thì nhịp không
phải nút thắt và tiền GPU nên để dành. Bảng đọc kết quả đầy đủ ở mục
[Clip gốc bất khả thi về động lực học](#clip-gốc-bất-khả-thi-về-động-lực-học--đọc-phần-này-trước-khi-train).

Mọi tham số còn lại nằm trong arg file (1024 env, hai GPU, `--save_int_models true`,
`--out_dir /kaggle/working/output/smp_m3_spinkick_slow2`), nên không cần lặp lại trên dòng lệnh.

**Chưa có con số `--max_samples` cho lần chạy full**, và đừng chép 200 M từ bản trước của tài liệu
này — nó là ngoại suy sai từ bốn mốc đầu. Chốt sau khi có kết quả 30 M.

Về tốc độ: M3.1 chạy **8 500–8 700 samples/s**, khoảng 65 % của humanoid (13 058), vì 27 dof, 30 body
và mesh STL thật. Nhanh hơn về sample nhưng chậm hơn về giây là hai chuyện ngược chiều, rất dễ lẫn.

Theo dõi **`Sds_Loss_Mean`**, không phải `Smp_Reward_Mean` — reward mang thang chia riêng của từng
run nên không so được, kể cả với chính run trước của cùng robot.

```python
# Cell 8b — tắt watchdog sau khi train xong
wd.terminate()
```

```python
# Cell 9 — video + upload
!python kaggle/make_videos.py \
    --out_dir      /kaggle/working/output/smp_m3_spinkick_slow2 \
    --env_config   data/envs/smp_vr_m3_1_spinkick_slow2_env.yaml \
    --agent_config data/agents/smp_vr_m3_1_spinkick_slow2_agent.yaml \
    --char_file    data/assets/vr_m3_1/vr_m3_1.xml \
    --motion_file  data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick_slow2.pkl \
    --wandb_project mimickit-smp \
    --steps 300

!python kaggle/wandb_upload.py --project mimickit-smp \
    --run_name smp_m3_spinkick_slow2_files \
    --files /kaggle/working/output/smp_m3_spinkick_slow2/model.pt \
            /kaggle/working/output/smp_m3_spinkick_slow2/log.txt
```

`--char_file` **phải** là MJCF của M3.1, mặc định của script là humanoid. Chỉ xem
`policy_final.mp4` và `reference_data.mp4`; `reference_sim_final.mp4` hỏng (đứng yên một tư thế),
lý do ở [SMP_PLAYBOOK.md §7](SMP_PLAYBOOK.md).

---

## Clip gốc bất khả thi về động lực học — đọc phần này trước khi train

Run `wjw4wwo3` chạy hết 216 M sample chỉ để chứng minh một điều: **không phải thiếu sample.**

Retarget chỉ khớp *tư thế*. Không có bước nào trong pipeline đó biết giới hạn actuator của robot, nên
một clip nhìn từng khung thì đúng vẫn có thể đòi mô-men mà phần cứng không sinh nổi. Đo trên chính
MJCF của M3.1 — quán tính hợp thành quanh từng trục khớp, `τ = I_eff · q̈`, so với `actuatorfrcrange`:

| khớp | τ cần | giới hạn | vượt |
|---|---|---|---|
| `left_shoulder_pitch` | 782 N·m | 66 | **11.8×** |
| `waist_yaw` | 389 N·m | 102 | **3.8×** |
| `right_hip_roll` | 1265 N·m | 360 | **3.5×** |
| `right_hip_pitch` | 884 N·m | 360 | **2.5×** |
| `left_hip_roll` | 514 N·m | 360 | 1.4× |

**13 trên 27 khớp vượt giới hạn**, gồm cả nhóm quyết định thăng bằng. Đây không phải nhiễu vi phân
số: chỉ 0.9–3.8 % năng lượng của clip nằm trên 8 Hz. Nó là chuyển động thật, tần số thấp — cú spinkick
1.28 s xoay trọn 397° mà người làm được còn robot này thì không.

Tự đo lại bất cứ clip nào:

```bash
python tools/retime_motion.py --input <clip>.pkl --char_file <robot>.xml --report
```

### Bằng chứng: bài toán không nằm ở imitation

So tại thời điểm humanoid *đã giải xong* thăng bằng:

| | humanoid @ 52 M | M3.1 @ 216 M |
|---|---|---|
| `Ep_Len_Frac` | **0.965** | 0.628 |
| `Sds_Loss_Mean` | 0.321 | **0.260** |

M3.1 bám tư thế **tốt hơn** humanoid mà vẫn ngã. Mạng biết phải làm gì; nó chỉ luôn trễ nhịp, và trễ
nhịp lúc trụ một chân thì đổ. Thêm sample không mua được thứ actuator không có.

Diễn biến của chính run đó nói rõ hai mục tiêu tách nhau từ 140 M:

```
iter 1200    78.6 M   EpLen 0.366   Sds 0.3008
iter 2100   137.6 M   EpLen 0.481   Sds 0.2582   <- Sds chạm đáy
iter 3300   216.3 M   EpLen 0.628   Sds 0.2597   <- 80 M sau, y nguyên
```

Từ 140 M trở đi không còn học chất lượng động tác, chỉ mua thêm thăng bằng — với giá rất đắt.

### Cách xử lý: giãn clip, không đụng robot

τ tỉ lệ `1/s²`, nên giãn thời gian hệ số `s` giảm mô-men cần theo bình phương. Ở **2.0×** mọi khớp
chân và thắt lưng vào trong giới hạn, chỉ còn bộ ba vai trái vượt (2.96/1.58/1.54×) — vai bão hòa
không làm robot ngã, hông bão hòa thì có. Tuân thủ toàn bộ cần 3.44×, chậm tới mức không còn ra cú đá.

```bash
python tools/retime_motion.py \
    --input  data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick.pkl \
    --output data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick_slow2.pkl \
    --char_file data/assets/vr_m3_1/vr_m3_1.xml --factor 2.0
```

`--auto` tự chọn `s` từ nhóm khớp còn lại sau `--exclude` (mặc định bỏ qua tay).

Việc này **không thay đổi một byte vật lý nào**. Đây là điểm quan trọng: `stiffness` / `damping` trong
MJCF là kp/kd của vòng PD (`newton_engine.py:837-848` copy chúng vào `joint_target_ke/kd`), chép từ
`vr_m3_1_constants.py` của mjlab, tức tham số điều khiển của phần cứng thật. Chỉnh chúng để train dễ
hơn là làm hỏng tính hợp lệ sim-to-real và cần đội phần cứng duyệt. Giãn clip thì không — robot y
nguyên, chỉ bảo nó đá chậm lại.

Bộ config `*_slow2` đã dựng sẵn và chỉ khác baseline đúng một dòng `motion_file`:

| | file |
|---|---|
| env | `data/envs/smp_vr_m3_1_spinkick_slow2_env.yaml` |
| agent | `data/agents/smp_vr_m3_1_spinkick_slow2_agent.yaml` |
| prior | `tools/diffusion_model/config/tinymdm_vr_m3_1_spinkick_slow2.yaml` |
| args | `args/smp_vr_m3_1_spinkick_slow2_kaggle_args.txt` |

Ba điều bắt buộc khi dùng chúng:

1. **Cell 4 phải `os.makedirs("data/motions/vr_m3_1", exist_ok=True)` trước `prepare_data.py`.**
   `link_tree()` symlink cả thư mục nếu đích chưa tồn tại, và thư mục đó trỏ vào `/kaggle/input` chỉ
   đọc — bước giãn clip sẽ báo `Permission denied`. Tạo trước thì từng file được symlink riêng và thư
   mục ghi được.
2. **Train lại prior.** Prior học nhịp của clip; dùng prior cũ là chấm điểm nhịp mới bằng thước cũ.
3. **Chạy sàng lọc 30 M trước.** `Ep_Len_Frac` so trực tiếp được với baseline vì `amp_env.py:252` ép
   `motion_len_term` False — episode kết thúc do ngã hoặc hết 10 s, không theo độ dài clip.

Mốc để đọc kết quả, baseline là **0.10 tại 30 M**:

| `Ep_Len_Frac` @ 30 M | kết luận |
|---|---|
| > 0.30 | nhịp đúng là nút thắt, chạy full |
| 0.15–0.30 | có tác dụng nhưng chưa đủ, thử 2.5× |
| < 0.15 | nhịp không phải nút thắt; bảng mô-men ở trên là lý lẽ bằng số để bàn với đội phần cứng |

Chi phí ~1 h 15 m GPU: prior 35 phút + RL 40 phút.

### Ngân sách sample

Chỉ chốt được sau khi thí nghiệm giãn clip cho kết quả. Con số duy nhất đã đo chắc chắn là của
humanoid: giải xong thăng bằng ở **52 M**, `Sds_Loss` còn giảm tới ~310 M, tổng 350 M.

Với clip gốc thì M3.1 tiệm cận ở đâu đó dưới 0.85 và không bao giờ tới — đừng chạy lại nó với
`--max_samples` cao hơn. Sửa clip trước, đo lại, rồi mới định ngân sách.

### Đừng ngoại suy tuyến tính từ `Ep_Len_Frac` sớm

Bài học đắt nhất của lần này. Từ bốn mốc đầu tôi kết luận "M3.1 hiệu quả gấp đôi humanoid" rồi
"M3.1 chỉ cần 200 M". Cả hai đều sai, vì humanoid không tăng tuyến tính — nó bò ở mức thấp rồi bật
lên gần thẳng đứng quanh 52 M, còn M3.1 tăng đều rồi từ từ chậm lại. Hai hình dạng đường cong khác
hẳn nhau, và ba điểm đầu không phân biệt được chúng.

Chỉ so ngân sách sau khi cả hai run đã vượt điểm bật, không bao giờ trước.

### Hai thứ KHÔNG so được giữa hai robot

**`Smp_Reward_Mean`** — mỗi run một `Sds_Norm_Scale` riêng nên reward không so được, kể cả giữa hai
run của cùng robot. Dùng `Sds_Loss_Mean`: nó là đại lượng tuyệt đối và chính nó cho bảng đối chiếu ở
trên. (Chỉ nên so khi hai prior có cùng không gian obs — cùng clip, cùng `num_disc_obs_steps` — và
hiểu rằng đó là so xấp xỉ.)

**Thời gian.** M3.1 chạy **8 500–8 700 samples/s**, khoảng 65 % của humanoid (13 058) vì 27 dof,
30 body và mesh STL thật. Đo trên toàn bộ run 216 M — con số 9 000–9 300 ở bản trước là ước lượng từ
vài trăm iteration đầu, khi episode còn ngắn nên reset nhiều và rẻ.

### Cảnh báo về credit

Quota GPU của Kaggle tính theo giờ đồng hồ, không theo sample. Với M3.1 thì một session 320 M ăn
**~10 h 30 m** quota — gần hết hạn mức tuần nếu bạn đã dùng trước đó. Tính credit còn lại **trước**
khi bấm Save & Run All, và luôn bật Cell 8a watchdog: hết quota giữa chừng thì version bị đánh dấu
failed và output không được lưu, tức mất sạch `model.pt` dù mọi metric đã lên WandB.

## Đã kiểm sẵn cho clip này

Forward kinematics (`MJCFCharModel`) qua toàn bộ frame của **cả hai** bản clip:

| body thấp nhất | clip gốc (78 frame) | bản giãn 2.0× (155 frame) |
|---|---|---|
| `left_ankle_pitch_link` | +4.43 cm | +4.43 cm |
| `left_ankle_roll_link` | +4.43 cm | +4.43 cm |
| `right_ankle_pitch_link` | +5.06 cm | +5.06 cm |

Cổ chân chạm thấp nhất ở +4.4 cm — đúng bằng độ dày đế, tức bàn chân có đạp đất thật. Body thấp kế
tiếp là đầu gối, không bao giờ xuống dưới +38.8 cm.

Nghĩa là `contact_bodies: [right_ankle_roll_link, left_ankle_roll_link]` an toàn với biên rất rộng:
clip tham chiếu không bao giờ tự vi phạm kiểm tra ngã. Không cần chạy
`tools/fix_ground_penetration.py` cho clip nào trong hai clip.

Giãn thời gian không đổi con số nào ở trên, đúng như thiết kế: `retime_motion.py` chỉ **lấy mẫu lại
theo thời gian**, không đụng tới tư thế. Khung chẵn tái tạo bản gốc chính xác tới 4.4e-16, khung lẻ
là nội suy giữa hai khung liền kề (slerp cho xoay gốc, tuyến tính cho góc khớp), nên không tư thế mới
nào được sinh ra ngoài bao lồi của clip gốc. Biên độ dof, chiều cao root và tổng góc xoay 397° đều
giữ nguyên.
