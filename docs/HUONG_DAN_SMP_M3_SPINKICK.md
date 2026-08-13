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
| Clip | 78 frame @ 60 fps | 78 frame @ 60 fps (bản retarget) |

Hai dòng in đậm cuối là toàn bộ công việc phát sinh: một Dataset và một lần train prior.

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

## Notebook — 9 cell

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
os.environ["WANDB_NAME"]    = "smp_m3_spinkick"
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Output phải là hai dòng `Tesla T4`. Thấy `P100` thì đổi accelerator rồi restart, đừng chạy tiếp.

```python
# Cell 2 — clone
import os, shutil
REPO = "/kaggle/working/MimicKit-intern"
token = os.environ["GITHUB_TOKEN"]
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
# Cell 5 — train prior TinyMDM (~30-40 phút, 1 GPU)
!python tools/diffusion_model/train_tinymdm.py \
    --cfg_path tools/diffusion_model/config/tinymdm_vr_m3_1_spinkick.yaml \
    --out_dir /kaggle/working/output/smp_prior_vr_m3_1_spinkick \
    --device cuda:0

!ls -lh /kaggle/working/output/smp_prior_vr_m3_1_spinkick/
```

Bắt buộc, không bỏ qua được: prior duy nhất trong data pack là của humanoid 28 dof, không dùng cho
robot 27 dof. Giai đoạn này chỉ dùng một GPU — `train_tinymdm.py` là vòng lặp một tiến trình, không
có DDP.

```python
# Cell 6 — trỏ agent config vào prior vừa train
PRIOR = "/kaggle/working/output/smp_prior_vr_m3_1_spinkick/model.pt"
assert os.path.isfile(PRIOR), "prior chua duoc tao, xem lai Cell 5"

import re, pathlib
p = pathlib.Path("data/agents/smp_vr_m3_1_spinkick_agent.yaml")
p.write_text(re.sub(r'^smp_prior_model:.*$', f'smp_prior_model: "{PRIOR}"',
                    p.read_text(), flags=re.M))
!grep smp_prior data/agents/smp_vr_m3_1_spinkick_agent.yaml
```

```python
# Cell 7 — smoke test 2 phút
!python mimickit/run.py \
    --arg_file args/smp_vr_m3_1_spinkick_kaggle_args.txt \
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
    "--model_file", "/kaggle/working/output/smp_m3_spinkick/model.pt",
    "--project", "mimickit-smp",
    "--run_name", "smp_m3_spinkick_ckpt",
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
# Cell 8 — train policy (~9h35m ở tốc độ đo được của M3.1)
!python mimickit/run.py \
    --arg_file args/smp_vr_m3_1_spinkick_kaggle_args.txt \
    --mode train \
    --num_envs 1024 \
    --max_samples 320000000 \
    --logger wandb \
    --out_dir /kaggle/working/output/smp_m3_spinkick \
    --devices cuda:0 cuda:1
```

320 M là con số đo được từ run humanoid — xem [SMP_PLAYBOOK.md §6.1](SMP_PLAYBOOK.md).

**Nhưng M3.1 chậm hơn humanoid đáng kể.** Throughput đo thực tế là **9 274 samples/s**, chỉ 71 %
của humanoid (13 058) — 27 dof, 30 body, mesh STL thật thay vì primitive. Nên 320 M ở đây tốn
**9 h 35 m** chứ không phải 6 h 35 m, và cả session (prior + setup + train + video) là **~10 h 30 m**.

Hệ quả: chỉ chạy được ở **batch 12 h**, còn dư khoảng 1.5 h. Interactive 9 h thì cell train bị giết
giữa chừng ở quanh 260–275 M và Cell 9 không bao giờ chạy.

Muốn dư dả hơn thì hạ xuống `--max_samples 260000000` (7 h 47 m). Humanoid ở mốc tương đương đã qua
điểm hết ngã từ lâu, nên đánh đổi này gần như không mất chất lượng.

```python
# Cell 8b — tắt watchdog sau khi train xong
wd.terminate()
```

```python
# Cell 9 — video + upload
!python kaggle/make_videos.py \
    --out_dir      /kaggle/working/output/smp_m3_spinkick \
    --env_config   data/envs/smp_vr_m3_1_spinkick_env.yaml \
    --agent_config data/agents/smp_vr_m3_1_spinkick_agent.yaml \
    --char_file    data/assets/vr_m3_1/vr_m3_1.xml \
    --motion_file  data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick.pkl \
    --wandb_project mimickit-smp \
    --steps 300

!python kaggle/wandb_upload.py --project mimickit-smp \
    --run_name smp_m3_spinkick_files \
    --files /kaggle/working/output/smp_m3_spinkick/model.pt \
            /kaggle/working/output/smp_m3_spinkick/log.txt
```

`--char_file` **phải** là MJCF của M3.1, mặc định của script là humanoid. Chỉ xem
`policy_final.mp4` và `reference_data.mp4`; `reference_sim_final.mp4` hỏng (đứng yên một tư thế),
lý do ở [SMP_PLAYBOOK.md §7](SMP_PLAYBOOK.md).

---

## Mốc so sánh với humanoid

Chạy trên cùng clip nên đối chiếu trực tiếp được. Từ run humanoid `m6rv7ht3`:

| Samples | `Sds_Loss_Mean` | `Ep_Len_Frac` |
|---|---|---|
| 60 M | 0.93 | 0.14 |
| 93 M | 0.472 | 0.79 ← bước ngoặt |
| 132 M | 0.270 | 0.98 |
| 309 M | 0.188 | 0.99 |

Nếu tới 130 M mà `Ep_Len_Frac` của M3.1 vẫn dưới 0.3 thì nó đang tụt xa so với humanoid — dừng lại
soi config thay vì đốt tiếp GPU. Nghi ngờ theo thứ tự: `init_pose` (chiều cao root, thứ tự dof),
gains PD trong MJCF, rồi mới tới chất lượng retarget của clip.

Ngược lại, `Sds_Loss_Mean` của hai robot **không** so trực tiếp được — chúng chuẩn hoá theo hai
prior khác nhau, trên hai không gian obs khác nhau. Chỉ so hình dạng đường cong, đừng so giá trị.

## Đã kiểm sẵn cho clip này

Forward kinematics qua toàn bộ 78 frame của `vr_m3_1_humanoid_spinkick.pkl`:

- Cổ chân thấp nhất ở **+4.4 cm** (trái) và **+5.1 cm** (phải) — đúng bằng độ dày đế, bàn chân có đạp đất thật.
- Body thấp kế tiếp là đầu gối, **không bao giờ xuống dưới +38.8 cm**.

Nghĩa là `contact_bodies: [right_ankle_roll_link, left_ankle_roll_link]` an toàn với biên rất rộng:
clip tham chiếu không bao giờ tự vi phạm kiểm tra ngã. Không cần chạy
`tools/fix_ground_penetration.py` cho clip này.
