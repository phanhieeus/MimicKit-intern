# Train SMP cho M3.1 — clip dance, curriculum tốc độ 4 chặng

Mọi thứ cần để train đã có trong repo. Tài liệu này là các cell Kaggle, chạy từ trên xuống.

> **Đọc mục [Vì sao lại là curriculum tốc độ](#vì-sao-lại-là-curriculum-tốc-độ) trước khi tốn GPU.**
> Có một cách rẻ hơn nhiều để bác bỏ giả thuyết, và nếu nó bác bỏ được thì cả lộ trình 4 chặng là vô ích.

---

## Bốn chặng

| Chặng | Hệ số | Clip | Khung | Chu kỳ | Tốc độ khớp L2 | Mô-men tay |
|---|---|---|---|---|---|---|
| 1 | 4.0× | `vr_m3_1_dance_what_1cycle_s4.pkl` | 108 | 3.600 s | **4.29** | 0.15× ✓ |
| 2 | 2.0× | `..._s2.pkl` | 54 | 1.800 s | 8.58 | 0.59× ✓ |
| 3 | 1.4× | `..._s1p4.pkl` | 38 | 1.267 s | 11.84 | 1.20× |
| 4 | 1.0× | `..._s1.pkl` | 27 | 0.900 s | 17.15 | 2.36× ✗ |

Đối chiếu: `zombie_walk` — clip **duy nhất** M3.1 học được — có tốc độ khớp **3.29 rad/s**. Chặng 1
đặt ở 4.29, ngay cạnh mốc đã chứng minh được.

Cả bốn clip đều `loop_mode: 1`, root về đúng điểm xuất phát (net 0.0000 m), không có pha bay.

---

## Clip gốc là một vòng lặp 19 lần

`vr_m3_1_dance_what.pkl` dài 17.07 s, nhưng:

```
513 = 27 × 19
max |dof[i] - dof[i+27]| = 0.0000 rad
```

Bit-exact. Nó chứa **0.90 giây** chuyển động, lặp 19 lần. Cắt xuống một chu kỳ không mất gì.

Hệ quả: giả thuyết "clip quá dài nên không học được" **sai** — không có nội dung thừa nào để mà dài.
Xem [SMP_PLAYBOOK.md](SMP_PLAYBOOK.md) để biết vì sao giả thuyết đó từng được đưa ra.

Hệ quả thứ hai, nghiêm trọng hơn: **hai metric trong lần chấm run `261u7tka` là rác**.
`motion_quality.py` lấy `period_frames = clip_seconds × fps = 512` rồi tìm chu kỳ trong dải lag
102–300, trong khi chu kỳ thật là **27** — nằm ngoài dải. Nên `Quality/Periodicity 0.002` và
`Quality/Tempo 0.201` không nói gì về policy. Hai metric còn lại vẫn đúng và vẫn kết luận sập mode:
**Coverage 0.853**, **Speed_Amplitude 0.092** (policy chỉ động đậy bằng 9 % tốc độ clip).

Với clip một chu kỳ thì `period_frames = 27` và metric đo đúng.

---

## Vì sao lại là curriculum tốc độ

Ba run M3.1 xếp đúng theo tốc độ khớp:

| Clip | Tốc độ khớp L2 | Pha bay | Coverage | Kết quả |
|---|---|---|---|---|
| zombie_walk | **3.29** | không | 0.526 | ✅ |
| dance_what | 17.18 | không | 0.853 | ❌ |
| spinkick | 24.35 | 0.47 s | 0.893 | ❌ |

Ba điểm dữ liệu chưa phải quy luật, nhưng tốc độ dự báo tốt hơn hẳn độ dài clip, và nó là biến duy
nhất ta chỉnh được mà không phải retarget lại.

Với dance thì làm chậm **an toàn**, khác hẳn spinkick: `retime_motion.py` báo `implied jump 0 cm`, nên
không có ràng buộc đạn đạo nào bị phá. Với spinkick, chiều cao nhảy đi theo s² nên làm chậm 2× đòi
108 cm — đó là lý do `slow2` tệ hơn bản gốc.

### Cách rẻ để bác bỏ trước

Đừng chạy cả 4 chặng ngay. **Chạy chặng 1 (4.0×) với 60 M sample (~2 giờ)** rồi đọc `Quality/Coverage`:

| Coverage @ 60 M, chặng 4.0× | Kết luận |
|---|---|
| **< 0.55** | tốc độ đúng là nút thắt — chạy tiếp chặng 2 |
| 0.55 – 0.70 | có tác dụng nhưng chưa đủ; thử 6× trước khi đi tiếp |
| **> 0.75** | tốc độ **không** phải nút thắt — dừng lại, cả lộ trình vô ích |

Ở nhánh cuối thì đừng đổ thêm sample. Nghi vấn tiếp theo là `num_disc_obs_steps` (cửa sổ reward chỉ
0.333 s, ngắn hơn chu kỳ 0.9 s) hoặc nhóm vai quá tải mô-men.

---

## Cạm bẫy: checkpoint mang theo prior

Đây là chỗ curriculum dễ hỏng âm thầm nhất.

`SMPAgent.save` ghi `self.state_dict()`, và **100 trong 121 tensor của checkpoint M3.1 là
`_prior_model.*`**. `load` gọi `load_state_dict` mặc định `strict=True`. Nên nếu chặng 2 nạp thẳng
checkpoint chặng 1:

```
run.py --model_file output/smp_m3_dance_s4/model.pt      ← SAI
```

thì prior 4.0× của chặng 1 **đè lên** prior 2.0× vừa train. Env phát clip 2.0× còn `sds_loss` chấm
theo phân phối 4.0×. Không có exception nào, chỉ là reward đo sai suốt cả run.

Dùng `tools/seed_from_stage.py` để ghép: giữ actor/critic/normalizer của chặng trước, thay prior bằng
prior của chặng mới. Script kiểm tra key set và shape trước khi ghi, và **reset `_sds_normalizer`**
(nó đo biên độ SDS loss dưới prior cũ; prior mới ở tốc độ khác cho loss thang khác).

---

## Cell 1 — secrets

```python
import os
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["WANDB_API_KEY"] = sec.get_secret("WANDB_API_KEY")
os.environ["GITHUB_TOKEN"]  = sec.get_secret("GITHUB_TOKEN")
os.environ["WANDB_PROJECT"] = "mimickit-smp"

STAGE = "dance_s4"          # dance_s4 -> dance_s2 -> dance_s1p4 -> dance_s1
PREV  = ""                  # để rỗng ở chặng 1; các chặng sau: tên chặng trước
os.environ["WANDB_NAME"] = f"smp_m3_{STAGE}"

!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Phải ra hai dòng `Tesla T4`. Thấy `P100` thì đổi accelerator rồi restart — sm_60 không chạy được.

Bốn chặng = bốn lần chạy notebook, mỗi lần chỉ sửa hai dòng `STAGE` và `PREV`.

## Cell 2 — clone

```python
import os, shutil
REPO = "/kaggle/working/MimicKit-intern"
token = os.environ["GITHUB_TOKEN"]
os.chdir("/kaggle/working")            # phải thoát khỏi REPO trước khi xoá nó
shutil.rmtree(REPO, ignore_errors=True)
!git clone --depth 1 https://{token}@github.com/phanhieeus/MimicKit-intern.git {REPO}
os.chdir(REPO)
!git log --oneline -1
```

## Cell 3 — dependencies (~4 phút)

```python
!bash kaggle/setup.sh
```

## Cell 4 — data

```python
import os, subprocess, sys
os.makedirs("data/motions/vr_m3_1", exist_ok=True)

packs = []
for dirpath, dirnames, _ in os.walk("/kaggle/input"):
    if {"assets", "motions"} <= set(dirnames):
        packs.append(dirpath)
        dirnames[:] = []
print("found {} pack(s)".format(len(packs)))
for path in packs:
    subprocess.run([sys.executable, "kaggle/prepare_data.py", "--src", path])

import yaml
CLIP = yaml.safe_load(open(f"data/envs/smp_vr_m3_1_{STAGE}_env.yaml"))["motion_file"]
assert os.path.isfile(CLIP), f"thieu clip {CLIP} -- dataset chua co ban moi, chay lai make_m3_dataset.sh"
print("clip ok:", CLIP)
```

Nếu assert này fail thì data pack trên Kaggle chưa có 4 clip mới. Chạy lại
`kaggle/make_m3_dataset.sh` trên máy rồi upload lại dataset — danh sách `CLIPS` đã được cập nhật.

## Cell 5 — prior (lần đầu ~35 phút, sau đó ~10 giây nhờ cache)

```python
!python kaggle/prior_cache.py \
    --cfg_path tools/diffusion_model/config/tinymdm_vr_m3_1_{STAGE}.yaml \
    --out_dir /kaggle/working/output/smp_prior_vr_m3_1_{STAGE} \
    --project mimickit-smp --device cuda:0

!ls -lh /kaggle/working/output/smp_prior_vr_m3_1_{STAGE}/
```

**Mỗi chặng cần prior riêng** — prior học trên clip, clip đổi tốc độ thì phân phối đổi. Không dùng
lại được. `prior_cache.py` băm vân tay từ nội dung config + bytes của clip nên không nhầm giữa các chặng.

Xem `samples/anim_*.gif` trong artifact: nếu prior sinh ra chuyển động lảo đảo thì PPO không cứu được.

> **Chặng 4 (1.0×) có một rủi ro riêng.** Cửa sổ của prior là 32 khung = 1.067 s, dài hơn chu kỳ
> 0.900 s. Bộ lấy mẫu phải quấn vòng qua clip mới đủ dữ liệu. Chưa kiểm chứng. Ba chặng đầu đều có
> chu kỳ dài hơn 1.067 s nên không dính. Nếu Cell 5 ở chặng 4 báo lỗi độ dài, cách xử lý là nhân đôi
> clip thành 54 khung (hai chu kỳ) — nội dung không đổi vì nó vốn tuần hoàn.

## Cell 6 — trỏ agent config vào prior vừa train

```python
import re, pathlib
PRIOR = f"/kaggle/working/output/smp_prior_vr_m3_1_{STAGE}/model.pt"
assert os.path.isfile(PRIOR), "prior chua duoc tao, xem lai Cell 5"
p = pathlib.Path(f"data/agents/smp_vr_m3_1_{STAGE}_agent.yaml")
p.write_text(re.sub(r'^smp_prior_model:.*$', f'smp_prior_model: "{PRIOR}"',
                    p.read_text(), flags=re.M))
!grep smp_prior data/agents/smp_vr_m3_1_{STAGE}_agent.yaml
```

## Cell 6b — hạt giống từ chặng trước (bỏ qua ở chặng 1)

```python
SEED = ""
if PREV:
    !python kaggle/wandb_upload.py --project mimickit-smp \
        --download smp_m3_{PREV}_ckpt_model:latest --dest /kaggle/working/prev

    SEED = f"/kaggle/working/seed_{STAGE}.pt"
    !python tools/seed_from_stage.py \
        --prev_ckpt /kaggle/working/prev/model.pt \
        --prior {PRIOR} \
        --out {SEED}
    assert os.path.isfile(SEED)
print("seed:", SEED or "(none, cold start)")
```

Phải in `100 prior tensors replaced` và `21 tensors carried over`. Con số khác đi nghĩa là hai chặng
không cùng kiến trúc — dừng lại, đừng train.

## Cell 7 — smoke test 2 phút

```python
!python mimickit/run.py \
    --arg_file args/smp_vr_m3_1_{STAGE}_kaggle_args.txt \
    --mode train --num_envs 256 --max_samples 200000 \
    --logger txt --out_dir /kaggle/working/output/smoke \
    --devices cuda:0
```

Bắt gần hết lỗi cấu hình trong 2 phút thay vì 2 giờ. `Ep_Len_Frac` đúng bằng 0 ở vài iteration đầu
nghĩa là mọi episode chết ngay frame một — sai `init_pose` hoặc sai tên link, không phải policy dở.

## Cell 8a — watchdog

```python
import subprocess, sys, time
wd = subprocess.Popen([
    sys.executable, "kaggle/checkpoint_watchdog.py",
    "--model_file", f"/kaggle/working/output/smp_m3_{STAGE}/model.pt",
    "--project", "mimickit-smp",
    "--run_name", f"smp_m3_{STAGE}_ckpt",
    "--interval", "1200",
])
time.sleep(20)
assert wd.poll() is None, (
    "watchdog died at startup -- rc={}. Fix it before training.".format(wd.returncode))
print("watchdog alive, pid", wd.pid)
```

Bắt buộc. `/kaggle/working` chỉ thành Output khi notebook kết thúc sạch, và checkpoint chặng này là
đầu vào của chặng sau — mất nó là mất cả chuỗi.

## Cell 8 — train, 60 M (~2 giờ)

```python
EXTRA = f"--model_file {SEED}" if SEED else ""
!python mimickit/run.py --arg_file args/smp_vr_m3_1_{STAGE}_kaggle_args.txt \
    --max_samples 60000000 {EXTRA}
```

Giữ cell ngắn. Bản cũ liệt kê lại mọi cờ và một lần sót dấu `\` đã giết cả session batch bằng
`IndentationError` sau 47 phút chờ prior.

Theo dõi `Misc/Sds_Loss_Mean`, **không** phải `Smp_Reward_Mean` — reward mang thang chuẩn hoá riêng
của từng run nên không so được giữa các chặng.

## Cell 8b — tắt watchdog

```python
wd.terminate()
```

## Cell 9 — video + chấm điểm

```python
import yaml
CLIP = yaml.safe_load(open(f"data/envs/smp_vr_m3_1_{STAGE}_env.yaml"))["motion_file"]
!python kaggle/make_videos.py \
    --out_dir      /kaggle/working/output/smp_m3_{STAGE} \
    --env_config   data/envs/smp_vr_m3_1_{STAGE}_env.yaml \
    --agent_config data/agents/smp_vr_m3_1_{STAGE}_agent.yaml \
    --char_file    data/assets/vr_m3_1/vr_m3_1.xml \
    --motion_file  {CLIP} \
    --wandb_project mimickit-smp \
    --steps 300

!python kaggle/wandb_upload.py --project mimickit-smp \
    --run_name smp_m3_{STAGE}_files \
    --files /kaggle/working/output/smp_m3_{STAGE}/model.pt
```

`--char_file` **phải** là MJCF của M3.1; mặc định của script là humanoid.

Cell cuối upload `model.pt` thật sự cuối cùng. Watchdog chỉ đẩy mỗi 20 phút, nên bản của nó luôn cũ
hơn bản cuối — với zombie_walk là **11 phút ≈ 6 M sample**. Chặng sau nên lấy từ artifact
`smp_m3_{STAGE}_files` này, không phải từ `_ckpt`.

---

## Tiêu chí nghiệm thu

Đọc `Quality/Coverage` ở Cell 9. **Không dùng `Ep_Len_Frac`** — run spinkick 320 M kết thúc ở 0.827
trong khi robot đứng một chân và không đá lần nào.

| Metric | Ngưỡng | Ý nghĩa khi trượt |
|---|---|---|
| `Quality/Coverage` | **≤ 0.55** | > 0.55 là sập mode: phần lớn clip không bao giờ được ghé tới |
| `Quality/Speed_Amplitude` | > 0.5 | policy động đậy quá ít so với clip |
| `Quality/Periodicity` | ≥ 0.45 | không lặp lại gì cả |
| `Quality/Tempo` | 1.0 ± 0.3 | nhịp đúng nhưng sai tốc độ |

Với clip một chu kỳ thì cả bốn đều đo đúng — khác lần chấm `261u7tka`.

`Tempo` đáng chú ý riêng ở đây: zombie_walk đạt mọi tiêu chí trừ tempo (**0.58**, chậm hơn clip 42 %)
**mà không ai bảo nó chậm lại**. Policy vốn có xu hướng tự đi chậm. Curriculum này huấn luyện chậm
trước một cách có chủ ý, nên rất có thể chặng cuối sẽ trượt tempo. Nếu chặng 4 đạt Coverage nhưng
tempo ~0.6 thì đó **không** phải thất bại của lộ trình — đó là chuyển động đúng chạy chậm, và cách
xử lý là train thêm ở 1.0× chứ không phải quay lại chặng chậm.

---

## Chi phí

| Hạng mục | Mỗi chặng | 4 chặng |
|---|---|---|
| Prior | ~35 phút | ~2 h 20 |
| PPO 60 M | ~2 h | ~8 h |
| Video + chấm | ~10 phút | ~40 phút |
| **Tổng** | **~2 h 45** | **~11 h** |

Ở 8 200 samples/s. Quá 12 h của một session Kaggle, nên **mỗi chặng một session**, nối bằng artifact
trên WandB.

Chạy chặng 1 trước và đọc Coverage. Nếu > 0.75 thì dừng — tiết kiệm 8 giờ.

---

## Phụ lục — tái tạo 4 clip

`data/motions/` nằm trong `.gitignore` (clip đi theo Kaggle dataset, không theo git), nên trên một
bản clone mới phải sinh lại. Chạy từ repo root, cần `vr_m3_1_dance_what.pkl` trong
`data/motions/vr_m3_1/`:

```python
import sys, pickle, numpy as np
sys.path.insert(0, "tools")
from retime_motion import retime

d = pickle.load(open("data/motions/vr_m3_1/vr_m3_1_dance_what.pkl", "rb"))
fr, fps = np.asarray(d["frames"]), float(d["fps"])

# The clip is 27 frames repeated 19 times, bit-exact. One cycle is all of it.
P = 27
assert np.abs(fr[P] - fr[0]).max() == 0.0, "cycle is not exact"

# retime() interpolates 0..n-1 and never wraps, so a loop loses the seam frame
# 26 -> 0. Append the wrap frame, retime across it, then discard it again.
closed = np.concatenate([fr[:P], fr[:1]], axis=0)
for factor, suf in [(4.0, "s4"), (2.0, "s2"), (1.4, "s1p4"), (1.0, "s1")]:
    out = retime(closed, factor)[:-1]
    path = f"data/motions/vr_m3_1/vr_m3_1_dance_what_1cycle_{suf}.pkl"
    pickle.dump({"loop_mode": 1, "fps": fps, "frames": out.tolist()}, open(path, "wb"))
    print(f"{factor:4.1f}x  {len(out):4d} frames = {len(out)/fps:5.3f} s  ->  {path}")
```

Phải in đúng 108 / 54 / 38 / 27 khung. Rồi `bash kaggle/make_m3_dataset.sh` và upload lại dataset —
danh sách `CLIPS` đã có sẵn bốn clip này.
