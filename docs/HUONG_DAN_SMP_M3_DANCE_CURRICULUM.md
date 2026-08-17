# Train SMP cho M3.1 — clip dance, curriculum tốc độ 4 chặng trong 1 session

Bảy cell, chạy từ trên xuống, tự động qua cả 4 chặng rồi in bảng so sánh để bạn quyết định.

Không phải sửa gì giữa chừng. Cell 7 in đủ số liệu 4 chặng.

---

## Bốn chặng

| Chặng | Hệ số | Clip | Khung | Chu kỳ | Tốc độ khớp L2 | Mô-men tay | Ngân sách |
|---|---|---|---|---|---|---|---|
| 1 | 4.0× | `..._1cycle_s4.pkl` | 108 | 3.600 s | **4.29** | 0.15× ✓ | 60 M |
| 2 | 2.0× | `..._1cycle_s2.pkl` | 54 | 1.800 s | 8.58 | 0.59× ✓ | 40 M |
| 3 | 1.4× | `..._1cycle_s1p4.pkl` | 38 | 1.267 s | 11.84 | 1.20× | 40 M |
| 4 | 1.0× | `..._1cycle_s1.pkl` | 27 | 0.900 s | 17.15 | 2.36× ✗ | 40 M |

Đối chiếu: `zombie_walk` — clip **duy nhất** M3.1 học được — có tốc độ khớp **3.29 rad/s**. Chặng 1
đặt ở 4.29, ngay cạnh mốc đã chứng minh được.

Cả bốn clip đều `loop_mode: 1`, root về đúng điểm xuất phát (net 0.0000 m), không có pha bay.

### Vì sao chặng 1 được 60 M còn ba chặng sau chỉ 40 M

Chặng 1 khởi động nguội nên cần đủ ngân sách để qua điểm chuyển pha — `zombie_walk` chuyển pha đâu
đó giữa 26 M và 60 M. Ba chặng sau warm-start từ chặng trước nên không phải học lại từ đầu.

Tổng: 180 M PPO ≈ **6 h 05** ở 8 200 samples/s, cộng 4 prior ≈ 2 h 20, cộng video/chấm ≈ 40 phút →
**≈ 9 h 05**. Trần một session Kaggle là 12 h, còn dư gần 3 giờ. Cell 6 có chốt chặn thời gian, quá
`BUDGET_HOURS` thì bỏ các chặng còn lại để notebook kết thúc sạch — hết giờ giữa chừng thì version bị
đánh dấu failed và `/kaggle/working` không thành Output.

---

## Trước khi mở notebook

1. **Dataset phải có 4 clip mới.** Trên máy: `bash kaggle/make_m3_dataset.sh` rồi
   `kaggle datasets version -p kaggle_dataset_vr_m3_1 --dir-mode zip -m "dance curriculum stages"`.
   Dùng `version`, **không** dùng `create` — `create` tạo dataset thứ hai và notebook vẫn attach bản cũ.
2. **Secrets**: `WANDB_API_KEY`, `GITHUB_TOKEN`.
3. **Accelerator `GPU T4 x2`**, Internet On, attach cả hai dataset (pack gốc + `mimickit-vr-m3-1`).
4. Chạy bằng **Save & Run All (Commit)**, không phải interactive — interactive hết hạn sớm hơn.

---

## Cell 1 — secrets và kế hoạch

```python
import os
from kaggle_secrets import UserSecretsClient

sec = UserSecretsClient()
os.environ["WANDB_API_KEY"] = sec.get_secret("WANDB_API_KEY")
os.environ["GITHUB_TOKEN"]  = sec.get_secret("GITHUB_TOKEN")
os.environ["WANDB_PROJECT"] = "mimickit-smp"

# (tên chặng, ngân sách sample). Thứ tự là thứ tự chạy, mỗi chặng warm-start từ chặng trước.
STAGES = [
    ("dance_s4",   60_000_000),
    ("dance_s2",   40_000_000),
    ("dance_s1p4", 40_000_000),
    ("dance_s1",   40_000_000),
]
BUDGET_HOURS = 10.5      # bỏ các chặng còn lại nếu đã quá mốc này
RESULTS_DIR  = "/kaggle/working/quality"

!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Phải ra **hai** dòng `Tesla T4`. Thấy `P100` thì đổi accelerator rồi restart — sm_60 không chạy được.

## Cell 2 — clone

```python
import os, shutil
REPO = "/kaggle/working/MimicKit-intern"
token = os.environ["GITHUB_TOKEN"]

# Phải thoát khỏi REPO trước khi xoá. Lần chạy lại, cwd của kernel vẫn nằm trong đó
# và rmtree rút mất nền dưới chân tiến trình: getcwd() bắt đầu lỗi, git clone chết
# với "Unable to read current working directory", rồi os.chdir bên dưới ném
# FileNotFoundError -- ba lỗi, một nguyên nhân.
os.chdir("/kaggle/working")
shutil.rmtree(REPO, ignore_errors=True)
!git clone --depth 1 https://{token}@github.com/phanhieeus/MimicKit-intern.git {REPO}
os.chdir(REPO)
!git log --oneline -1
```

## Cell 3 — dependencies (~4 phút)

```python
!bash kaggle/setup.sh
```

## Cell 4 — data, và kiểm tra đủ 4 clip

```python
import os, subprocess, sys, yaml

os.makedirs("data/motions/vr_m3_1", exist_ok=True)

# Tìm data pack ở bất kỳ độ sâu nào Kaggle mount. Bố cục không ổn định: khi thì
# /kaggle/input/<slug>/, khi thì /kaggle/input/datasets/<owner>/<slug>/. Duyệt một
# tầng từng trả về đúng một entry tên "datasets" nên chỉ một pack được link còn cái
# kia biến mất. Pack là thư mục chứa cả assets/ lẫn motions/ -- tìm theo hình dạng đó.
packs = []
for dirpath, dirnames, _ in os.walk("/kaggle/input"):
    if {"assets", "motions"} <= set(dirnames):
        packs.append(dirpath)
        dirnames[:] = []
print("found {} pack(s)".format(len(packs)))
for path in packs:
    subprocess.run([sys.executable, "kaggle/prepare_data.py", "--src", path])

missing = []
for stage, _ in STAGES:
    clip = yaml.safe_load(open(f"data/envs/smp_vr_m3_1_{stage}_env.yaml"))["motion_file"]
    print(f"  {stage:11s} {clip}  {'ok' if os.path.isfile(clip) else 'MISSING'}")
    if not os.path.isfile(clip):
        missing.append(clip)
assert not missing, (
    "dataset chua co cac clip nay: " + ", ".join(missing) +
    " -- chay lai kaggle/make_m3_dataset.sh tren may roi day version moi len Kaggle")
print("\ndu 4 clip")
```

Assert này là lý do bạn không mất 35 phút train prior rồi mới phát hiện thiếu clip.

## Cell 5 — định nghĩa một chặng

```python
import json, pathlib, re, subprocess, sys, time

os.makedirs(RESULTS_DIR, exist_ok=True)

def sh(cmd, env=None):
    """Chạy và ném lỗi nếu thất bại. In lệnh ra để log đọc được."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {' '.join(str(c) for c in cmd)}")

def run_stage(stage, max_samples, prev):
    t0 = time.time()
    out    = f"/kaggle/working/output/smp_m3_{stage}"
    prior  = f"/kaggle/working/output/smp_prior_vr_m3_1_{stage}/model.pt"
    clip   = yaml.safe_load(open(f"data/envs/smp_vr_m3_1_{stage}_env.yaml"))["motion_file"]

    # --- prior: mot prior rieng cho moi chang ---
    # Prior hoc tren clip. Doi toc do clip la doi phan phoi, nen khong dung lai duoc.
    sh([sys.executable, "kaggle/prior_cache.py",
        "--cfg_path", f"tools/diffusion_model/config/tinymdm_vr_m3_1_{stage}.yaml",
        "--out_dir", os.path.dirname(prior),
        "--project", "mimickit-smp", "--device", "cuda:0"])
    assert os.path.isfile(prior), f"prior khong duoc tao: {prior}"

    p = pathlib.Path(f"data/agents/smp_vr_m3_1_{stage}_agent.yaml")
    p.write_text(re.sub(r'^smp_prior_model:.*$', f'smp_prior_model: "{prior}"',
                        p.read_text(), flags=re.M))

    # --- hat giong tu chang truoc ---
    # Khong duoc dua thang checkpoint chang truoc vao --model_file: 100/121 tensor
    # cua no la _prior_model.*, load_state_dict la strict, nen prior chang moi bi de.
    extra = []
    if prev:
        seed = f"/kaggle/working/seed_{stage}.pt"
        sh([sys.executable, "tools/seed_from_stage.py",
            "--prev_ckpt", f"/kaggle/working/output/smp_m3_{prev}/model.pt",
            "--prior", prior, "--out", seed])
        extra = ["--model_file", seed]

    # --- watchdog ---
    wd = subprocess.Popen([sys.executable, "kaggle/checkpoint_watchdog.py",
                           "--model_file", f"{out}/model.pt",
                           "--project", "mimickit-smp",
                           "--run_name", f"smp_m3_{stage}_ckpt",
                           "--interval", "1200"],
                          env={**os.environ, "WANDB_NAME": f"smp_m3_{stage}_ckpt"})
    time.sleep(20)
    assert wd.poll() is None, f"watchdog chet luc khoi dong, rc={wd.returncode}"

    try:
        sh([sys.executable, "mimickit/run.py",
            "--arg_file", f"args/smp_vr_m3_1_{stage}_kaggle_args.txt",
            "--max_samples", max_samples] + extra,
           env={"WANDB_NAME": f"smp_m3_{stage}"})
    finally:
        wd.terminate()

    # --- video + cham diem ---
    sh([sys.executable, "kaggle/make_videos.py",
        "--out_dir", out,
        "--env_config", f"data/envs/smp_vr_m3_1_{stage}_env.yaml",
        "--agent_config", f"data/agents/smp_vr_m3_1_{stage}_agent.yaml",
        "--char_file", "data/assets/vr_m3_1/vr_m3_1.xml",
        "--motion_file", clip,
        "--wandb_project", "mimickit-smp",
        "--steps", 300])

    # Cham lai lan nua chi de lay JSON cho bang tong ket. Chi tinh toan numpy, vai giay.
    subprocess.run([sys.executable, "tools/motion_quality.py",
                    "--policy", f"{out}/playback_final/policy.pkl",
                    "--reference", clip, "--quiet",
                    "--json_out", f"{RESULTS_DIR}/{stage}.json"])

    sh([sys.executable, "kaggle/wandb_upload.py", "--project", "mimickit-smp",
        "--run_name", f"smp_m3_{stage}_files",
        "--files", f"{out}/model.pt"])

    print(f"\n=== {stage} xong sau {(time.time()-t0)/3600:.2f} h ===", flush=True)
```

Vì sao upload `model.pt` ở cuối: watchdog chỉ đẩy mỗi 20 phút nên bản của nó luôn cũ hơn bản cuối —
với `zombie_walk` là **11 phút ≈ 6 M sample**. Chặng sau trong cùng session lấy checkpoint từ đĩa
nên không ảnh hưởng, nhưng nếu phải chạy lại chặng nào ở session khác thì lấy từ artifact
`smp_m3_<stage>_files`, không phải `_ckpt`.

## Cell 6 — chạy cả 4 chặng

```python
import time, traceback

t_start = time.time()
prev, done, failed = None, [], []

for stage, budget in STAGES:
    elapsed = (time.time() - t_start) / 3600
    if elapsed > BUDGET_HOURS:
        print(f"\n### BO QUA {stage}: da chay {elapsed:.2f} h > BUDGET_HOURS {BUDGET_HOURS}")
        failed.append((stage, "skipped: out of time"))
        continue

    print(f"\n{'='*70}\n### CHANG {stage} — {budget/1e6:.0f} M — da chay {elapsed:.2f} h\n{'='*70}",
          flush=True)
    try:
        run_stage(stage, budget, prev)
        done.append(stage)
        prev = stage          # chi noi tiep tu mot chang da chay xong
    except Exception as e:
        print(f"\n### CHANG {stage} THAT BAI: {e}")
        traceback.print_exc()
        failed.append((stage, str(e)))
        # khong doi prev: chang sau van warm-start tu chang thanh cong gan nhat

print(f"\ntong {(time.time()-t_start)/3600:.2f} h | xong: {done} | hong: {failed}")
```

Một chặng hỏng không giết cả session. `prev` chỉ tiến khi chặng đó thành công, nên chặng kế tiếp
warm-start từ chặng thành công gần nhất chứ không nối vào một checkpoint không tồn tại.

## Cell 7 — bảng so sánh 4 chặng

```python
import json, glob, os

rows = []
for stage, _ in STAGES:
    path = f"{RESULTS_DIR}/{stage}.json"
    if not os.path.isfile(path):
        rows.append((stage, None)); continue
    rows.append((stage, json.load(open(path))))

hdr = f"{'chang':12s} {'Coverage':>9s} {'Fidelity':>9s} {'SpdAmp':>7s} {'Period':>7s} {'Tempo':>7s} {'Pass':>5s}"
print(hdr); print("-" * len(hdr))
for stage, d in rows:
    if d is None:
        print(f"{stage:12s} {'(khong co ket qua)':>40s}"); continue
    print(f"{stage:12s} {d['Quality/Coverage']:9.3f} {d['Quality/Fidelity']:9.3f} "
          f"{d['Quality/Speed_Amplitude']:7.3f} {d['Quality/Periodicity']:7.3f} "
          f"{d['Quality/Tempo']:7.3f} {int(d['Quality/Passed']):5d}")

print("\nnguong: Coverage <= 0.55 | SpdAmp > 0.5 | Period >= 0.45 | Tempo 1.0 +- 0.3\n")
for stage, d in rows:
    if d and d["problems"]:
        print(f"--- {stage} ---")
        for p in d["problems"]:
            print("   " + p)
```

---

## Đọc kết quả

`Quality/Coverage` là chỉ số quyết định. **Không dùng `Ep_Len_Frac`** — run spinkick 320 M kết thúc
ở 0.827 trong khi robot đứng một chân và không đá lần nào.

| Metric | Ngưỡng | Ý nghĩa khi trượt |
|---|---|---|
| `Coverage` | **≤ 0.55** | > 0.55 là sập mode: phần lớn clip không bao giờ được ghé tới |
| `Speed_Amplitude` | > 0.5 | policy động đậy quá ít so với clip |
| `Periodicity` | ≥ 0.45 | không lặp lại gì cả |
| `Tempo` | 1.0 ± 0.3 | nhịp đúng nhưng sai tốc độ |

Đọc theo **hình dạng của cột Coverage qua 4 chặng**, không phải từng chặng riêng lẻ:

| Hình dạng | Kết luận |
|---|---|
| Thấp ở 4.0× rồi **tăng dần** lên 1.0× | tốc độ đúng là nút thắt. Chặng cuối trượt thì train thêm ở 1.0×, hoặc chèn 1.2× |
| **Thấp đều** cả 4 chặng | curriculum thắng hẳn — dùng chặng 1.0× |
| **Cao đều** cả 4 chặng, kể cả 4.0× | tốc độ **không** phải nút thắt. Dừng suy nghĩ theo hướng này |
| Thấp ở 4.0×, **nhảy vọt** ngay ở 2.0× | ngưỡng nằm giữa 4× và 2×; chèn 3× thay vì đi tiếp |

Nhánh "cao đều" là nhánh quan trọng nhất và nó **bác bỏ giả thuyết của tôi**. Khi đó nghi vấn tiếp
theo là `num_disc_obs_steps`: cửa sổ reward chỉ **0.333 s**, ngắn hơn chu kỳ 0.9 s ở mọi chặng, nên
prior không nhìn thấy cấu trúc dài hơn một phần ba chu kỳ. Nâng 10 → 30 là thí nghiệm kế tiếp.

### Riêng về Tempo

`zombie_walk` đạt mọi tiêu chí trừ tempo (**0.58**, chậm hơn clip 42 %) **mà không ai bảo nó chậm
lại**. Policy vốn có xu hướng tự đi chậm. Curriculum này huấn luyện chậm trước một cách có chủ ý, nên
rất có thể chặng cuối trượt tempo. Coverage đạt mà tempo ~0.6 **không** phải thất bại của lộ trình —
đó là chuyển động đúng chạy chậm, và cách xử lý là train thêm ở 1.0×, không phải quay lại chặng chậm.

---

## Vì sao lại là curriculum tốc độ

Ba run M3.1 xếp đúng theo tốc độ khớp:

| Clip | Tốc độ khớp L2 | Pha bay | Coverage | Kết quả |
|---|---|---|---|---|
| zombie_walk | **3.29** | không | 0.526 | ✅ |
| dance_what | 17.18 | không | 0.853 | ❌ |
| spinkick | 24.35 | 0.47 s | 0.893 | ❌ |

**Ba điểm dữ liệu chưa phải quy luật.** Nhưng tốc độ dự báo tốt hơn hẳn độ dài clip, và nó là biến
duy nhất chỉnh được mà không phải retarget lại.

Với dance thì làm chậm **an toàn**, khác hẳn spinkick: `retime_motion.py` báo `implied jump 0 cm` nên
không có ràng buộc đạn đạo nào bị phá. Với spinkick, chiều cao nhảy đi theo s² — làm chậm 2× đòi
108 cm, đó là lý do `slow2` tệ hơn bản gốc.

## Clip gốc là một vòng lặp 19 lần

`vr_m3_1_dance_what.pkl` dài 17.07 s, nhưng:

```
513 = 27 × 19
max |dof[i] - dof[i+27]| = 0.0000 rad
```

Bit-exact. Nó chứa **0.90 giây** chuyển động lặp 19 lần. Cắt xuống một chu kỳ không mất gì, và giả
thuyết cũ "clip quá dài nên không học được" là **sai** — không có nội dung thừa nào để mà dài.

Hệ quả thứ hai: **hai metric trong lần chấm run `261u7tka` là rác**. `motion_quality.py` lấy
`period_frames = clip_seconds × fps = 512` rồi tìm chu kỳ trong dải lag 102–300, trong khi chu kỳ
thật là **27** — nằm ngoài dải. `Quality/Periodicity 0.002` và `Quality/Tempo 0.201` không nói gì về
policy. Hai metric còn lại vẫn đúng và vẫn kết luận sập mode: **Coverage 0.853**,
**Speed_Amplitude 0.092**. Với clip một chu kỳ thì `period_frames = 27` và cả bốn metric đo đúng.

## Cạm bẫy: checkpoint mang theo prior

`SMPAgent.save` ghi `self.state_dict()`, và **100 trong 121 tensor của checkpoint M3.1 là
`_prior_model.*`**. `load` gọi `load_state_dict` mặc định `strict=True`. Nên nạp thẳng checkpoint
chặng trước:

```
run.py --model_file output/smp_m3_dance_s4/model.pt      ← SAI
```

khiến prior 4.0× của chặng 1 **đè lên** prior 2.0× vừa train. Env phát clip 2.0× còn `sds_loss` chấm
theo phân phối 4.0×. Không có exception, chỉ là reward đo sai suốt cả run.

`tools/seed_from_stage.py` giữ actor/critic/normalizer của chặng trước, thay prior bằng prior chặng
mới, kiểm tra key set và shape trước khi ghi, và **reset `_sds_normalizer`** (nó đo biên độ SDS loss
dưới prior cũ; prior mới ở tốc độ khác cho loss thang khác). Cell 5 gọi nó tự động.

## Rủi ro đã biết ở chặng 4

Cửa sổ của prior là 32 khung = **1.067 s**, dài hơn chu kỳ **0.900 s** của chặng 1.0×. Bộ lấy mẫu
phải quấn vòng qua clip mới đủ dữ liệu, và điều này **chưa được kiểm chứng**. Ba chặng đầu có chu kỳ
dài hơn 1.067 s nên không dính.

Nếu Cell 6 báo lỗi độ dài ở chặng `dance_s1`, cách xử lý là nhân đôi clip thành 54 khung (hai chu kỳ)
— nội dung không đổi vì nó vốn tuần hoàn. Ba chặng kia vẫn có kết quả nên bảng ở Cell 7 vẫn đọc được.

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

Phải in đúng 108 / 54 / 38 / 27 khung. Rồi `bash kaggle/make_m3_dataset.sh` và đẩy version mới.
