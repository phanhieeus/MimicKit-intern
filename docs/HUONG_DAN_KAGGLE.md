# Hướng dẫn chạy MimicKit trên Kaggle

Làm theo từng bước bên dưới. Bản tiếng Anh chi tiết hơn về lý do kỹ thuật:
[README_Kaggle.md](README_Kaggle.md).

**Tóm tắt:** trên Kaggle không cài được Isaac Gym (cần tải thủ công sau khi đăng nhập NVIDIA, và chỉ
hỗ trợ Python ≤ 3.8, trong khi Kaggle chạy Python 3.11). Vì vậy ta dùng engine **Newton**, cài
được qua PyPI. Hệ quả: **mọi lệnh đều phải thêm** `--engine_config data/engines/newton_engine.yaml`,
vì các file trong `args/` mặc định trỏ tới Isaac Gym.

---

## Bước 0 — Chuẩn bị data (chỉ làm 1 lần)

Assets, motions và model pretrained (~534 MB) không nằm trong git, phải upload lên Kaggle thành
Dataset để notebook gắn vào.

1. Tải data pack từ [link OneDrive trong README](../README.md#installation), giải nén ra
   `data/MimicKit_Data/` (bạn đã có sẵn ở máy).
2. Nén thư mục đó thành 1 file zip.
3. Lên Kaggle → **Datasets** → **New Dataset** → upload file zip → đặt tên `mimickit-data` → Create.

Hoặc dùng CLI:

```bash
pip install kaggle          # cần ~/.kaggle/kaggle.json
mkdir -p /tmp/mimickit_data && cp -r data/MimicKit_Data/* /tmp/mimickit_data/
kaggle datasets init -p /tmp/mimickit_data
# sửa /tmp/mimickit_data/dataset-metadata.json: điền "title" và "id" (vd "phanhieeus/mimickit-data")
kaggle datasets create -p /tmp/mimickit_data --dir-mode zip
```

Upload xong thì lần sau dùng lại mãi, không cần làm lại bước này.

> **Không tải thẳng từ OneDrive trong notebook được.** Link chia sẻ SharePoint trả về
> `401 Access denied` cho `wget`/`curl` (cần cookie từ phiên đăng nhập trình duyệt), kể cả khi thêm
> `&download=1`. Bắt buộc phải qua Kaggle Dataset như trên.

---

## Bước 1 — Tạo notebook và chỉnh settings

Lên Kaggle → **Code** → **New Notebook**, rồi mở panel bên phải:

| Mục | Chọn |
| --- | --- |
| **Accelerator** | `GPU T4 x2` (hoặc `P100`) |
| **Internet** | `On` — bắt buộc, để `git clone` và `pip install` chạy được |
| **Add Input → Datasets** | chọn `mimickit-data` vừa upload ở Bước 0 |

Có thể upload luôn file [`kaggle/mimickit_kaggle.ipynb`](../kaggle/mimickit_kaggle.ipynb) có sẵn
trong repo (**File → Import Notebook**) rồi chạy tuần tự từ trên xuống, khỏi gõ lại các cell.

Lưu ý: chỉ thư mục `/kaggle/working` được giữ lại sau khi session kết thúc. Mọi thứ muốn giữ
(model, log) phải nằm trong đó.

---

## Bước 2 — Clone repo

```python
import os

REPO_URL = "https://github.com/phanhieeus/MimicKit-intern.git"
REPO_DIR = "/kaggle/working/MimicKit-intern"

if not os.path.isdir(REPO_DIR):
    !git clone --depth 1 $REPO_URL $REPO_DIR

os.chdir(REPO_DIR)     # bắt buộc: mọi đường dẫn trong config đều tính từ thư mục gốc repo
print(os.getcwd())
```

Nếu repo để **private**: vào **Add-ons → Secrets**, tạo secret tên `GITHUB_TOKEN` chứa GitHub
personal access token, rồi clone bằng:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO_URL = f"https://{token}@github.com/phanhieeus/MimicKit-intern.git"
```

---

## Bước 3 — Cài đặt (~3–4 phút)

```python
!bash kaggle/setup.sh
```

Script cài `xvfb` + thư viện GL, rồi cài `newton==1.0.0` cùng `mujoco` / `mujoco-warp`.

> **Không chạy `pip install -r requirements.txt` trên Kaggle.** File đó có `torch>=1.9.1`, pip có
> thể thay bản torch CUDA dựng sẵn của Kaggle bằng wheel CPU, và mất GPU. `setup.sh` cố tình bỏ qua
> torch vì lý do này.

Cuối cell sẽ in ra version của torch / warp / newton và `cuda: True`. Nếu thấy `cuda: False` thì
dừng lại, restart session và chạy lại — chạy tiếp cũng vô nghĩa vì sẽ rơi về CPU.

---

## Bước 4 — Nối data vào `data/`

```python
!ls /kaggle/input
!python kaggle/prepare_data.py
```

Script tự tìm data pack trong `/kaggle/input` (dò sâu tối đa 2 cấp để tìm thư mục có chứa `assets/`
và `motions/`) rồi tạo symlink vào `data/`. File nào đã có sẵn trong git (như
`data/assets/humanoid/humanoid.xml`) thì giữ nguyên, không đè.

Cuối cell phải thấy 2 dòng `[ok]`:

```
  [ok] data/assets/humanoid/humanoid.xml
  [ok] data/motions/humanoid/humanoid_spinkick.pkl
```

Nếu là `[MISSING]`: chưa gắn dataset vào notebook, hoặc cấu trúc thư mục khác — chỉ đường dẫn thủ
công bằng `!python kaggle/prepare_data.py --src /kaggle/input/mimickit-data`.

---

## Bước 5 — Chạy thử (smoke test)

Cách rẻ nhất để xác nhận Newton + MuJoCo-Warp + data đều hoạt động. Chỉ phát lại một motion clip:

```bash
!python mimickit/run.py \
    --arg_file args/view_motion_humanoid_args.txt \
    --engine_config data/engines/newton_engine.yaml \
    --num_envs 4 --mode test --test_episodes 1 \
    --visualize false --devices cuda:0
```

Lần chạy đầu tiên Warp phải biên dịch JIT các kernel, mất vài phút và có vẻ như bị treo — cứ đợi,
những lần sau đã được cache nên nhanh hơn nhiều.

**Vì sao cờ trên dòng lệnh ghi đè được `--arg_file`?** Parser của MimicKit giữ giá trị **đầu tiên**
nó gặp cho mỗi key (`mimickit/util/arg_parser.py:26`), và `run.py` đọc dòng lệnh trước rồi mới đọc
arg file. Nên `--engine_config` ở trên thay thế Isaac Gym trong file args.

---

## Bước 6 — Train

```bash
!python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --engine_config data/engines/newton_engine.yaml \
    --mode train \
    --num_envs 1024 \
    --max_samples 200000000 \
    --visualize false --video false --logger tb \
    --out_dir /kaggle/working/output/deepmimic_humanoid \
    --devices cuda:0
```

Giải thích các cờ quan trọng:

- `--visualize false` — **bắt buộc**. Mặc định là `true`, sẽ cố mở cửa sổ GL và lỗi trên Kaggle.
- `--num_envs 1024` — file args mặc định 4096, không vừa 16 GB của T4. Nếu dùng P100/A100 hoặc thấy
  `nvidia-smi` còn dư VRAM thì tăng lên.
- `--max_samples` — giới hạn để job kết thúc trong giới hạn session (9 giờ interactive / 12 giờ
  batch). Bỏ cờ này thì job chạy vô hạn và bị kill giữa chừng, mất tiến độ.
- `--out_dir` — phải nằm trong `/kaggle/working` mới giữ được model sau khi session kết thúc.
- `--devices cuda:0 cuda:1` — dùng cả 2 GPU nếu chọn máy T4 x2.

Xem log trong lúc/sau khi train:

```python
!tail -30 /kaggle/working/output/deepmimic_humanoid/log.txt
```

Với `--logger tb`, file TensorBoard events nằm cùng thư mục, tải về máy rồi mở bằng
`tensorboard --logdir=...`.

---

## Bước 7 — Test model đã train (tuỳ chọn)

```bash
!python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --engine_config data/engines/newton_engine.yaml \
    --num_envs 16 --mode test --test_episodes 8 \
    --visualize false \
    --model_file /kaggle/working/output/deepmimic_humanoid/model.pt \
    --devices cuda:0
```

Đổi `--model_file` thành `data/models/deepmimic_humanoid_spinkick_model.pt` để chạy model
pretrained. Lưu ý các model pretrained được train bằng Isaac Gym, nên chạy trên Newton kết quả chỉ
mang tính tham khảo, không phải điểm số thật của chúng.

---

## Về hiển thị và quay video

- `--visualize true` **không bao giờ chạy được** trên Kaggle — không có màn hình để hiển thị.
- `--video true` quay video headless qua Xvfb (`setup.sh` đã cài, `util/display.py` tự bật virtual
  display). Container GPU của Kaggle không phải lúc nào cũng cấp được context GL/EGL dùng được, nên
  đây là tính năng "được thì tốt". Nếu lỗi thì cứ train với `--video false`, không ảnh hưởng gì.

---

## Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
| --- | --- |
| `ModuleNotFoundError: isaacgym` | Quên `--engine_config data/engines/newton_engine.yaml` |
| `FileNotFoundError: data/motions/...pkl` | Chưa gắn dataset hoặc chưa chạy `prepare_data.py` |
| `torch.cuda.is_available() == False` | pip đã thay torch bằng bản CPU. Restart session, đừng chạy `pip install -r requirements.txt` |
| CUDA out of memory | Giảm `--num_envs` (thử 512, rồi 256) |
| Lỗi import `mujoco_warp` | `pip install mujoco==3.5.0 mujoco-warp==3.5.0.2` |
| Chạy mãi không thấy output | Warp đang biên dịch JIT lần đầu, đợi vài phút |
| `WarpCodegenKeyError: Referencing undefined symbol: J_kj` | warp-lang bị lên 1.16.0. Phải dùng đúng 1.15.0: `pip install "warp-lang==1.15.0"` rồi **Restart session** |
| Mất hết model sau khi đóng notebook | `--out_dir` không nằm trong `/kaggle/working` |
| Job bị kill giữa chừng | Vượt giới hạn thời gian session — đặt `--max_samples` nhỏ hơn |
