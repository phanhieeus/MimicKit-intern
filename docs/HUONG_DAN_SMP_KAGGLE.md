# Train SMP (mimic 1 clip) trên Kaggle — humanoid native của MimicKit

Hướng dẫn từng cell, từ `git clone` đến mp4 so sánh policy với motion gốc, mọi thứ đẩy lên WandB.

> Công thức này **đã chạy thành công và kiểm chứng bằng video**. Muốn áp dụng cho motion khác hoặc
> robot khác (M3.1) thì đọc [SMP_PLAYBOOK.md](SMP_PLAYBOOK.md) — ở đó có số liệu thực đo, ngân sách
> sample, danh sách những gì phải sửa khi đổi robot, và cách đọc metric.

- Nhân vật: **humanoid native** của MimicKit (`data/assets/humanoid/humanoid.xml`), **không phải** robot
  `vr_m3_1` / g1 / go2.
- Motion: `data/motions/humanoid/humanoid_spinkick.pkl` (spinkick — cú đá xoay).
- Phương pháp: **SMP** (Score-Matching Motion Priors) — single clip. Xem
  [README_SMP.md](README_SMP.md) cho phần lý thuyết.
- Engine: **Newton** (Isaac Gym/Isaac Lab không cài được trên Kaggle — xem
  [README_Kaggle.md](README_Kaggle.md)). Nên **mọi lệnh đều phải có**
  `--engine_config data/engines/newton_engine.yaml`.

SMP chạy 2 giai đoạn:

| Giai đoạn | Làm gì | Bắt buộc? |
| --- | --- | --- |
| 1. Train **prior** (TinyMDM, diffusion) | học phân phối chuyển động từ clip spinkick | **Optional** — data pack đã có sẵn `data/models/smp_priors/smp_prior_spinkick.pt` |
| 2. Train **policy** (PPO + SDS reward từ prior) | học điều khiển vật lý để bám prior | Bắt buộc |

Giai đoạn 2 dùng PPO: reward = `smp_reward_weight * exp(-SDS_loss * sds_loss_scale)`, tức là
"giống clip tới đâu" chấm bằng diffusion prior chứ không bằng tracking từng frame như DeepMimic.

---

## Bước 0 — Chuẩn bị trước khi mở notebook

1. **Data pack** đã upload thành Kaggle Dataset (làm 1 lần) — xem
   [HUONG_DAN_KAGGLE.md § Bước 0](HUONG_DAN_KAGGLE.md). Cần có `models/smp_priors/` trong pack.
2. **Secrets** (Add-ons → Secrets), bạn đã lưu rồi:
   - `WANDB_API_KEY`
   - `GITHUB_TOKEN` (PAT — chỉ cần nếu repo private)
3. **Notebook settings**: Accelerator `GPU T4 x2` hoặc `P100`, Internet `On`,
   Add Input → Datasets → `mimickit-data`.

---

## Cell 1 — Nạp secrets vào biến môi trường

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["GITHUB_TOKEN"]  = secrets.get_secret("GITHUB_TOKEN")   # bỏ nếu repo public

# Tên project + tên run trên WandB. wandb_logger đọc 2 biến này.
os.environ["WANDB_PROJECT"] = "mimickit-smp"
os.environ["WANDB_NAME"]    = "smp_spinkick_humanoid"

print("secrets ok")
```

> Các cell `!lệnh` bên dưới kế thừa biến môi trường của kernel, nên không cần export lại.

---

## Cell 2 — Clone repo

```python
import os

REPO = "/kaggle/working/MimicKit-intern"
token = os.environ.get("GITHUB_TOKEN", "")
url = f"https://{token}@github.com/phanhieeus/MimicKit-intern.git" if token \
      else "https://github.com/phanhieeus/MimicKit-intern.git"

if not os.path.exists(REPO):
    !git clone --depth 1 {url} {REPO}
os.chdir(REPO)          # mọi path trong config đều tính từ repo root
print(os.getcwd())
```

---

## Cell 3 — Cài dependencies (~3–4 phút)

```python
!bash kaggle/setup.sh
```

Cài `newton==1.0.0`, `mujoco`, `mujoco-warp`, `warp-lang==1.15.0` (pin có lý do — xem comment trong
`setup.sh`), `diffusers` (cần cho TinyMDM), `wandb`, `moviepy`, `ffmpeg`, `xvfb`.
**Không** cài lại torch — bản CUDA của Kaggle phải giữ nguyên.

---

## Cell 4 — Link data pack vào `data/`

```python
!python kaggle/prepare_data.py
!ls -l data/models/smp_priors/ data/motions/humanoid/humanoid_spinkick.pkl
```

Phải thấy `smp_prior_spinkick.pt`. Nếu thiếu → data pack upload chưa đủ, hoặc train prior ở Cell 5.

---

## Cell 5 — (Optional) Train prior TinyMDM từ clip spinkick

Bỏ qua cell này nếu dùng prior pretrained. Prior chỉ học từ **motion data**, không cần simulator,
nên chạy khá nhanh (~25–40 phút trên T4 với 50k iters).

```python
!python tools/diffusion_model/train_tinymdm.py \
    --cfg_path tools/diffusion_model/config/tinymdm_single_clip.yaml \
    --out_dir /kaggle/working/output/smp_prior_spinkick
```

Config [`tinymdm_single_clip.yaml`](../tools/diffusion_model/config/tinymdm_single_clip.yaml) đã trỏ
sẵn `motion_file: data/motions/humanoid/humanoid_spinkick.pkl` và
`env_config: data/envs/smp_humanoid_env.yaml`. Muốn nhanh hơn thì giảm `num_iterations` xuống
`20_000` (sửa bằng `sed` ngay trong cell nếu ngại mở file):

```python
!sed -i 's/^num_iterations:.*/num_iterations: 20_000/' tools/diffusion_model/config/tinymdm_single_clip.yaml
```

Output: `model.pt`, `diffusion_config.yaml`, `env_config.yaml`, `log.txt`, `samples/` (motion sinh ra
+ ảnh skeleton để mắt thường kiểm tra prior có học được gì không).

Muốn policy dùng prior mới này, sửa agent config:

```python
!sed -i \
  -e 's#^smp_prior_cfg:.*#smp_prior_cfg: "/kaggle/working/output/smp_prior_spinkick/diffusion_config.yaml"#' \
  -e 's#^smp_prior_model:.*#smp_prior_model: "/kaggle/working/output/smp_prior_spinkick/model.pt"#' \
  data/agents/smp_humanoid_agent.yaml
!tail -5 data/agents/smp_humanoid_agent.yaml
```

> `smp_agent` kiểm tra prior và env phải khớp nhau (`global_obs`, `num_disc_obs_steps`,
> `key_bodies`, `control_freq`…). Nếu bạn sửa `data/envs/smp_humanoid_env.yaml` thì **phải train lại
> prior**, nếu không sẽ assert `SMP prior env mismatch ...` ngay khi khởi động.

---

## Cell 6 — Smoke test (2 phút, tránh train 6 tiếng rồi mới lỗi)

```python
!python mimickit/run.py \
    --arg_file args/smp_humanoid_kaggle_args.txt \
    --mode test --num_envs 4 --test_episodes 1 \
    --logger txt --visualize false --devices cuda:0 \
    --model_file data/models/smp_humanoid_spinkick_model.pt \
    --out_dir /kaggle/working/output/smoke
```

Chạy được nghĩa là env + prior + Newton + warp đều ổn. Lần chạy đầu warp JIT-compile kernel mất vài
phút, không có output — cứ đợi.

> File [`args/smp_humanoid_kaggle_args.txt`](../args/smp_humanoid_kaggle_args.txt) là bản Kaggle của
> `smp_humanoid_args.txt`: engine Newton, `num_envs 1024`, `visualize false`, `logger wandb`,
> `out_dir` trong `/kaggle/working`. Arg parser giữ **giá trị đầu tiên** nó thấy và command line được
> đọc trước arg file, nên flag gõ trên dòng lệnh luôn thắng flag trong arg file.

---

## Cell 7 — Train policy (PPO + SMP reward)

```python
!python mimickit/run.py \
    --arg_file args/smp_humanoid_kaggle_args.txt \
    --mode train \
    --num_envs 1024 \
    --max_samples 60000000 \
    --logger wandb \
    --out_dir /kaggle/working/output/smp_spinkick \
    --devices cuda:0 cuda:1
```

Ý nghĩa các flag:

| Flag | Vì sao |
| --- | --- |
| `--devices cuda:0 cuda:1` | dùng cả 2 GPU của instance `T4 x2`. `run.py` spawn 1 process/GPU, gradient đồng bộ qua `torch.distributed`; root process (cuda:0) là process duy nhất ghi log/WandB/checkpoint. |
| `--num_envs 1024` | **tính cho mỗi GPU**, nên 2 GPU = 2048 env song song. Arg gốc để 4096/GPU, không vừa 16 GB của T4. |
| `--max_samples 60000000` | chặn để run kết thúc trong giới hạn session (9h interactive / 12h batch). Không có nó thì run vô hạn và bị kill giữa chừng. `Samples` đếm gộp cả 2 GPU. |
| `--logger wandb` | metrics (`Test_Return`, `Train_Return`, SDS loss…) stream thẳng lên WandB project `mimickit-smp`. |
| `--visualize false` | mặc định là `true`, sẽ cố mở cửa sổ GL và crash. |
| `--save_int_models true` (trong arg file) | lưu checkpoint theo iter vào `int_models/`, để session sau train tiếp. |

> Chỉ chọn được P100 (1 GPU) thì thêm `--devices cuda:0` để ghi đè arg file, và cân nhắc
> `--num_envs 2048` cho bằng lượng env cũ. Các cell test/playback bên dưới luôn chạy 1 GPU —
> đó là chủ ý, chúng chỉ roll out 1 env.

Kết quả nằm ở `/kaggle/working/output/smp_spinkick/`: `model.pt`, `log.txt`, `int_models/`,
`agent_config.yaml` / `env_config.yaml` / `engine_config.yaml` (bản copy của config đã dùng).

**Train tiếp ở session sau:** thêm `--model_file /kaggle/input/<dataset-checkpoint>/model.pt` (hoặc
path trong `/kaggle/working` nếu vẫn cùng session). 60M samples trên T4 thường chưa hội tụ hẳn —
spinkick là clip khó; cứ nối 2–3 session.

### (Optional) quay video ngay trong lúc train

```python
!python mimickit/run.py \
    --arg_file args/smp_humanoid_kaggle_args.txt \
    --mode train --num_envs 1024 --max_samples 60000000 \
    --video true --logger wandb \
    --out_dir /kaggle/working/output/smp_spinkick --devices cuda:0 cuda:1
```

`--video true` cho Newton dựng thêm 1 viewer GL headless qua Xvfb, quay lại các đợt test định kỳ
(mỗi `iters_per_output` = 100 iter) và `wandb_logger` upload thẳng mp4 (`Sim_Recording`) lên run —
tức là bạn xem được tiến triển **trong lúc train** chứ không phải đợi tới cuối.

Đánh đổi: container GPU của Kaggle không phải lúc nào cũng có GL context dùng được, và nó làm chậm
train. Nếu fail thì quay lại `--video false` rồi dùng `kaggle/make_videos.py` ở Cell 8 — đường đó
dựng bằng MuJoCo/EGL, ổn định hơn hẳn.

---

## Cell 8 — Sinh mp4: policy vs. motion gốc

Ba video để trả lời câu hỏi "nó có mimic được không":

1. `reference_data.mp4` — clip spinkick **gốc** trong data (ground truth).
2. `policy.mp4` — humanoid do policy điều khiển trong physics sim.
3. `reference_sim.mp4` — nhân vật tham chiếu mà env dựng cùng lúc với policy (cùng thời điểm, cùng khung hình).

```python
OUT = "/kaggle/working/output/smp_spinkick"
RUN_ID = "mbi5axg4"      # id của run training, lấy từ URL WandB. Bỏ đi thì tạo run mới.

!python kaggle/make_videos.py \
    --out_dir      {OUT} \
    --env_config   data/envs/smp_humanoid_env.yaml \
    --agent_config data/agents/smp_humanoid_agent.yaml \
    --char_file    data/assets/humanoid/humanoid.xml \
    --motion_file  data/motions/humanoid/humanoid_spinkick.pkl \
    --wandb_project mimickit-smp \
    --wandb_run_id {RUN_ID} \
    --steps 300
```

Script làm tuần tự: render clip gốc → roll out policy ra `.pkl` → render policy + reference → upload
lên WandB (tab **Media** của run, key `video/...`). **Mỗi bước đều được kiểm tra, sai là dừng ngay
kèm lỗi** — khác với chuỗi 3 lệnh rời trước đây, nơi rollout chết thì render không có input và bước
upload chỉ in một dòng `[WARN] not a file` rồi vẫn báo thành công (đúng cái đã xảy ra ở lần chạy
trước: run `smp_spinkick_media` lên WandB mà không có mp4 nào).

Vài flag đáng nhớ:

- `--wandb_run_id <id>`: video nằm **chung run với training**, cạnh các đường cong. Không có flag này
  thì tạo run riêng tên `<out_dir>_videos`.
- `--no_upload`: chỉ tạo file, không cần `WANDB_API_KEY`.
- `--int_models`: render **mọi checkpoint** trong `int_models/` → xem được cả quá trình tiến hoá,
  checkpoint sớm ngã sấp mặt, checkpoint muộn đứng được. Tốn thời gian (mỗi checkpoint 1 rollout).
- `--gl osmesa`: dùng khi EGL fail.
- `--steps 300` = 10 giây ở 30 Hz = trọn 1 episode (`episode_length: 10.0`). Rollout in ra
  `episodes terminated during the recording` — khác 0 nghĩa là humanoid đã ngã giữa chừng và clip có
  chứa cả đoạn reset.

Nếu vẫn muốn chạy tay từng bước để soi lỗi: `tools/play_policy_to_mp4.py` rồi
`tools/render_robot_video.py` — `make_videos.py` chỉ gọi đúng hai script đó.

Xem ngay trong notebook:

```python
import base64
from IPython.display import HTML, display

def show(path, w=480):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f'<p>{path}</p><video width={w} controls src="data:video/mp4;base64,{b64}">'))

for name in ["reference_data.mp4", "policy_final.mp4", "reference_sim_final.mp4"]:
    show(f"{OUT}/{name}")
```

**Đọc kết quả thế nào:** policy mimic được khi `policy.mp4` có cùng *cấu trúc* động tác với
`reference_data.mp4` — xoay người, tung chân, tiếp đất — chứ không cần trùng khít từng frame (SMP
không tracking theo phase, nó chỉ ép phân phối chuyển động giống prior). Các dấu hiệu chưa được:
đứng yên rung rung, ngã ngay giây đầu, hoặc lặp 1 tư thế trung bình. Đối chiếu thêm các metric ở
mục [Đọc metric](#đọc-metric-trainreturn--0-là-bình-thường) bên dưới.

---

## Notebook riêng chỉ để render video (session train đã đóng)

Khi session train kết thúc, `/kaggle/working` mất sạch — checkpoint chỉ còn trong WandB Artifact.
Muốn render video cho một run cũ thì mở **notebook mới, bật GPU** (1 GPU là đủ, rollout không cần 2),
rồi chạy đúng 6 cell dưới. Không có cách nào rút ngắn hơn: `play_policy_to_mp4.py` phải dựng lại
đúng cái env đã train, nên vẫn cần repo + dependencies + data pack. Tổng khoảng 10–15 phút, phần lớn
là `setup.sh`.

```python
# Cell 1 — secrets
import os
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["WANDB_API_KEY"] = sec.get_secret("WANDB_API_KEY")
os.environ["GITHUB_TOKEN"]  = sec.get_secret("GITHUB_TOKEN")
os.environ["WANDB_PROJECT"] = "mimickit-smp"
```

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
# Cell 4 — data pack (assets/, motions/, models/)
!python kaggle/prepare_data.py
!ls data/motions/humanoid/ | head
```

> **Clone lại là phải chạy lại Cell 4.** `prepare_data.py` tạo symlink *bên trong* `data/` của repo,
> nên `shutil.rmtree(REPO)` ở Cell 2 cuốn theo cả chúng. Bỏ qua bước này thì lỗi rơi xuống tận
> `make_videos.py`: `FileNotFoundError: data/motions/humanoid/humanoid_spinkick.pkl`. Cell 3 thì
> ngược lại, pip cài ra ngoài repo nên chạy một lần cho cả session là đủ.

```python
# Cell 5 — kéo checkpoint của run cũ về từ WandB Artifact
OUT = "/kaggle/working/output/smp_spinkick"
!mkdir -p {OUT}
!python kaggle/wandb_upload.py \
    --project mimickit-smp \
    --download smp_spinkick_s2_files:latest \
    --dest {OUT}
!ls -lh {OUT}
```

```python
# Cell 6 — render + đẩy lên chính run đã train
RUN_ID = "m6rv7ht3"      # id lấy từ URL của run training

!python kaggle/make_videos.py \
    --out_dir      {OUT} \
    --motion_file  data/motions/humanoid/humanoid_spinkick.pkl \
    --wandb_project mimickit-smp \
    --wandb_run_id {RUN_ID} \
    --steps 300
```

Cell 5 đổ thẳng `model.pt` vào `OUT`, đúng chỗ `make_videos.py` tìm mặc định, nên Cell 6 không cần
`--model_file`. Sau đó dùng lại đoạn `show()` ở Cell 8 để xem ngay trong notebook.

Nếu Cell 6 báo lỗi ở stage rollout: thêm `--device cuda:0` (mặc định đã vậy) và kiểm tra accelerator
đã bật. Lỗi ở stage render thì đổi `--gl osmesa`. Muốn tách bạch để soi: chạy `--no_upload` trước cho
ra file, rồi mới upload bằng `kaggle/wandb_upload.py --run_id {RUN_ID} --files {OUT}/*.mp4`.

---

## Cell 9 — Đẩy checkpoint + log lên WandB

Video đã lên ở Cell 8 rồi. Cell này để giữ những thứ còn lại — quan trọng nhất là `model.pt`, vì
đó là thứ bạn cần để **train tiếp ở session sau**:

```python
!python kaggle/wandb_upload.py \
    --project mimickit-smp \
    --run_name smp_spinkick_files \
    --files {OUT}/model.pt {OUT}/log.txt \
            {OUT}/agent_config.yaml {OUT}/env_config.yaml {OUT}/engine_config.yaml
```

Tất cả được gói vào một WandB Artifact nên không mất khi session Kaggle kết thúc. Thêm
`--run_id {RUN_ID}` nếu muốn gắn vào chính run training.

Nếu có train prior ở Cell 5, upload luôn sample của nó để soi prior:

```python
!python kaggle/wandb_upload.py --project mimickit-smp --run_name smp_prior_spinkick \
    --dir /kaggle/working/output/smp_prior_spinkick
```

---

## Cell 10 — Giữ lại kết quả

`/kaggle/working` chỉ tồn tại tới hết session. Ba cách giữ, dùng cách nào cũng được:

```python
# a) nén nhỏ lại để tải về từ Output panel
!cd /kaggle/working && tar czf smp_spinkick.tar.gz output/smp_spinkick --exclude=int_models
```

- b) WandB Artifact ở Cell 9 (đã xong).
- c) Save Version → notebook output thành dataset cho session sau attach vào để train tiếp.

---

## Đọc metric: `Train_Return = 0` là bình thường

Với SMP single-clip, **`Train_Return` và `Test_Return` luôn bằng 0** — không phải bạn cấu hình sai.
Lý do nằm ở chỗ reward được cộng vào lúc nào:

- `AMPEnv._update_reward()` (mà `SMPEnv` kế thừa) là hàm rỗng — env **không bao giờ** ghi vào
  `_reward_buf`, nên reward trả về mỗi step là 0.
- Return tracker (`Train_Return` / `Test_Return`) cộng dồn đúng cái reward-mỗi-step đó → 0.
- SMP reward được tính **sau** khi rollout xong, trong `SMPAgent._compute_rewards()`: nó lấy
  `disc_obs` trong experience buffer, chấm bằng diffusion prior, rồi *ghi đè* trường `reward` của
  buffer bằng `task_reward_weight * task_r + smp_reward_weight * smp_r`. PPO học từ giá trị này,
  nhưng return tracker thì không thấy.
- Với `smp_humanoid_agent.yaml`, `task_reward_weight: 0.0` nên phần task reward cũng bằng 0 —
  không có task nào cả, chỉ có "giống clip tới đâu".

Vậy nhìn vào đâu. Bảng dưới là các metric có trên WandB, xếp theo mức quan trọng:

| Metric | Kỳ vọng | Ý nghĩa |
| --- | --- | --- |
| **`Fail_Frac`** | **giảm về 0** | tỉ lệ episode kết thúc vì **ngã** (`DoneFlags.FAIL`) trên tổng số episode kết thúc. Chỉ số thẳng thắn nhất cho câu hỏi "policy đứng được chưa" |
| **`Ep_Len_Frac`** | **tăng về 1.0** | độ dài episode trung bình / độ dài tối đa (300 step). 0.14 = ngã sau 1.4 s |
| **`Timeout_Frac`** | tăng về 1.0 | phần bù của `Fail_Frac` — episode sống hết 10 s |
| `Train_Episode_Length` / `Test_Episode_Length` | tăng tới 300 | như trên nhưng tính bằng step. Giờ được **in ra console** luôn, không còn `quiet` |
| `Sds_Loss_Mean` | **giảm** | SDS loss thô — motion càng nằm trong phân phối của prior thì càng nhỏ. Đây là thang **tuyệt đối**, so sánh được giữa các run |
| `Sds_Loss_Norm_Mean` | giảm chậm | giá trị thật sự đi vào `exp(-x * sds_loss_scale)` |
| `Sds_Norm_Scale` | giảm theo `Sds_Loss_Mean` | mẫu số của normalizer. Xem cùng 2 dòng trên thì hiểu ngay vì sao reward đứng yên |
| `Smp_Reward_Mean` | tăng rồi bão hoà | reward trung bình. Bão hoà **không** = hết học, xem đoạn dưới |
| `Smp_Reward_Max` / `Smp_Reward_P90` | tăng | nếu mean đứng mà P90 tăng → có một nhóm frame tốt hẳn lên |
| `Reward_Mean` | = `Smp_Reward_Mean` khi `smp_reward_weight=1` | reward tổng PPO nhận, sau khi cân với task reward |
| `Terminations` | — | số episode kết thúc trong iteration, để biết mẫu số của các `*_Frac` có đủ lớn không |
| `Critic_Loss`, `Clip_Frac`, `Imp_Ratio` | ổn định | `Clip_Frac` ~0.1–0.2 và `Imp_Ratio` ~1.0 là PPO khoẻ |

**Vì sao `Smp_Reward_Mean` bão hoà mà vẫn đang học:** `DiffNormalizer` chia SDS loss cho **trung
bình động tuyệt đối của chính nó** (cumulative, không reset). Loss giảm thì mẫu số cũng giảm theo,
nên tỉ số — và reward — đứng yên. Đó là lý do cặp `Sds_Loss_Mean` (tử) + `Sds_Norm_Scale` (mẫu)
được log riêng: **cứ nhìn `Sds_Loss_Mean` và `Fail_Frac`, đừng đánh giá tiến độ qua reward.**

---

## Train bao nhiêu samples là đủ?

Số đo thực tế từ run đầu tiên (`smp_spinkick_humanoid`, 2× T4, `num_envs 1024`/GPU):

| | |
| --- | --- |
| Tốc độ | **~11.500 samples/s** |
| 60M samples | 915 iteration, **87 phút** |
| Ngân sách 1 session Kaggle (9 h) | **~370M samples** |
| Kết quả sau 60M | `Sds_Loss_Mean` 2.41 → 0.93 (vẫn đang giảm), ep_len 17 → 42 step (**mới 14% của 300**) |

Kết luận: **60M samples là quá ngắn cho mimic.** Nó đủ để thấy xu hướng đúng, không đủ để ra động
tác. Đặt `--max_samples` cho chạy gần hết session:

```python
!python mimickit/run.py --arg_file args/smp_humanoid_kaggle_args.txt \
    --mode train --num_envs 1024 --max_samples 350000000 \
    --logger wandb --out_dir /kaggle/working/output/smp_spinkick \
    --devices cuda:0 cuda:1
```

Nối nhiều session bằng `--model_file`. Checkpoint lấy từ WandB Artifact của session trước
(Cell 9), hoặc từ notebook output đã Save Version rồi attach làm input:

```python
# Kéo checkpoint của session trước từ WandB Artifact về
!python kaggle/wandb_upload.py --project mimickit-smp \
    --download smp_spinkick_files:latest --dest /kaggle/working/prev

!python mimickit/run.py --arg_file args/smp_humanoid_kaggle_args.txt \
    --mode train --num_envs 1024 --max_samples 350000000 \
    --model_file /kaggle/working/prev/model.pt \
    --logger wandb --out_dir /kaggle/working/output/smp_spinkick_s2 \
    --devices cuda:0 cuda:1
```

> `--model_file` chỉ nạp trọng số model (actor/critic/normalizer) — bộ đếm `Samples` và optimizer
> state bắt đầu lại từ 0, nên trên WandB session 2 là một đường cong mới. Muốn nhìn liền mạch thì so
> `Sds_Loss_Mean` giữa các run, đừng so theo trục `Samples`.

---

## Tinh chỉnh khi kết quả chưa ổn

Sửa trong [`data/agents/smp_humanoid_agent.yaml`](../data/agents/smp_humanoid_agent.yaml). Thứ tự ưu
tiên theo README_SMP: `smp_reward_weight` > `sds_loss_scale` >= `diffusion_steps`.

| Triệu chứng | Thử |
| --- | --- |
| Ngã liên tục (`Fail_Frac` ~1.0, `Ep_Len_Frac` không tăng) | train thêm samples trước đã; sau đó giảm `action_std` (0.05 → 0.03); tăng `num_envs` nếu còn VRAM |
| Chuyển động mượt nhưng "nhạt", không ra spinkick | tăng `sds_loss_scale` (6 → 8–10) |
| Giật, run rẩy | giảm `sds_loss_scale`; thêm `action_reg_weight` nhỏ (vd 0.01) |
| Học chậm | tăng `steps_per_iter` (32 → 64) để mỗi batch nhiều data hơn |

Single clip nên `enable_gsi: False` và state init lấy từ `motion_file` trong
[`data/envs/smp_humanoid_env.yaml`](../data/envs/smp_humanoid_env.yaml) — muốn đổi clip (cartwheel,
roll…) thì đổi `motion_file`, đổi `contact_bodies` cho hợp (xem comment sẵn trong file) **và train
lại prior**.

---

## Lỗi hay gặp

| Lỗi | Nguyên nhân / cách xử lý |
| --- | --- |
| `Train_Return` / `Test_Return` luôn = 0 | **không phải lỗi** — xem mục [Đọc metric](#đọc-metric-trainreturn--0-là-bình-thường) |
| Upload WandB xong mà tab Media trống | mp4 chưa được tạo, `wandb_upload.py` chỉ in `[WARN] not a file` rồi bỏ qua. Dùng `kaggle/make_videos.py` — nó dừng ngay khi một bước fail |
| `--video true` + 2 GPU crash ở logger | đã sửa: `_mp_aggregate` giờ bỏ qua entry không phải số (video) thay vì nhét vào all-reduce |
| `ModuleNotFoundError: isaacgym` / `isaaclab` | thiếu `--engine_config data/engines/newton_engine.yaml` |
| `SMP prior env mismatch for <key>` | prior train với env config khác env đang dùng → train lại prior, hoặc trỏ `smp_prior_cfg` về đúng prior |
| `FileNotFoundError: data/models/smp_priors/...pt` | data pack chưa attach/link → chạy lại `kaggle/prepare_data.py`, hoặc train prior ở Cell 5 |
| CUDA OOM | giảm `--num_envs` (1024 → 512) — nhớ con số này tính cho **mỗi** GPU |
| `Address already in use` khi train 2 GPU | port ngẫu nhiên bị trùng → thêm `--master_port 6543` |
| Chỉ thấy 1 GPU hoạt động trong `nvidia-smi` | accelerator đang là P100/T4 đơn, hoặc quên `--devices cuda:0 cuda:1` |
| Process thứ 2 chết mà process 1 vẫn chạy | thường là OOM ở cuda:1 — giảm `--num_envs`, đọc log của cả 2 process |
| `WarpCodegenKeyError: ... J_kj` | warp-lang sai version → `pip install warp-lang==1.15.0` rồi **restart session** |
| Treo lúc khởi động, không log gì | warp đang JIT-compile kernel lần đầu, vài phút, sau đó được cache |
| `motion has N dofs, ... expects M` (render) | render nhầm robot xml — humanoid native phải dùng `data/assets/humanoid/humanoid.xml` |
| `EGLError` lúc script render thoát | destructor của MuJoCo, mp4 đã ghi xong rồi — bỏ qua. Nếu render thật sự fail: thêm `--gl osmesa` |
| `ffmpeg: not found` | `apt-get install -y ffmpeg` (đã có trong `setup.sh`, chỉ gặp nếu bỏ qua Cell 3) |
| wandb hỏi login | `WANDB_API_KEY` chưa vào env — chạy lại Cell 1 **trước** khi chạy cell train |
| Run trên WandB tên là `log` | bản cũ; hiện `wandb_logger` lấy tên theo `WANDB_NAME` hoặc tên thư mục `out_dir` |
