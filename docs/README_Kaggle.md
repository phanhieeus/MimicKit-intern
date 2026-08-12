# Running MimicKit on a Kaggle Notebook

Everything needed to go from `git clone` to `!python mimickit/run.py` inside a Kaggle notebook.
The ready-made notebook is [`kaggle/mimickit_kaggle.ipynb`](../kaggle/mimickit_kaggle.ipynb) —
upload it to Kaggle and run the cells top to bottom.

Step-by-step Vietnamese walkthrough: [HUONG_DAN_KAGGLE.md](HUONG_DAN_KAGGLE.md).
For the SMP single-clip recipe end to end — prior, PPO policy, mp4 comparison, WandB upload —
see [HUONG_DAN_SMP_KAGGLE.md](HUONG_DAN_SMP_KAGGLE.md).

## Which engine

**Newton**, not Isaac Gym. Isaac Gym ships as a manual download behind an NVIDIA login and only
supports Python ≤ 3.8; Kaggle runs Python 3.11 and blocks that download. Newton installs from PyPI
and runs on Kaggle's CUDA GPUs.

Consequence: every command needs `--engine_config data/engines/newton_engine.yaml`, because the
arg files in [`args/`](../args/) default to Isaac Gym. MimicKit's arg parser keeps the **first**
value it sees for a key and command-line args are parsed before the arg file, so a flag on the
command line always wins over the same flag in `--arg_file`.

## Notebook settings

- **Accelerator**: GPU T4 x2 or P100 (the code uses a single device unless you pass several to `--devices`).
- **Internet**: On — required for `git clone` and `pip install`.
- **Persistence**: only `/kaggle/working` survives as notebook output; put `--out_dir` there.

## One-time step: upload the data pack

Assets, motions and pretrained models (~534 MB) are not in git. Download them once from the
[OneDrive link in the README](../README.md#installation), then publish them as a Kaggle Dataset so
notebooks can attach them instead of re-downloading:

```bash
# from the repo root, with the pack extracted at data/MimicKit_Data/
pip install kaggle          # credentials in ~/.kaggle/kaggle.json
mkdir -p /tmp/mimickit_data && cp -r data/MimicKit_Data/* /tmp/mimickit_data/
kaggle datasets init -p /tmp/mimickit_data
# edit /tmp/mimickit_data/dataset-metadata.json -> set "title" and "id" (e.g. "<user>/mimickit-data")
kaggle datasets create -p /tmp/mimickit_data --dir-mode zip
```

Or just use the Kaggle web UI: *Datasets → New Dataset → upload a zip of `data/MimicKit_Data/`*.

Downloading the pack directly inside the notebook does not work: the SharePoint share link returns
`401 Access denied` to `wget`/`curl` even with `&download=1`, since it needs a browser session
cookie.

Then in the notebook, *Add Input → Datasets →* your `mimickit-data`.

`kaggle/prepare_data.py` locates the pack under `/kaggle/input` (it looks up to two levels deep for
a directory containing `assets/` and `motions/`) and symlinks its contents into `data/`. Files
already tracked in git — such as `data/assets/humanoid/humanoid.xml` — are left alone.

## The three cells

```python
# 1. clone
!git clone --depth 1 https://github.com/phanhieeus/MimicKit-intern.git /kaggle/working/MimicKit-intern
import os; os.chdir("/kaggle/working/MimicKit-intern")   # all config paths are repo-root relative

# 2. install (~3-4 min)
!bash kaggle/setup.sh

# 3. link the data pack
!python kaggle/prepare_data.py
```

For a private repo, store a GitHub token in *Add-ons → Secrets* as `GITHUB_TOKEN` and clone with
`https://{token}@github.com/...`.

`kaggle/setup.sh` installs `newton==1.0.0` (the version this codebase is tested against) plus
`mujoco` / `mujoco-warp`, which Newton's `SolverMuJoCo` needs and which are not pulled in as hard
dependencies. It deliberately does **not** install torch — Kaggle's preinstalled CUDA build must
stay, and `pip install -r requirements.txt` risks replacing it with a CPU wheel.

## Smoke test

```bash
!python mimickit/run.py \
    --arg_file args/view_motion_humanoid_args.txt \
    --engine_config data/engines/newton_engine.yaml \
    --num_envs 4 --mode test --test_episodes 1 \
    --visualize false --devices cuda:0
```

## Training

```bash
!python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --engine_config data/engines/newton_engine.yaml \
    --mode train --num_envs 1024 --max_samples 200000000 \
    --visualize false --video false --logger tb \
    --out_dir /kaggle/working/output/deepmimic_humanoid \
    --devices cuda:0
```

- `--visualize false` is required — the default is `true`, which tries to open a GL window.
- `--num_envs 1024` instead of the arg file's 4096: 4096 humanoids do not fit in a T4's 16 GB.
  Raise it on a P100/A100 or if `nvidia-smi` shows headroom.
- `--max_samples` bounds the run so it ends inside the session limit (9 h interactive / 12 h batch).
  Without it the run is open-ended and gets killed mid-training.
- Use `--devices cuda:0 cuda:1` on a T4 x2 instance for distributed training.

Checkpoints and `log.txt` land in `--out_dir`; keep it under `/kaggle/working` or the results are
lost when the session ends. With `--logger tb` the TensorBoard events file is written next to the
log and can be downloaded from the notebook output.

## Rendering and video

Headless video (`--video true`) drives Newton's OpenGL viewer through Xvfb — `setup.sh` installs
`xvfb` and the GL libraries, and `util/display.py` starts the virtual display automatically. Kaggle
GPU containers do not reliably expose a usable GL/EGL context, so treat this as best-effort: if it
fails, train with `--video false`. Interactive visualization (`--visualize true`) never works on
Kaggle — there is no display to attach to.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: isaacgym` | The engine config is still the Isaac Gym one. Pass `--engine_config data/engines/newton_engine.yaml`. |
| `FileNotFoundError: data/motions/...pkl` | Data pack not attached or not linked. Re-run `kaggle/prepare_data.py` and check `!ls /kaggle/input`. |
| `torch.cuda.is_available() == False` | A pip install replaced torch with a CPU wheel. Restart the session; do not `pip install -r requirements.txt`. |
| CUDA OOM during training | Lower `--num_envs`. |
| `mujoco_warp` import error from `SolverMuJoCo` | `pip install mujoco==3.5.0 mujoco-warp==3.5.0.2`. |
| `WarpCodegenKeyError: Referencing undefined symbol: J_kj` | warp-lang 1.16.0 cannot compile mujoco-warp 3.5.0.2's tiled solver kernel. Install `warp-lang==1.15.0` and restart the session. |
| Hangs at startup, no output | Warp is JIT-compiling its kernels on first run; this takes a few minutes and is cached afterwards. |
