#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roll out a trained policy and write it as a clip you can render to mp4.

Why not ``run.py --video true``: that path records through
``newton.viewer.ViewerGL`` and hands the frames to the logger, which for
``--logger tb`` embeds them in the TensorBoard event file. No mp4 lands on disk,
and the whole thing needs a working GL stack plus Xvfb -- the two most fragile
things in a Kaggle image.

This instead records the *state* the policy produces -- root pose and joint
angles, one env, one episode -- and writes it in MimicKit clip format. Rendering
is then a separate, purely offline step through ``render_robot_video.py``, which
draws with MuJoCo over EGL and has no dependency on the simulator that produced
the motion.

The reference character is recorded too, into a second file. Playing the two
side by side is the point: a tracking policy that has learnt the average pose of
the clip looks fine alone and obviously wrong next to the reference.

    # from the MimicKit repo root, after training
    python tools/play_policy_to_mp4.py \\
        --env_config   data/envs/smp_vr_m3_1_env.yaml \\
        --agent_config data/agents/smp_vr_m3_1_agent.yaml \\
        --engine_config data/engines/newton_engine.yaml \\
        --model_file   output/smp_zombie/model.pt \\
        --out_dir      output/smp_zombie/playback \\
        --steps 600

Then:

    python tools/render_robot_video.py \\
        --motion output/smp_zombie/playback/policy.pkl \\
        --robot-xml data/assets/vr_m3_1/vr_m3_1.xml \\
        --output policy.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# MimicKit's modules import each other as `envs.foo` / `util.bar`, which only
# resolves with the `mimickit/` package directory itself on the path.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "mimickit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _quat_to_expmap(q_xyzw: np.ndarray) -> np.ndarray:
    """MimicKit frames store the root rotation as an exponential map."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    xyz, w = q[..., :3], q[..., 3]
    # Keep the canonical hemisphere so the log does not jump between frames.
    flip = w < 0
    xyz = np.where(flip[..., None], -xyz, xyz)
    w = np.where(flip, -w, w)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / (sin_half[..., None] + 1e-9)
    return axis * angle[..., None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env_config", required=True)
    ap.add_argument("--agent_config", required=True)
    ap.add_argument("--engine_config", required=True)
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=600, help="Env steps to record.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Clip fps. 0 = derive from the env's control timestep.")
    args = ap.parse_args()

    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import util.mp_util as mp_util

    # Playback is single-process by design: one env, one episode, one GPU. But
    # mp_util must still be initialised, because `is_root_proc()` reads the
    # globals it sets and several agent methods gate on it. Signature is
    # (rank, num_procs, device, master_port); with num_procs == 1 it skips
    # torch.distributed entirely and the port is unused.
    mp_util.init(0, 1, args.device, 0)

    device = args.device if torch.cuda.is_available() else "cpu"
    env = env_builder.build_env(args.env_config, args.engine_config,
                                num_envs=1, device=device,
                                visualize=False, record_video=False)
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file)
    agent.eval()

    import learning.base_agent as base_agent
    agent.set_mode(base_agent.AgentMode.TEST)

    engine = env._engine
    fps = args.fps or 1.0 / float(engine.get_timestep())

    pol_frames: list[np.ndarray] = []
    ref_frames: list[np.ndarray] = []
    n_done = 0

    def snapshot() -> None:
        root_pos = engine.get_root_pos(0)[0].detach().cpu().numpy()
        root_rot = engine.get_root_rot(0)[0].detach().cpu().numpy()  # xyzw
        dof_pos = engine.get_dof_pos(0)[0].detach().cpu().numpy()
        pol_frames.append(
            np.concatenate([root_pos, _quat_to_expmap(root_rot), dof_pos])
        )
        # The reference character the env tracks against, if this env has one.
        if hasattr(env, "_ref_root_pos"):
            r_pos = env._ref_root_pos[0].detach().cpu().numpy()
            r_rot = env._ref_root_rot[0].detach().cpu().numpy()
            r_dof = env._ref_dof_pos[0].detach().cpu().numpy()
            ref_frames.append(
                np.concatenate([r_pos, _quat_to_expmap(r_rot), r_dof])
            )

    with torch.no_grad():
        obs, info = agent._reset_envs()
        snapshot()
        for _ in range(args.steps):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = agent._step_env(action)
            snapshot()
            if bool(done[0].item()):
                n_done += 1
            obs, info = agent._reset_done_envs(done)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, frames: list[np.ndarray]) -> None:
        payload = {"loop_mode": 0, "fps": float(fps),
                   "frames": np.asarray(frames, dtype=np.float64).tolist()}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[INFO] Wrote {path}  ({len(frames)} frames @ {fps:g} fps, "
              f"width {len(frames[0])})")

    write(out_dir / "policy.pkl", pol_frames)
    if ref_frames:
        write(out_dir / "reference.pkl", ref_frames)
    else:
        print("[WARN] This env exposes no reference character; only the policy "
              "was recorded.")

    print(f"[INFO] episodes terminated during the recording: {n_done}")
    if n_done > 0:
        print("       Early termination means the policy fell or hit a "
              "contact_bodies link. The clip contains the resets.")

    print("\nRender with:")
    print(f"  python tools/render_robot_video.py --motion {out_dir}/policy.pkl \\")
    print("      --robot-xml data/assets/vr_m3_1/vr_m3_1.xml --output policy.mp4")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
