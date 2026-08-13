#!/usr/bin/env python3
"""Measure the SDS loss of things whose answer we already know, so the number means something.

Sds_Loss_Mean is logged every iteration and read as if it were interpretable. It
is not, on its own. It is the error a diffusion prior makes when re-denoising a
motion window, which has a floor it can never go below and a scale set by the
prior, the observation space and the clip. Two runs with different priors cannot
be compared on it -- and that is exactly the mistake that made the M3.1 spinkick
run look like a success: it finished at 0.1696 against the humanoid's converged
0.186, which was read as "imitates better" when the two numbers share no scale
and the robot was in fact never kicking.

This gives the number a ruler by scoring the same quantity on references:

  * the clip itself       -- the floor a perfect policy would reach
  * the prior's own samples -- what the prior considers typical, and unlike the
                            clip not something it was trained on
  * any rollout           -- policy.pkl from play_policy_to_mp4.py

    python tools/sds_calibrate.py \\
        --env_config   data/envs/smp_vr_m3_1_spinkick_env.yaml \\
        --agent_config data/agents/smp_vr_m3_1_spinkick_agent.yaml \\
        --prior_model  /path/to/prior/model.pt \\
        --extra "prior samples"=/path/to/samples/motion_*.pkl \\
        --extra "policy 320M"=/path/to/playback_final/policy.pkl

Everything runs on CPU; the prior is small and a few hundred windows is nothing.
No simulator is involved -- windows come from MotionLib and forward kinematics,
the same path amp_env._compute_disc_obs_demo uses to feed the discriminator.

Read the ratio, not the absolute. "The policy sits at 3.2x the clip's own floor"
is a sentence that survives a change of robot, clip or prior. "Sds_Loss is 0.17"
is not.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mimickit"))

import anim.mjcf_char_model as mjcf_char_model          # noqa: E402
import anim.motion_lib as motion_lib_mod                # noqa: E402
import envs.amp_env as amp_env                          # noqa: E402
from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: E402


def build_windows(motion_lib, char_model, env_cfg, n_windows, device, seed):
    """Reproduce amp_env._fetch_disc_demo_data + _compute_disc_obs_demo offline.

    Windows are spaced by the control timestep, not by clip frames -- that is the
    detail that makes the observation encode speed rather than pose order, and
    getting it wrong here would calibrate against the wrong thing.
    """
    steps = env_cfg["num_disc_obs_steps"]
    dt = 1.0 / 30.0                      # control_freq, asserted equal in the configs
    span = (steps - 1) * dt

    length = motion_lib.get_motion_length(torch.zeros(1, dtype=torch.long, device=device)).item()
    if length < span:
        raise ValueError("clip is {:.3f} s, shorter than one {:.3f} s window".format(length, span))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = span + torch.rand(n_windows, generator=gen).to(device) * max(length - span, 1e-6)

    motion_ids = torch.zeros(n_windows * steps, dtype=torch.long, device=device)
    offsets = -dt * torch.arange(0, steps, device=device, dtype=torch.float32)
    times = (t0.unsqueeze(-1) + torch.flip(offsets, dims=[0])).reshape(-1)

    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
        motion_lib.calc_motion_frame(motion_ids, times)
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    def fold(x):
        return x.reshape([n_windows, steps] + list(x.shape[1:]))

    root_pos, root_rot = fold(root_pos), fold(root_rot)
    root_vel, root_ang_vel = fold(root_vel), fold(root_ang_vel)
    joint_rot, dof_vel, body_pos = fold(joint_rot), fold(dof_vel), fold(body_pos)

    # _track_global_root() is enable_tar_obs and global_obs; enable_tar_obs is
    # False for pure SMP, so the reference frame is the window's last pose.
    ref_root_pos = root_pos[..., -1, :]
    ref_root_rot = root_rot[..., -1, :]

    key_names = env_cfg.get("key_bodies", [])
    if key_names:
        all_names = char_model.get_body_names()
        ids = [all_names.index(n) for n in key_names]
        key_pos = body_pos[..., ids, :]
    else:
        key_pos = torch.zeros([0], device=device)

    return amp_env.compute_disc_obs(
        ref_root_pos=ref_root_pos, ref_root_rot=ref_root_rot,
        root_pos=root_pos, root_rot=root_rot,
        root_vel=root_vel, root_ang_vel=root_ang_vel,
        joint_rot=joint_rot, dof_vel=dof_vel, key_pos=key_pos,
        global_obs=env_cfg["global_obs"],
        root_height_obs=env_cfg["root_height_obs"],
        dof_vel_obs=env_cfg["disc_dof_vel_obs"])


def score(prior, disc_obs, t_lst, repeats, seed):
    """Mean SDS loss over the windows, averaged over the noise draws.

    ESM_SDS_loss samples fresh noise every call, so a single pass is a noisy
    estimate of the thing we are trying to compare. Repeating and averaging costs
    nothing on a model this small.
    """
    steps = prior.num_disc_obs_steps
    x = prior.normalize(disc_obs.reshape(disc_obs.shape[0], steps, -1))
    x = x.reshape(disc_obs.shape[0], -1)

    per_window = []
    with torch.no_grad():
        for r in range(repeats):
            torch.manual_seed(seed + r)
            losses = prior.ESM_SDS_loss(norm_x_obs=x, t_lst=t_lst)   # (n, len(t_lst))
            per_window.append(losses.mean(dim=-1))
    return torch.stack(per_window).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env_config", required=True)
    parser.add_argument("--agent_config", required=True)
    parser.add_argument("--prior_model", required=True)
    parser.add_argument("--extra", action="append", default=[], metavar="LABEL=GLOB",
                        help="More clips to score, e.g. 'policy'=out/playback_final/policy.pkl. "
                             "A glob matching several files is treated as one set.")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8,
                        help="Noise draws to average over (default 8).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_cfg = yaml.safe_load(open(args.env_config))
    agent_cfg = yaml.safe_load(open(args.agent_config))
    prior_cfg = yaml.safe_load(open(agent_cfg["smp_prior_cfg"]))
    t_lst = agent_cfg["diffusion_steps"]
    device = torch.device(args.device)

    char = mjcf_char_model.MJCFCharModel(device)
    char.load(env_cfg["char_file"])

    def windows_for(path_or_paths, n):
        paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        chunks = []
        per = max(1, n // len(paths))
        for p in paths:
            lib = motion_lib_mod.MotionLib(p, char, device)
            chunks.append(build_windows(lib, char, env_cfg, per, device, args.seed))
        return torch.cat(chunks, dim=0)

    ref_obs = windows_for(env_cfg["motion_file"], args.windows)

    prior_cfg["input_dim"] = ref_obs.shape[-1]
    prior = TinyMDMModel(prior_cfg, device)
    state = torch.load(args.prior_model, map_location=device)
    prior.load_state_dict(state)
    prior.eval().to(device)

    rows = [("the clip itself (floor)", ref_obs)]
    for spec in args.extra:
        if "=" not in spec:
            print("[ERROR] --extra needs LABEL=PATH, got {!r}".format(spec), file=sys.stderr)
            return 2
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            print("[WARN] nothing matches {!r}, skipping".format(pattern))
            continue
        rows.append((label, windows_for(paths, args.windows)))

    print("prior      {}".format(args.prior_model))
    print("clip       {}".format(env_cfg["motion_file"]))
    print("obs        {} dims over {} steps, diffusion_steps {}".format(
        ref_obs.shape[-1], env_cfg["num_disc_obs_steps"], t_lst))
    print("averaging  {} noise draws per window\n".format(args.repeats))

    print("%-28s %8s %8s %8s %8s   %s" % ("", "windows", "mean", "std", "p90", "vs floor"))
    floor = None
    for label, obs in rows:
        per_window = score(prior, obs, t_lst, args.repeats, args.seed)
        mean = per_window.mean().item()
        if floor is None:
            floor = mean
        print("%-28s %8d %8.4f %8.4f %8.4f   %.2fx" % (
            label, len(obs), mean, per_window.std().item(),
            torch.quantile(per_window, 0.9).item(), mean / floor))

    print("\nRead the last column. The absolute values carry no meaning across "
          "priors, clips or robots; the ratio does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
