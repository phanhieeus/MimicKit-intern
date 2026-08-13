#!/usr/bin/env python3
"""Roll out a trained policy, render it to mp4, and push the result to WandB.

One command instead of the play_policy_to_mp4 -> render_robot_video -> upload
chain, because that chain fails silently in a notebook: if the rollout dies, the
render has no input, and the upload then skips the missing mp4 with a warning
nobody reads. Here every stage is checked and a failure stops the script.

Three clips come out of a run, and the comparison between them is the point:

    reference_data.mp4  the source clip, straight from the motion file
    policy.mp4          what the policy does in the physics sim
    reference_sim.mp4   the env's reference character over the same episode

    export WANDB_API_KEY=...
    python kaggle/make_videos.py \
        --out_dir output/smp_spinkick \
        --env_config   data/envs/smp_humanoid_env.yaml \
        --agent_config data/agents/smp_humanoid_agent.yaml \
        --char_file    data/assets/humanoid/humanoid.xml \
        --motion_file  data/motions/humanoid/humanoid_spinkick.pkl \
        --wandb_run_id <id of the training run>

`--wandb_run_id` puts the videos in the training run itself, next to its curves;
without it a separate run named by `--wandb_run_name` is created. Pass
`--no_upload` to only produce the files.

With `--int_models` the same is done for every intermediate checkpoint in
`<out_dir>/int_models`, which gives a progression -- early checkpoints falling
over, later ones (hopefully) not.
"""

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(name, cmd):
    print("\n=== {} ===".format(name))
    print("$ {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError("{} failed with exit code {}".format(name, result.returncode))
    return


def render(motion_file, char_file, out_mp4, fps, gl):
    run_step("render {}".format(os.path.basename(out_mp4)), [
        sys.executable, "tools/render_robot_video.py",
        "--motion", motion_file,
        "--robot-xml", char_file,
        "--output", out_mp4,
        "--fps", str(fps),
        "--gl", gl,
    ])
    if not os.path.isfile(out_mp4):
        raise RuntimeError("renderer reported success but {} does not exist".format(out_mp4))
    print("    {} ({:.1f} MB)".format(out_mp4, os.path.getsize(out_mp4) / 1e6))
    return out_mp4


def make_policy_videos(model_file, tag, args):
    """Roll out one checkpoint and render it. Returns the mp4 paths."""
    playback_dir = os.path.join(args.out_dir, "playback_{}".format(tag))

    run_step("rollout {}".format(tag), [
        sys.executable, "tools/play_policy_to_mp4.py",
        "--env_config", args.env_config,
        "--agent_config", args.agent_config,
        "--engine_config", args.engine_config,
        "--model_file", model_file,
        "--out_dir", playback_dir,
        "--steps", str(args.steps),
        "--device", args.device,
    ])

    policy_pkl = os.path.join(playback_dir, "policy.pkl")
    if not os.path.isfile(policy_pkl):
        raise RuntimeError("rollout produced no policy.pkl in {}".format(playback_dir))

    videos = [render(policy_pkl, args.char_file,
                     os.path.join(args.out_dir, "policy_{}.mp4".format(tag)),
                     args.fps, args.gl)]

    ref_pkl = os.path.join(playback_dir, "reference.pkl")
    if os.path.isfile(ref_pkl):
        videos.append(render(ref_pkl, args.char_file,
                             os.path.join(args.out_dir, "reference_sim_{}.mp4".format(tag)),
                             args.fps, args.gl))
    else:
        print("[WARN] no reference character recorded for this env")
    return videos


def upload(videos, args):
    import wandb

    if args.wandb_run_id:
        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow")
        print("Attaching {} video(s) to existing run {}".format(len(videos), args.wandb_run_id))
    else:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    artifact = wandb.Artifact("{}_videos".format(args.wandb_run_name), type="run_output")
    payload = dict()
    for path in videos:
        key = "video/{}".format(os.path.splitext(os.path.basename(path))[0])
        payload[key] = wandb.Video(path, fps=args.fps, format="mp4")
        artifact.add_file(path)

    run.log(payload)
    run.log_artifact(artifact)
    run.finish()
    print("Uploaded {} video(s) to project '{}'.".format(len(videos), args.wandb_project))
    return


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True, help="Training output dir; videos land here too.")
    parser.add_argument("--env_config", default="data/envs/smp_humanoid_env.yaml")
    parser.add_argument("--agent_config", default="data/agents/smp_humanoid_agent.yaml")
    parser.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    parser.add_argument("--char_file", default="data/assets/humanoid/humanoid.xml",
                        help="MJCF used for rendering. Must match the env's character.")
    parser.add_argument("--motion_file", default="",
                        help="Source clip to render as the ground-truth video. Skipped if empty.")
    parser.add_argument("--model_file", default="",
                        help="Checkpoint to play. Default: <out_dir>/model.pt")
    parser.add_argument("--int_models", action="store_true",
                        help="Also render every checkpoint in <out_dir>/int_models.")
    parser.add_argument("--steps", type=int, default=300, help="Env steps per rollout.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_upload", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "mimickit"))
    parser.add_argument("--wandb_run_id", default=None,
                        help="Log into this existing run (copy the id from its URL).")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    if args.wandb_run_name is None:
        args.wandb_run_name = "{}_videos".format(os.path.basename(args.out_dir))
    model_file = args.model_file or os.path.join(args.out_dir, "model.pt")

    if not os.path.isfile(model_file):
        print("ERROR: no checkpoint at {}".format(model_file))
        return 1
    if not os.path.isfile(args.char_file):
        print("ERROR: no character file at {}".format(args.char_file))
        return 1
    if not args.no_upload and "WANDB_API_KEY" not in os.environ:
        print("ERROR: WANDB_API_KEY is not set (use --no_upload to only write files).")
        return 1

    videos = []

    # The source clip first: it is the cheapest stage and it needs no simulator,
    # so if the GL stack is broken this fails before an expensive rollout.
    if args.motion_file:
        videos.append(render(args.motion_file, args.char_file,
                             os.path.join(args.out_dir, "reference_data.mp4"),
                             args.fps, args.gl))

    videos += make_policy_videos(model_file, "final", args)

    if args.int_models:
        int_dir = os.path.join(args.out_dir, "int_models")
        for path in sorted(glob.glob(os.path.join(int_dir, "*.pt"))):
            tag = os.path.splitext(os.path.basename(path))[0]
            try:
                videos += make_policy_videos(path, tag, args)
            except RuntimeError as e:
                print("[WARN] skipping {}: {}".format(tag, e))

    print("\n=== {} video(s) ===".format(len(videos)))
    for path in videos:
        print("  {}".format(path))

    if not args.no_upload:
        upload(videos, args)

    # Scoring runs last and never blocks the upload: the videos and the
    # checkpoint are the things worth losing sleep over, and a scoring bug must
    # not take them down with it. The exit code still carries the verdict.
    return score_rollout(args)


def score_rollout(args):
    """Grade the final rollout against the clip, and say so on WandB.

    Ep_Len_Frac and Sds_Loss cannot distinguish imitation from mode collapse --
    the M3.1 spinkick run finished at Ep_Len_Frac 0.827 with a lower Sds_Loss
    than the converged humanoid while balancing on one leg and never kicking. So
    the videos are not the last word either; something has to measure the
    rollout. See tools/motion_quality.py.
    """
    policy_pkl = os.path.join(args.out_dir, "playback_final", "policy.pkl")
    if not args.motion_file or not os.path.isfile(policy_pkl):
        return 0

    cmd = [sys.executable, "tools/motion_quality.py",
           "--policy", policy_pkl, "--reference", args.motion_file]
    if not args.no_upload:
        cmd += ["--wandb_project", args.wandb_project,
                "--wandb_run_name", "{}_quality".format(os.path.basename(args.out_dir))]
        if args.wandb_run_id:
            cmd += ["--wandb_run_id", args.wandb_run_id]

    print("\n=== motion quality ===")
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
