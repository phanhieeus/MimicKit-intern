#!/usr/bin/env python3
"""Push files produced by a run -- mp4s, logs, checkpoints -- to Weights & Biases.

Training itself logs metrics through `--logger wandb`. The mp4s, however, are
made *after* training by tools/play_policy_to_mp4.py + tools/render_robot_video.py,
so they need a separate upload step. This script does that: every .mp4 is logged
as a WandB video (playable in the run page), and every file is also stored in a
WandB artifact so nothing is lost when the Kaggle session ends.

    export WANDB_API_KEY=...        # on Kaggle: from Add-ons -> Secrets
    python kaggle/wandb_upload.py \
        --project mimickit \
        --run_name smp_spinkick_media \
        --files output/smp_spinkick/policy.mp4 \
                output/smp_spinkick/reference.mp4 \
                output/smp_spinkick/log.txt \
                output/smp_spinkick/model.pt

Pass `--dir <path>` to upload every file under a directory instead of listing
them, and `--run_id <id>` to attach the media to an existing run (the training
run) rather than creating a new one -- copy the id from that run's URL.

The reverse direction pulls an artifact back down, which is how a checkpoint
gets from one Kaggle session into the next:

    python kaggle/wandb_upload.py --download smp_spinkick_files:latest \
        --dest /kaggle/working/prev --project mimickit-smp
"""

import argparse
import os
import sys

VIDEO_EXTS = (".mp4", ".gif", ".webm")


def collect_files(files, dirs, max_bytes):
    out = []
    for path in files:
        if not os.path.isfile(path):
            print("[WARN] not a file, skipped: {}".format(path))
            continue
        out.append(path)

    for root_dir in dirs:
        if not os.path.isdir(root_dir):
            print("[WARN] not a directory, skipped: {}".format(root_dir))
            continue
        for root, _, names in os.walk(root_dir):
            for name in sorted(names):
                out.append(os.path.join(root, name))

    kept = []
    for path in out:
        size = os.path.getsize(path)
        if max_bytes > 0 and size > max_bytes:
            print("[WARN] {} is {:.0f} MB > --max_file_mb, skipped".format(path, size / 1e6))
            continue
        kept.append(path)
    return kept


def download_artifact(args):
    """Fetch an artifact into --dest. Uses the public API, so no run is created."""
    import wandb

    ref = args.download
    if "/" not in ref:
        ref = "{}/{}".format(args.project, ref)

    api = wandb.Api()
    artifact = api.artifact(ref)
    path = artifact.download(root=args.dest)

    print("Downloaded {} -> {}".format(ref, path))
    for root, _, names in os.walk(path):
        for name in sorted(names):
            full = os.path.join(root, name)
            print("  {} ({:.1f} MB)".format(full, os.path.getsize(full) / 1e6))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mimickit"))
    parser.add_argument("--run_name", default=None, help="Name for a new run.")
    parser.add_argument("--run_id", default=None,
                        help="Attach to this existing run instead of creating a new one.")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--dir", nargs="*", default=[], dest="dirs",
                        help="Upload every file under these directories.")
    parser.add_argument("--artifact_name", default=None,
                        help="Artifact name (default: <run_name>_files).")
    parser.add_argument("--fps", type=int, default=30, help="fps declared for logged videos.")
    parser.add_argument("--max_file_mb", type=float, default=500.0,
                        help="Skip files larger than this. 0 = no limit.")
    parser.add_argument("--download", default=None, metavar="ARTIFACT",
                        help="Download an artifact (e.g. 'smp_spinkick_files:latest') instead of uploading.")
    parser.add_argument("--dest", default="wandb_download",
                        help="Where --download puts the files.")
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        print("ERROR: WANDB_API_KEY is not set.")
        return 1

    if args.download:
        return download_artifact(args)

    if not args.files and not args.dirs:
        parser.error("nothing to upload: pass --files and/or --dir")

    paths = collect_files(args.files, args.dirs, args.max_file_mb * 1e6)
    if not paths:
        print("ERROR: no uploadable files found.")
        return 1

    import wandb

    run_name = args.run_name or "media"
    if args.run_id is not None:
        run = wandb.init(project=args.project, id=args.run_id, resume="allow")
    else:
        run = wandb.init(project=args.project, name=run_name)

    artifact_name = args.artifact_name or "{}_files".format(run_name)
    artifact = wandb.Artifact(artifact_name, type="run_output")

    videos = dict()
    for path in paths:
        artifact.add_file(path)
        if path.lower().endswith(VIDEO_EXTS):
            key = os.path.splitext(os.path.basename(path))[0]
            videos[key] = wandb.Video(path, fps=args.fps, format="mp4")
        print("  + {}".format(path))

    if videos:
        run.log(videos)
    run.log_artifact(artifact)
    run.finish()

    print("Uploaded {} file(s), {} video(s) to project '{}'.".format(
        len(paths), len(videos), args.project))
    return 0


if __name__ == "__main__":
    sys.exit(main())
