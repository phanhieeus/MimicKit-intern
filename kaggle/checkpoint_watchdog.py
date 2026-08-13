#!/usr/bin/env python3
"""Copy the training checkpoint to WandB every few minutes, in the background.

The checkpoint only exists in /kaggle/working, and /kaggle/working only becomes
a saved Output when the notebook finishes cleanly. A batch version that runs past
the 12 h limit is marked failed, and a failed version is not something to count on
for output -- so an overrun can cost the entire run's weights even though every
metric made it to WandB. run.py itself uploads nothing while training.

This watchdog closes that hole. Start it *before* the training cell; because a
notebook cell does not wait on a detached subprocess, it keeps running alongside
the trainer:

    import subprocess
    wd = subprocess.Popen([
        "python", "kaggle/checkpoint_watchdog.py",
        "--model_file", "/kaggle/working/output/smp_m3_spinkick/model.pt",
        "--project", "mimickit-smp",
        "--run_name", "smp_m3_spinkick_ckpt",
        "--interval", "1200",
    ])

    # ... training cell ...

    wd.terminate()   # after training, in the same or a later cell

One WandB run holds every upload as a new artifact version, so the run list stays
clean and `:latest` is always the newest checkpoint. Recover with:

    python kaggle/wandb_upload.py --project mimickit-smp \
        --download smp_m3_spinkick_ckpt_model:latest --dest /kaggle/working/recovered

Uploads are skipped while the file's mtime is unchanged, so a stalled trainer
costs nothing.
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_file", required=True, help="Checkpoint to watch.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mimickit"))
    parser.add_argument("--run_name", default="checkpoint_watchdog")
    parser.add_argument("--interval", type=float, default=1200.0,
                        help="Seconds between checks (default 20 min).")
    parser.add_argument("--max_hours", type=float, default=13.0,
                        help="Give up after this long, so the process cannot outlive the session.")
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        print("[watchdog] WANDB_API_KEY is not set, nothing to do.")
        return 1

    import wandb

    artifact_name = "{}_model".format(args.run_name)
    run = wandb.init(project=args.project, name=args.run_name, job_type="checkpoint")
    print("[watchdog] watching {} every {:.0f}s".format(args.model_file, args.interval))

    deadline = time.time() + args.max_hours * 3600.0
    last_mtime = None
    uploads = 0

    try:
        while time.time() < deadline:
            time.sleep(args.interval)

            if not os.path.isfile(args.model_file):
                print("[watchdog] no checkpoint yet at {}".format(args.model_file))
                continue

            mtime = os.path.getmtime(args.model_file)
            if mtime == last_mtime:
                # The trainer overwrites model.pt every iters_per_output; an
                # unchanged mtime means nothing new has been written since the
                # last upload, so uploading again would just burn bandwidth.
                print("[watchdog] unchanged, skipping")
                continue

            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(args.model_file)
            run.log_artifact(artifact)
            last_mtime = mtime
            uploads += 1
            print("[watchdog] uploaded v{} ({:.1f} MB, mtime {})".format(
                uploads - 1, os.path.getsize(args.model_file) / 1e6,
                time.strftime("%H:%M:%S", time.localtime(mtime))))
    except KeyboardInterrupt:
        print("[watchdog] stopping on request")
    finally:
        run.finish()

    print("[watchdog] done, {} upload(s)".format(uploads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
