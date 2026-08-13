#!/usr/bin/env python3
"""Train the SMP diffusion prior once, then reuse it in every later session.

The prior is 35 minutes of GPU that produces one small file, and until now that
file only ever existed in /kaggle/working -- so a session that died took it with
it, and every new session paid the 35 minutes again.

Paying twice is the smaller problem. The larger one is that two trainings of the
same config do not produce the same prior, and the prior *is* the reward
function: SMP's reward is exp(-sds_loss * scale) measured against it. Two runs
that trained their own priors are not comparable, which is exactly what happened
with the two M3.1 runs -- at an identical 131.1M samples one sat at Ep_Len_Frac
0.470 and the other at 0.288, with no other difference anyone could point to.

So this script makes the prior a cached artifact rather than a per-session build:

    python kaggle/prior_cache.py \
        --cfg_path tools/diffusion_model/config/tinymdm_vr_m3_1_spinkick_slow2.yaml \
        --out_dir /kaggle/working/output/smp_prior_vr_m3_1_spinkick_slow2 \
        --project mimickit-smp

Cache hit downloads and exits in seconds; cache miss trains, uploads, and leaves
the file in the same place either way, so the caller does not branch. Point
`smp_prior_model` in the agent yaml at <out_dir>/model.pt as before.

While training runs, a watchdog uploads whatever train_tinymdm.py has written so
far every few minutes -- it checkpoints every `output_iter` iterations, so a
session that dies at minute 30 still leaves something to resume from rather than
nothing.

The artifact carries a fingerprint of the config and the clip it was trained on.
A cached prior whose fingerprint does not match what you are asking for is a
stale prior, and reusing one silently would reintroduce the bug this script
exists to prevent -- so that case refuses by default. `--force_retrain` retrains
and overwrites, `--accept_stale` uses it anyway.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

import yaml


def fingerprint(cfg_path):
    """Hash the things that change what the prior learns.

    The config file and the clip, and nothing else. Paths are deliberately left
    out -- the same clip mounted at a different point on Kaggle is the same clip,
    and hashing its location would miss every cache it should hit.
    """
    with open(cfg_path, "rb") as handle:
        raw = handle.read()

    digest = hashlib.sha256()
    cfg = yaml.safe_load(raw)
    for key in sorted(cfg):
        if key in ("env_config", "motion_file"):
            continue
        digest.update("{}={}\n".format(key, cfg[key]).encode())

    for path in (cfg.get("motion_file"), cfg.get("env_config")):
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
        else:
            # A missing input is not a cache key we can trust, so make it one
            # that never matches instead of pretending the file was empty.
            digest.update("MISSING:{}".format(path).encode())
            digest.update(str(time.time()).encode())

    return digest.hexdigest()[:16]


def try_download(api, ref, dest, want_fp, accept_stale):
    """Return True if a usable prior now sits in `dest`."""
    try:
        artifact = api.artifact(ref)
    except Exception as exc:
        print("[prior_cache] no cached prior ({}: {})".format(ref, type(exc).__name__))
        return False

    have_fp = (artifact.metadata or {}).get("fingerprint")
    if have_fp != want_fp:
        print("[prior_cache] cached prior is STALE")
        print("               cached  {}".format(have_fp))
        print("               wanted  {}".format(want_fp))
        if not accept_stale:
            print("               refusing -- pass --accept_stale to use it anyway,")
            print("               or --force_retrain to build a fresh one.")
            return None                       # distinct from "absent"
        print("               --accept_stale given, using it")

    artifact.download(root=dest)
    model = os.path.join(dest, "model.pt")
    if not os.path.isfile(model):
        print("[prior_cache] artifact had no model.pt, will retrain")
        return False

    print("[prior_cache] reused {} -> {} ({:.1f} MB)".format(
        ref, model, os.path.getsize(model) / 1e6))
    return True


def start_watchdog(model_file, project, run_name, interval):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.Popen([
        sys.executable, os.path.join(repo_root, "kaggle", "checkpoint_watchdog.py"),
        "--model_file", model_file,
        "--project", project,
        "--run_name", run_name,
        "--interval", str(interval),
    ])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg_path", required=True, help="tinymdm_*.yaml for this prior.")
    parser.add_argument("--out_dir", required=True, help="Where model.pt ends up either way.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mimickit"))
    parser.add_argument("--artifact", default=None,
                        help="Artifact name. Defaults to the out_dir basename.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Ignore any cached prior and train a new one.")
    parser.add_argument("--accept_stale", action="store_true",
                        help="Use a cached prior whose fingerprint does not match.")
    parser.add_argument("--no_upload", action="store_true",
                        help="Train without publishing the result.")
    parser.add_argument("--watchdog_interval", type=float, default=300.0,
                        help="Seconds between mid-training uploads (default 5 min).")
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        print("[prior_cache] WANDB_API_KEY is not set; no cache, no upload.")
        if not args.force_retrain:
            print("               training anyway, but nothing will be saved.")

    name = args.artifact or os.path.basename(os.path.normpath(args.out_dir))
    ref = "{}/{}:latest".format(args.project, name)
    want_fp = fingerprint(args.cfg_path)
    model_file = os.path.join(args.out_dir, "model.pt")

    print("[prior_cache] config      {}".format(args.cfg_path))
    print("[prior_cache] fingerprint {}".format(want_fp))
    print("[prior_cache] artifact    {}".format(ref))

    have_key = "WANDB_API_KEY" in os.environ
    if have_key and not args.force_retrain:
        import wandb
        got = try_download(wandb.Api(), ref, args.out_dir, want_fp, args.accept_stale)
        if got is None:
            return 2
        if got:
            return 0

    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    watchdog = None
    if have_key and not args.no_upload:
        watchdog = start_watchdog(model_file, args.project,
                                  "{}_wip".format(name), args.watchdog_interval)
        print("[prior_cache] watchdog pid {}".format(watchdog.pid))

    print("[prior_cache] training...")
    started = time.time()
    try:
        result = subprocess.run([
            sys.executable, os.path.join(repo_root, "tools", "diffusion_model", "train_tinymdm.py"),
            "--cfg_path", args.cfg_path,
            "--out_dir", args.out_dir,
            "--device", args.device,
        ], cwd=repo_root)
    finally:
        if watchdog is not None:
            watchdog.terminate()

    if result.returncode != 0:
        print("[prior_cache] training failed with exit code {}".format(result.returncode))
        return result.returncode

    if not os.path.isfile(model_file):
        print("[prior_cache] training reported success but wrote no {}".format(model_file))
        return 1

    print("[prior_cache] trained in {:.1f} min -> {} ({:.1f} MB)".format(
        (time.time() - started) / 60.0, model_file,
        os.path.getsize(model_file) / 1e6))

    if args.no_upload or not have_key:
        return 0

    import wandb
    run = wandb.init(project=args.project, name="{}_build".format(name),
                     job_type="prior")
    artifact = wandb.Artifact(name, type="model",
                              metadata={"fingerprint": want_fp,
                                        "cfg_path": args.cfg_path})
    artifact.add_file(model_file)
    artifact.add_file(args.cfg_path)

    # train_tinymdm.py writes samples/ -- motions the prior generates itself,
    # plus gifs of them. They are the only way to tell a bad prior from a bad
    # policy, and they cost 50 KB. Keeping them out of the artifact once meant
    # having to dig them back out of a Kaggle kernel's output to answer
    # "did the prior learn the motion, or did the policy give up on it?".
    samples_dir = os.path.join(args.out_dir, "samples")
    if os.path.isdir(samples_dir):
        artifact.add_dir(samples_dir, name="samples")
        print("[prior_cache] attached {} sample file(s)".format(
            len(os.listdir(samples_dir))))

    run.log_artifact(artifact)
    run.finish()
    print("[prior_cache] published {}".format(ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
