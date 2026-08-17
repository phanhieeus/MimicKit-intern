#!/usr/bin/env python3
"""Score a policy rollout against the clip it was supposed to imitate.

Ep_Len_Frac and Sds_Loss_Mean cannot tell success from mode collapse, and on the
M3.1 spinkick run they both said success while the robot was doing nothing of the
kind. At 320M samples it finished with Ep_Len_Frac 0.827 and Sds_Loss 0.1696 --
a *lower* imitation error than the humanoid's converged 0.186 -- and the video
showed it balancing on one leg, jittering in place, never completing a kick.

Both metrics were honest about what they measure, and both were useless here:

* Ep_Len_Frac measures not falling. Standing still does not fall.
* Sds_Loss scores 10-step (0.333 s) windows independently, with nothing tying
  them into a whole, so a motion whose every window is locally plausible can
  still never assemble into the clip.

Why the M3.1 policy scored so well is not settled. The obvious explanation --
that the clip's calm stretches gave the prior a low-motion mode to hide in --
is wrong: sampling the trained prior shows it generates windows as dynamic as
the clip's (mean joint speed 4.24 against 4.73 rad/s) and covers the whole
motion including the kick apex. The clip has no near-static window at all; its
slowest is 3.06 rad/s. So standing still ought to score badly, and something
else is going on. Settling it needs the rollout measured in joint space rather
than guessed at from a video, which is the other reason this script exists.

Either way the failure is invisible from the training curves by construction,
and something has to look at the rollout itself.

    python tools/motion_quality.py \
        --policy    output/smp_m3_spinkick/playback/policy.pkl \
        --reference data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick.pkl

Both files are plain MimicKit clips -- play_policy_to_mp4.py writes the rollout
in exactly that format (play_policy_to_mp4.py:164), so no extra plumbing is
needed. Run it right after make_videos.py and read the verdict.

The decisive number is COVERAGE. Mode collapse has a signature no scalar reward
captures but two do: the poses the policy produces are individually fine
(fidelity is good) while most of the reference motion is never visited at all
(coverage is terrible). A policy that holds one pose scores perfect fidelity.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np


def load_clip(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    frames = np.asarray(data["frames"], dtype=np.float64)
    return frames, float(data["fps"])


def pose_metrics(pol_dof, ref_dof):
    """Two-sided nearest-neighbour distance between the two pose sets.

    Distances are in radians per joint (RMS across joints), then divided by the
    reference clip's own spread so the numbers mean the same thing on a robot
    with a different number of joints or a different range of motion.
    """
    # (n_pol, n_ref) pairwise RMS-per-joint distance.
    diff = pol_dof[:, None, :] - ref_dof[None, :, :]
    dist = np.sqrt((diff ** 2).mean(axis=2))

    fidelity = dist.min(axis=1).mean()      # policy pose -> nearest reference pose
    coverage = dist.min(axis=0).mean()      # reference pose -> nearest policy pose

    # The reference's own spread: how far apart two random frames of it are.
    ref_diff = ref_dof[:, None, :] - ref_dof[None, :, :]
    spread = np.sqrt((ref_diff ** 2).mean(axis=2)).mean()

    return fidelity / spread, coverage / spread, spread


def speed_signal(dof, fps):
    """Per-frame joint speed, in radians/second, RMS across joints."""
    return np.sqrt(((np.diff(dof, axis=0) * fps) ** 2).mean(axis=1))


def periodicity(signal, period_frames):
    """How periodic the motion is, and at what tempo relative to the clip.

    Returned as (strength, tempo) where strength is the highest autocorrelation
    over a generous band of lags and tempo is that lag divided by the clip's own
    period. Deliberately not a lookup at one exact lag: the clip's length in
    rollout frames is rarely a whole number (78 frames at 60 fps is 38.5 at 30),
    and autocorrelation at a short period swings hard over a single frame of
    offset -- a faithful policy scored -0.01 at lag 38 while peaking at 0.86 one
    frame later.

    The two numbers separate the failure modes. Strength near zero means nothing
    repeats at all. Strength high with tempo far from 1.0 means the motion is
    perfectly rhythmic at the wrong rate, which is what the M3.1 rollout does:
    roughly 0.4 s against a 1.28 s clip, just above the reward's 0.333 s window
    and so invisible to it.
    """
    x = signal - signal.mean()
    if x.std() < 1e-9 or len(x) < 4:
        return 0.0, 0.0, 0

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]

    lo = max(2, int(0.2 * period_frames))
    hi = min(len(ac), int(2.6 * period_frames) + 1)
    if hi <= lo:
        return 0.0, 0.0, 0

    best_lag = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[best_lag]), best_lag / period_frames, best_lag


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="policy.pkl from play_policy_to_mp4.py")
    parser.add_argument("--reference", required=True, help="The clip it was trained on.")
    parser.add_argument("--skip", type=int, default=15,
                        help="Leading frames to drop; the rollout starts in a reference "
                             "state and the first moments are that pose settling, not policy "
                             "behaviour (default 15).")
    parser.add_argument("--min_coverage", type=float, default=0.55,
                        help="Coverage below this is called collapse (default 0.55).")
    parser.add_argument("--min_periodicity", type=float, default=0.45,
                        help="Peak autocorrelation below this means nothing repeats "
                             "(default 0.45).")
    parser.add_argument("--tempo_tol", type=float, default=0.3,
                        help="Allowed fractional deviation of the cycle period from the "
                             "clip's; 0.3 accepts 0.7x to 1.3x (default 0.3).")
    parser.add_argument("--quiet", action="store_true", help="Verdict lines only.")
    parser.add_argument("--wandb_project", default=None,
                        help="Log the scores to this WandB project. Without --wandb_run_id "
                             "a new run is created, named after --wandb_run_name.")
    parser.add_argument("--wandb_run_id", default=None,
                        help="Attach to an existing run instead -- pass the training run's "
                             "id so the scores sit beside its curves.")
    parser.add_argument("--wandb_run_name", default="motion_quality")
    parser.add_argument("--json_out", default=None,
                        help="Also write the scores and verdict here as JSON. A curriculum "
                             "runs several stages in one session and the decision is made by "
                             "comparing them; scraping that out of the log is worse than "
                             "writing one file per stage.")
    args = parser.parse_args()

    pol, pol_fps = load_clip(args.policy)
    ref, ref_fps = load_clip(args.reference)

    if pol.shape[1] != ref.shape[1]:
        print("[ERROR] policy frames are {} wide, reference {} -- different robots?".format(
            pol.shape[1], ref.shape[1]), file=sys.stderr)
        return 2

    pol = pol[args.skip:]
    if len(pol) < 30:
        print("[ERROR] only {} usable policy frames".format(len(pol)), file=sys.stderr)
        return 2

    pol_dof, ref_dof = pol[:, 6:], ref[:, 6:]

    fidelity, coverage, spread = pose_metrics(pol_dof, ref_dof)
    pol_speed = speed_signal(pol_dof, pol_fps)
    ref_speed = speed_signal(ref_dof, ref_fps)

    amplitude = pol_speed.mean() / ref_speed.mean()
    burst_pol = pol_speed.std() / max(pol_speed.mean(), 1e-9)
    burst_ref = ref_speed.std() / max(ref_speed.mean(), 1e-9)

    clip_seconds = (len(ref) - 1) / ref_fps
    period_frames = max(2, int(round(clip_seconds * pol_fps)))
    peak_r, tempo, peak_lag = periodicity(pol_speed, period_frames)

    if not args.quiet:
        print("policy    {:4d} frames @ {:g} fps = {:.2f} s".format(
            len(pol) + args.skip, pol_fps, (len(pol) + args.skip - 1) / pol_fps))
        print("reference {:4d} frames @ {:g} fps = {:.2f} s  (spread {:.3f} rad)".format(
            len(ref), ref_fps, clip_seconds, spread))
        print()
        print("  pose fidelity      {:5.2f}   (policy poses -> nearest reference pose; lower is better)".format(fidelity))
        print("  pose COVERAGE      {:5.2f}   (reference poses -> nearest policy pose; lower is better)".format(coverage))
        print("  speed amplitude    {:5.2f}   (policy mean joint speed / reference; 1.0 matches)".format(amplitude))
        print("  burstiness         {:5.2f}   (reference {:.2f}; a kick is bursty, a jitter is flat)".format(burst_pol, burst_ref))
        print("  periodicity        {:5.2f}   strength of the strongest cycle".format(peak_r))
        print("  tempo              {:5.2f}x  cycle is {:.2f} s, clip is {:.2f} s".format(
            tempo, peak_lag / pol_fps, clip_seconds))
        print()

    # Coverage is normalised by the reference's own spread, so 1.0 means the
    # policy is on average as far from the reference motion as two random frames
    # of that motion are from each other -- i.e. it is not tracking it at all.
    problems = []
    if coverage > args.min_coverage:
        problems.append(
            "MODE COLLAPSE: coverage {:.2f} > {:.2f}. Most of the reference motion is "
            "never visited. Fidelity {:.2f} says the poses it does make are fine -- it "
            "found a small safe corner of the prior and stayed there.".format(
                coverage, args.min_coverage, fidelity))
    if peak_r < args.min_periodicity:
        problems.append(
            "NO CYCLE: strongest periodicity is only {:.2f} (want > {:.2f}). Nothing "
            "repeats -- the motion is drift or noise, not a reproduced clip.".format(
                peak_r, args.min_periodicity))
    elif abs(tempo - 1.0) > args.tempo_tol:
        why = ("A cycle much faster than the clip is the collapse signature: the SMP "
               "reward scores 0.333 s windows and cannot see structure longer than one, "
               "so a short cycle is free."
               if tempo < 1.0 else
               "Cycling slower than the clip usually means the motion is there but the "
               "robot cannot drive it at speed -- run retime_motion.py --report.")
        problems.append(
            "WRONG TEMPO: the motion is strongly periodic ({:.2f}) but cycles every "
            "{:.2f} s against the clip's {:.2f} s -- {:.2f}x. {}".format(
                peak_r, peak_lag / pol_fps, clip_seconds, tempo, why))
    if amplitude < 0.5:
        problems.append(
            "TOO SLOW: joint speed is {:.0f}% of the reference. Either the clip is beyond "
            "the actuators (run retime_motion.py --report) or the policy gave up on "
            "matching it.".format(100 * amplitude))

    scores = {
        "Quality/Coverage": coverage,
        "Quality/Fidelity": fidelity,
        "Quality/Speed_Amplitude": amplitude,
        "Quality/Burstiness": burst_pol,
        "Quality/Burstiness_Ref": burst_ref,
        "Quality/Periodicity": peak_r,
        "Quality/Tempo": tempo,
        "Quality/Passed": 0.0 if problems else 1.0,
    }

    if args.wandb_project:
        log_to_wandb(scores, problems, args)

    if args.json_out:
        payload = dict(scores)
        payload["problems"] = problems
        payload["policy"] = args.policy
        payload["reference"] = args.reference
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print("wrote {}".format(args.json_out))

    if problems:
        print("VERDICT: FAIL")
        for problem in problems:
            print("  - " + problem)
        return 1

    print("VERDICT: PASS -- the policy reproduces the motion, not just its safe parts.")
    return 0


def log_to_wandb(scores, problems, args):
    """Put the scores where the training curves are, so the two are read together.

    They belong next to Ep_Len_Frac in particular: that metric is what made this
    script necessary, and seeing Quality/Coverage beside it is the fastest way to
    notice a run whose episode length looks healthy because the policy stopped
    trying.
    """
    import wandb

    if args.wandb_run_id:
        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow")
    else:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                         job_type="eval")

    run.summary.update(scores)
    run.summary["Quality/Verdict"] = "FAIL" if problems else "PASS"
    if problems:
        # Kept as text so the reason survives in the run page, not just the log.
        run.summary["Quality/Problems"] = " | ".join(p.split(":")[0] for p in problems)
    run.finish()
    print("[wandb] scores logged to {}\n".format(args.wandb_project))


if __name__ == "__main__":
    sys.exit(main())
