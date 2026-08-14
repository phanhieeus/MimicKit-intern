#!/usr/bin/env python3
"""Turn a retargeted-animation CSV into a MimicKit motion clip.

Animation tools export a frame per row and a channel per column -- root
translation, root Euler angles, one column per joint -- and none of that matches
what MimicKit reads, which is a pickle of
`[root_pos(3), root_rot_expmap(3), dof(N)]` rows in the target MJCF's own joint
order, in metres and radians.

Three of the four gaps are silent if you get them wrong, so this script measures
rather than assumes:

* **Joint order.** Columns are matched to the MJCF **by name**, never by
  position. The two orders genuinely differ: a G1-style export lists the wrist as
  roll, pitch, yaw while the M3.1 MJCF declares it yaw, roll, pitch, so copying
  column-by-column silently swaps three wrist axes per arm.
* **Units.** Detected from the data. A root height near 87 is centimetres; a knee
  angle near 76 is degrees.
* **Euler order.** Not recoverable from the file, so all twelve conventions (six
  orders, intrinsic and extrinsic) are tried, forward kinematics is run for each,
  and the one that puts the feet on the floor wins. A wrong order tips the whole
  character over, which is obvious in the score and invisible in the numbers.

Extra columns are dropped with a warning, which is the case worth reading: the
M3.1 has one waist joint and a G1-style export has three, so `waist_roll` and
`waist_pitch` have nowhere to go. That is not cosmetic -- the shoulders ride on
the torso, so discarding a 20-79 degree forward lean moves the hands as well.

    python tools/csv_to_motion.py \\
        --csv "csv-data/dancewhat_retarget_vrm3_loop(in).csv" \\
        --char_file data/assets/vr_m3_1/vr_m3_1.xml \\
        --output data/motions/vr_m3_1/vr_m3_1_dance_what.pkl \\
        --fps 30

Ends by reporting joint-limit violations and ground clearance, then hands you the
command to check flight phase and torque with retime_motion.py.
"""

import argparse
import csv as csvlib
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mimickit"))

# Renames that carry no meaning: the same axis under a different house style.
ALIASES = [
    ("_knee_joint", "_knee_pitch_joint"),
    ("_elbow_joint", "_elbow_pitch_joint"),
]

AXIS_ROT = {
    "X": lambda c, s: np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
    "Y": lambda c, s: np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
    "Z": lambda c, s: np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
}
ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


def canonical(name):
    out = name[:-4] if name.endswith("_dof") else name
    for old, new in ALIASES:
        out = out.replace(old, new)
    return out


def euler_to_mat(angles, order, intrinsic):
    """Compose per-axis rotations. Intrinsic applies them about the moving frame."""
    mats = []
    for axis, angle in zip(order, angles):
        mats.append(AXIS_ROT[axis](np.cos(angle), np.sin(angle)))
    a, b, c = mats
    return a @ b @ c if intrinsic else c @ b @ a


def mat_to_expmap(mats):
    """Rotation matrices to axis-angle vectors, which is what MimicKit stores."""
    out = np.zeros((len(mats), 3))
    for i, m in enumerate(mats):
        cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos)
        if angle < 1e-8:
            continue
        if angle > np.pi - 1e-6:
            # Near 180 degrees the skew part vanishes; recover the axis from the
            # diagonal of (R + I) instead, where it is still well conditioned.
            d = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
            k = int(np.argmax(d))
            axis = (m[:, k] + np.eye(3)[:, k]) / (2.0 * max(d[k], 1e-9))
            axis /= np.linalg.norm(axis)
        else:
            axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
            axis /= (2.0 * np.sin(angle))
        out[i] = axis * angle
    return out


_CHAR = {}


def _char(char_file):
    """Build the kinematic model once; the auto-detect pass reuses it twelve times."""
    if char_file not in _CHAR:
        import anim.mjcf_char_model as mjcf_char_model
        model = mjcf_char_model.MJCFCharModel("cpu")
        model.load(char_file)
        names = model.get_body_names()
        feet = [i for i, n in enumerate(names)
                if "ankle" in n or "foot" in n or "toe" in n]
        _CHAR[char_file] = (model, feet)
    return _CHAR[char_file]


def foot_heights(char_file, frames):
    import torch
    import util.torch_util as torch_util

    char, feet = _char(char_file)
    f = torch.tensor(frames, dtype=torch.float32)
    body_pos, _ = char.forward_kinematics(
        f[:, 0:3], torch_util.exp_map_to_quat(f[:, 3:6]), char.dof_to_rot(f[:, 6:]))
    return body_pos[:, feet, 2].min(dim=1).values.numpy()


def score_orientation(char_file, root_pos, root_mats, dof, stride=1):
    """How much like standing on a floor this orientation looks.

    A correct convention keeps a foot near the ground in most frames. A wrong one
    tips the character, and the lowest foot then wanders over tens of centimetres.
    """
    frames = np.concatenate([root_pos, mat_to_expmap(root_mats), dof], axis=1)
    h = foot_heights(char_file, frames[::stride])
    floor = np.percentile(h, 5)
    near = float(np.mean(h < floor + 0.05))
    return near, float(np.std(h)), frames


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--char_file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, required=True,
                        help="Not stored in the CSV, so it has to be told.")
    parser.add_argument("--euler_order", default="auto",
                        help="auto, or e.g. XYZ / ZYX-extrinsic.")
    parser.add_argument("--loop_mode", type=int, default=0, choices=[0, 1],
                        help="0 CLAMP, 1 WRAP. A looping clip wants 1.")
    parser.add_argument("--clamp", action="store_true",
                        help="Clip joints to the MJCF range. Off by default so a bad "
                             "retarget is visible rather than quietly squashed; on when "
                             "the overshoot is real and small, because a reference pose "
                             "outside a joint stop is one the robot can never reach, and "
                             "leaving it in penalises the policy forever for a frame it "
                             "cannot match.")
    parser.add_argument("--ground_offset", default="auto",
                        help="Metres to add to root z so the soles reach the floor, "
                             "or 'auto' to match the lowest foot to 0, or 'none'.")
    args = parser.parse_args()

    with open(args.csv) as handle:
        rows = list(csvlib.reader(handle))
    header, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    col = {h: i for i, h in enumerate(header)}
    print("%s: %d frames x %d columns" % (os.path.basename(args.csv), *data.shape))

    joints = [j.get("name") for j in ET.parse(args.char_file).iter("joint")
              if j.get("type") != "free"]
    dof_cols = [h for h in header if h.endswith("_dof")]
    available = {canonical(h): h for h in dof_cols}

    missing = [j for j in joints if j not in available]
    if missing:
        print("\nERROR: the MJCF needs joints the CSV does not have:", file=sys.stderr)
        for m in missing:
            print("   " + m, file=sys.stderr)
        return 2

    dropped = [h for h in dof_cols if canonical(h) not in joints]
    if dropped:
        print("\nDROPPED -- the CSV has these, the robot does not:")
        for h in dropped:
            v = data[:, col[h]]
            print("   %-28s range %6.1f  (%.1f .. %.1f)" % (
                h, v.max() - v.min(), v.min(), v.max()))
        print("   Their motion is discarded. Check the range before accepting that.")

    # Units, from the data rather than from faith.
    root_xyz = data[:, [col["root_translateX"], col["root_translateY"], col["root_translateZ"]]]
    pos_scale = 0.01 if np.abs(root_xyz).max() > 10.0 else 1.0
    ang_cols = [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]] + \
               [col[available[j]] for j in joints]
    ang_scale = np.pi / 180.0 if np.abs(data[:, ang_cols]).max() > 6.5 else 1.0
    print("\nunits: translation x%g (%s), angles x%.5f (%s)" % (
        pos_scale, "cm -> m" if pos_scale != 1 else "already m",
        ang_scale, "deg -> rad" if ang_scale != 1 else "already rad"))

    root_pos = root_xyz * pos_scale
    root_eul = data[:, [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]]] * ang_scale
    dof = np.stack([data[:, col[available[j]]] for j in joints], axis=1) * ang_scale

    def mats_for(order, intrinsic, sel=None):
        eul = root_eul if sel is None else root_eul[sel]
        idx = ["XYZ".index(a) for a in order]
        return [euler_to_mat(e[idx], order, intrinsic) for e in eul]

    if args.euler_order == "auto":
        print("\ntrying every Euler convention; the one that stands on the floor wins")
        print("   %-22s %10s %10s" % ("convention", "on floor", "foot z std"))
        best = None
        # Score on a quarter of the frames; the conventions differ by tens of
        # centimetres, not by anything a denser sample would reveal.
        sel = slice(None, None, 4)
        for order in ORDERS:
            for intrinsic in (True, False):
                mats = mats_for(order, intrinsic, sel)
                near, std, _ = score_orientation(
                    args.char_file, root_pos[sel], mats, dof[sel])
                label = "%s-%s" % (order, "intrinsic" if intrinsic else "extrinsic")
                print("   %-22s %9.1f%% %10.3f" % (label, 100 * near, std))
                if best is None or near > best[0]:
                    best = (near, std, (order, intrinsic), label)
        near, std, (order, intrinsic), label = best
        frames = np.concatenate(
            [root_pos, mat_to_expmap(mats_for(order, intrinsic)), dof], axis=1)
        print("   -> %s" % label)
        if near < 0.3:
            print("   WARNING: even the best convention keeps a foot near the floor in only "
                  "%.0f%% of frames. Treat the result with suspicion." % (100 * near))
    else:
        order, _, kind = args.euler_order.partition("-")
        mats = mats_for(order.upper(), kind != "extrinsic")
        near, std, frames = score_orientation(args.char_file, root_pos, mats, dof)
        print("\n%s: a foot is near the floor in %.1f%% of frames" % (args.euler_order, 100 * near))

    # Joint limits, reported rather than clamped: a violation means the retarget
    # targeted a different skeleton, and silently clipping would hide that.
    lim = {j.get("name"): [float(x) for x in (j.get("range") or "0 0").split()]
           for j in ET.parse(args.char_file).iter("joint") if j.get("type") != "free"}
    print("\njoint limits:")
    bad = 0
    for i, j in enumerate(joints):
        lo, hi = lim[j]
        over = max(dof[:, i].max() - hi, lo - dof[:, i].min())
        if over > 1e-6:
            bad += 1
            print("   %-28s over by %5.1f deg on %d frame(s)" % (
                j, np.degrees(over), int(((dof[:, i] > hi) | (dof[:, i] < lo)).sum())))
    if not bad:
        print("   every joint stays inside its range")
    elif args.clamp:
        for i, j in enumerate(joints):
            dof[:, i] = np.clip(dof[:, i], lim[j][0], lim[j][1])
        frames[:, 6:] = dof
        print("   --clamp: clipped to range, so every target pose is now reachable")

    heights = foot_heights(args.char_file, frames)
    if args.ground_offset != "none":
        offset = -float(np.min(heights)) if args.ground_offset == "auto" else float(args.ground_offset)
        frames[:, 2] += offset
        heights = heights + offset
        print("\nground: shifted root z by %+.1f cm so the lowest sole sits at 0" % (offset * 100))
    print("lowest foot %+.1f cm, median %+.1f cm" % (heights.min() * 100, np.median(heights) * 100))

    if not args.output:
        print("\nNothing written: pass --output.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump({"loop_mode": args.loop_mode, "fps": float(args.fps),
                     "frames": frames.tolist()}, handle)
    print("\nwrote %s: %d frames @ %g fps = %.2f s, %d wide (6 + %d dof)" % (
        args.output, len(frames), args.fps, (len(frames) - 1) / args.fps,
        frames.shape[1], len(joints)))
    print("\nNext, check it is trainable at all:")
    print("  python tools/retime_motion.py --input %s \\\n      --char_file %s --report"
          % (args.output, args.char_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
