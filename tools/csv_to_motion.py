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


def foot_heights(char_file, frames, soles=False):
    """Lowest foot height per frame.

    With soles=False this is the ankle body origin, which is all the orientation
    search needs. With soles=True it is the lowest point of the foot's collision
    geometry, which is the only thing that can be aligned to the floor: the sole
    hangs about 6 cm below the ankle origin, so aligning origins buries the foot.
    """
    import torch
    import util.torch_util as torch_util

    char, feet = _char(char_file)
    f = torch.tensor(frames, dtype=torch.float32)
    body_pos, body_rot = char.forward_kinematics(
        f[:, 0:3], torch_util.exp_map_to_quat(f[:, 3:6]), char.dof_to_rot(f[:, 6:]))
    if not soles:
        return body_pos[:, feet, 2].min(dim=1).values.numpy()

    names = char.get_body_names()
    tree = {b.get("name"): b for b in ET.parse(char_file).iter("body")}
    lows = np.full(len(frames), np.inf)
    for i in feet:
        body = tree.get(names[i])
        if body is None:
            continue
        P = body_pos[:, i, :].numpy()
        Q = body_rot[:, i, :].numpy()                      # xyzw
        for g in body.findall("geom"):
            if "collision" not in (g.get("class") or ""):
                continue
            gp = _attr(g, "pos", [0, 0, 0]); gq = _attr(g, "quat", [1, 0, 0, 0])
            size = _attr(g, "size", [0]); kind = g.get("type", "sphere")
            for k in range(len(P)):
                rot = _quat_mat(np.array([Q[k, 3], Q[k, 0], Q[k, 1], Q[k, 2]]))
                rg = rot @ _quat_mat(gq)
                c = P[k] + rot @ gp
                if kind == "capsule":
                    axis = rg @ np.array([0, 0, 1.0])
                    z = min((c + size[1] * axis)[2], (c - size[1] * axis)[2]) - size[0]
                elif kind == "box":
                    z = c[2] - sum(abs(rg[2, j]) * size[j] for j in range(3))
                else:
                    z = c[2] - size[0]
                lows[k] = min(lows[k], z)
    return lows


def _attr(elem, name, default):
    raw = elem.get(name)
    return np.array([float(v) for v in raw.split()]) if raw else np.array(default, float)


def _quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


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


def _feet_xyz(char_file, frames):
    """World position of every foot body, per frame: (n_frames, n_feet, 3)."""
    import torch
    import util.torch_util as torch_util
    char, feet = _char(char_file)
    f = torch.tensor(frames, dtype=torch.float32)
    body_pos, _ = char.forward_kinematics(
        f[:, 0:3], torch_util.exp_map_to_quat(f[:, 3:6]), char.dof_to_rot(f[:, 6:]))
    return body_pos[:, feet, :].numpy()


def _axis_rot(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _log(m):
    cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return v * (angle / (2.0 * np.sin(angle)))


def solve_chain(axes, target, guess):
    """Angles about `axes`, composed in order, that reproduce `target`.

    Gauss-Newton on the rotation error, seeded from the pose we already have. The
    hip is a genuine 3-dof chain so an exact solution exists; the tilted pitch
    axis is why this cannot be read off as Euler angles.
    """
    th = np.array(guess, float)

    def compose(a):
        m = np.eye(3)
        for axis, angle in zip(axes, a):
            m = m @ _axis_rot(axis, angle)
        return m

    for _ in range(24):
        err = _log(target.T @ compose(th))
        if np.linalg.norm(err) < 1e-9:
            break
        jac = np.zeros((3, 3))
        for i in range(3):
            d = th.copy(); d[i] += 1e-6
            jac[:, i] = (_log(target.T @ compose(d)) - err) / 1e-6
        try:
            th -= np.linalg.solve(jac, err)
        except np.linalg.LinAlgError:
            break
    return th


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
    parser.add_argument("--fold_waist", action="store_true",
                        help="Reproduce waist roll/pitch the robot does not have by tilting "
                             "the pelvis and counter-rotating both hips. Exact in "
                             "orientation, and it is how a stiff-waisted robot would do the "
                             "move for real -- but it hands the work to the hips.")
    parser.add_argument("--fold_scale", type=float, default=1.0,
                        help="Fraction of the waist motion to fold in (try 0.5 first).")
    parser.add_argument("--limit_fix", default="report",
                        choices=["report", "shift", "clamp"],
                        help="What to do with joints outside the MJCF range. 'shift' adds "
                             "a constant offset so the trajectory slides inside, keeping "
                             "the shape of the motion and changing only where the limb "
                             "sits; 'clamp' flattens the overshooting frames instead. "
                             "Shift is usually right when the range of motion fits and "
                             "only its centre is off, which is what a differing rest pose "
                             "looks like.")
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

    def apply_fold(mats, dof):
        """Give the pelvis the lean the waist cannot, and take it back out of the legs.

        C is the waist rotation beyond yaw, expressed in the pelvis frame. Setting
        root' = root @ C and H' = C.T @ H leaves the torso and both legs in exactly
        the orientation the source had -- the algebra cancels. What it does not
        leave alone is the load: holding the trunk out over the hips is now the
        hips' job, which is also how a stiff-waisted robot would really do it.
        """
        dof_before = dof.copy()
        wy = data[:, col["waist_yaw_joint_dof"]] * ang_scale
        wr = data[:, col["waist_roll_joint_dof"]] * ang_scale * args.fold_scale
        wp = data[:, col["waist_pitch_joint_dof"]] * ang_scale * args.fold_scale
        base = np.percentile(wp, 5)
        print("\nfolding waist into pelvis + hips (scale %.2f); %.1f deg of pitch treated "
              "as rest pose and left out" % (args.fold_scale, np.degrees(base)))
        wp = wp - base

        hip_axes = [(0, 0.965926, -0.258819), (1, 0, 0), (0, 0, 1)]
        legs = [[joints.index("%s_hip_%s_joint" % (s, k)) for k in ("pitch", "roll", "yaw")]
                for s in ("left", "right")]
        out, cmats, mats_in = [], [], mats
        for k in range(len(mats)):
            ryaw = _axis_rot((0, 0, 1), wy[k])
            cmat = ryaw @ _axis_rot((1, 0, 0), wr[k]) @ _axis_rot((0, 1, 0), wp[k]) @ ryaw.T
            cmats.append(cmat)
            out.append(mats[k] @ cmat)
            for idx in legs:
                cur = [dof[k, i] for i in idx]
                h = np.eye(3)
                for axis, angle in zip(hip_axes, cur):
                    h = h @ _axis_rot(axis, angle)
                for i, v in zip(idx, solve_chain(hip_axes, cmat.T @ h, cur)):
                    dof[k, i] = v
        # The hip is not a ball joint: its three axes sit 0, 6 and 29 cm apart down
        # the leg. Counter-rotating them reproduces the leg's orientation exactly and
        # its *position* not at all, so the feet drift by tens of centimetres. Fixing
        # that properly means full leg IK. Short of it, translate the root by the mean
        # displacement of both feet -- continuous, unlike anchoring to whichever foot
        # is lowest, which jumped 44 cm every time the support swapped.
        pre = _feet_xyz(args.char_file, np.concatenate(
            [root_pos, mat_to_expmap(mats_in), dof_before], axis=1))
        post = _feet_xyz(args.char_file, np.concatenate(
            [root_pos, mat_to_expmap(out), dof], axis=1))
        shift = (pre - post).mean(axis=1)
        print("   feet drifted %.1f cm on average; root translated to absorb it "
              "(residual %.1f cm)" % (np.linalg.norm(pre - post, axis=2).mean() * 100,
                                      np.abs(shift).max() * 100))
        return out, dof, root_pos + shift

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
        mats = mats_for(order, intrinsic)
        if args.fold_waist:
            mats, dof, root_pos = apply_fold(mats, dof)
        frames = np.concatenate([root_pos, mat_to_expmap(mats), dof], axis=1)
        print("   -> %s" % label)
        if near < 0.3:
            print("   WARNING: even the best convention keeps a foot near the floor in only "
                  "%.0f%% of frames. Treat the result with suspicion." % (100 * near))
    else:
        order, _, kind = args.euler_order.partition("-")
        mats = mats_for(order.upper(), kind != "extrinsic")
        if args.fold_waist:
            mats, dof, root_pos = apply_fold(mats, dof)
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
    elif args.limit_fix == "shift" or args.clamp or args.limit_fix == "clamp":
        mode = "clamp" if (args.clamp or args.limit_fix == "clamp") else "shift"
        for i, j in enumerate(joints):
            lo, hi = lim[j]
            lo_d, hi_d = dof[:, i].min(), dof[:, i].max()
            if lo_d >= lo and hi_d <= hi:
                continue
            if mode == "shift":
                span, room = hi_d - lo_d, hi - lo
                if span <= room:
                    # The motion fits; only its centre is out. One nudge is enough,
                    # and the shape of the movement survives untouched.
                    off = (lo - lo_d) if lo_d < lo else (hi - hi_d)
                    dof[:, i] += off
                    print("   %-28s shifted %+5.1f deg" % (j, np.degrees(off)))
                    continue
                # Wider than the joint can go: centre it, then clip what is left.
                off = 0.5 * (lo + hi) - 0.5 * (lo_d + hi_d)
                dof[:, i] += off
                before = dof[:, i].copy()
                dof[:, i] = np.clip(dof[:, i], lo, hi)
                print("   %-28s shifted %+5.1f deg, still %.1f deg too wide -> clipped "
                      "%d frame(s)" % (j, np.degrees(off), np.degrees(span - room),
                                       int((before != dof[:, i]).sum())))
            else:
                dof[:, i] = np.clip(dof[:, i], lo, hi)
                print("   %-28s clamped" % j)
        frames[:, 6:] = dof

    soles = foot_heights(args.char_file, frames, soles=True)
    if args.ground_offset != "none":
        offset = -float(np.min(soles)) if args.ground_offset == "auto" else float(args.ground_offset)
        frames[:, 2] += offset
        soles = soles + offset
        print("\nground: shifted root z by %+.1f cm so the lowest point of the foot "
              "collision geometry sits at 0" % (offset * 100))
    print("sole height: min %+.1f cm, median %+.1f cm, frames below floor: %d" % (
        soles.min() * 100, np.median(soles) * 100, int((soles < -1e-4).sum())))

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
