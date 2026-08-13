#!/usr/bin/env python3
"""Slow a retargeted clip down until the robot can actually produce the torques it asks for.

Retargeting matches *pose*, never *dynamics*. It maps a human's joint angles onto the
robot and stops there -- nothing in that pipeline knows the robot's actuator limits, so
a clip that looks right frame by frame can demand torques the hardware cannot deliver.
Train against such a clip and the policy tops out early: it reproduces the shape of the
motion but cannot keep the timing, so it falls.

vr_m3_1_humanoid_spinkick.pkl is one of those. Measured against the M3.1's own MJCF
(composite rigid-body inertia about each joint axis, tau = I_eff * qdd, versus
actuatorfrcrange):

    right_hip_roll     1265 N.m / 360  = 3.5x over
    waist_yaw           389 N.m / 102  = 3.8x over
    left_shoulder_pitch 782 N.m /  66  = 11.8x over

Torque scales as 1/s^2 under a retime of factor s, so s = sqrt(overshoot) is what it
takes to bring a joint inside its limit. Full compliance including the arms needs 3.4x,
which is so slow it stops looking like a kick. The balance-critical set -- hips, knees,
waist -- needs only about 2.0x, and arms that saturate do not make the robot fall. That
is the trade this script is for: pick s from the joints that decide whether it stays up.

    python tools/retime_motion.py \
        --input data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick.pkl \
        --output data/motions/vr_m3_1/vr_m3_1_humanoid_spinkick_slow2.pkl \
        --char_file data/assets/vr_m3_1/vr_m3_1.xml \
        --factor 2.0

Pass --report on its own to print the feasibility table without writing anything, and
--auto to let the script pick s from --exclude'd joints (arms are excluded by default).

Resampling keeps the original fps and adds frames, rather than just relabelling fps
downward: MotionLib finite-differences consecutive frames for velocity, so preserving
temporal resolution keeps those velocities meaningful.

NOTE: the diffusion prior is trained on the clip, so a retimed clip needs a retrained
prior. Point tinymdm_*.yaml at the new file and rerun the prior stage before the RL
stage -- reusing the old prior would score the new motion against the old timing.
"""

import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np


# ---------------------------------------------------------------- MJCF inertia

def _quat_to_mat(q):
    """MuJoCo stores quaternions wxyz."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _attr(elem, name, default):
    raw = elem.get(name)
    return np.array([float(v) for v in raw.split()]) if raw else np.array(default, dtype=float)


def load_joint_limits(char_file):
    """Effective inertia, gains and torque limit for every actuated joint, in MJCF order.

    I_eff is the composite rigid-body inertia of the joint's whole distal subtree about
    the joint axis, taken in the model's rest pose plus the joint's own armature. It is
    an upper bound while the robot is airborne -- a floating base recoils, so the
    articulated-body inertia a joint really sees is lower -- and about right when the
    foot is planted, which is when balance is decided and when this matters.
    """
    root = ET.parse(char_file).getroot().find("worldbody")
    bodies, joints = [], []

    def walk(elem, r_parent, p_parent, parent_idx):
        for body in elem.findall("body"):
            rot = r_parent @ _quat_to_mat(_attr(body, "quat", [1, 0, 0, 0]))
            pos = p_parent + r_parent @ _attr(body, "pos", [0, 0, 0])

            inertial = body.find("inertial")
            if inertial is not None:
                mass = float(inertial.get("mass"))
                com = pos + rot @ _attr(inertial, "pos", [0, 0, 0])
                r_i = rot @ _quat_to_mat(_attr(inertial, "quat", [1, 0, 0, 0]))
                inertia = r_i @ np.diag(_attr(inertial, "diaginertia", [0, 0, 0])) @ r_i.T
            else:
                mass, com, inertia = 0.0, pos.copy(), np.zeros((3, 3))

            idx = len(bodies)
            bodies.append(dict(mass=mass, com=com, inertia=inertia, parent=parent_idx))

            for joint in body.findall("joint"):
                if joint.get("type") == "free":
                    continue
                joints.append(dict(
                    name=joint.get("name"),
                    axis=rot @ _attr(joint, "axis", [0, 0, 1]),
                    anchor=pos + rot @ _attr(joint, "pos", [0, 0, 0]),
                    body=idx, attrib=joint.attrib))

            walk(body, rot, pos, idx)

    walk(root, np.eye(3), np.zeros(3), -1)

    children = {}
    for i, body in enumerate(bodies):
        children.setdefault(body["parent"], []).append(i)

    def subtree(i):
        out, stack = [], [i]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(children.get(k, []))
        return out

    limits = []
    for joint in joints:
        axis = joint["axis"] / np.linalg.norm(joint["axis"])
        i_eff = 0.0
        for k in subtree(joint["body"]):
            body = bodies[k]
            offset = body["com"] - joint["anchor"]
            perp = offset - np.dot(offset, axis) * axis
            i_eff += axis @ body["inertia"] @ axis + body["mass"] * np.dot(perp, perp)

        attrib = joint["attrib"]
        i_eff += float(attrib.get("armature", 0.0))
        frc = attrib.get("actuatorfrcrange")
        limits.append(dict(
            name=joint["name"],
            inertia=i_eff,
            tau_max=abs(float(frc.split()[1])) if frc else np.inf))

    return limits


# ---------------------------------------------------------------- feasibility

def torque_table(frames, fps, limits):
    """Peak |tau| each joint must produce to follow the clip, against what it has."""
    dof = frames[:, 6:]
    if dof.shape[1] != len(limits):
        raise ValueError("clip has {} dofs but {} has {} actuated joints".format(
            dof.shape[1], "the char file", len(limits)))

    vel = np.gradient(dof, 1.0 / fps, axis=0)
    acc = np.gradient(vel, 1.0 / fps, axis=0)

    rows = []
    for i, limit in enumerate(limits):
        tau = limit["inertia"] * np.abs(acc[:, i]).max()
        rows.append(dict(limit, tau=tau, ratio=tau / limit["tau_max"],
                         vel_peak=np.abs(vel[:, i]).max()))
    return rows


def print_report(rows, factor=1.0):
    scaled = [dict(r, tau=r["tau"] / factor ** 2, ratio=r["ratio"] / factor ** 2,
                   vel_peak=r["vel_peak"] / factor) for r in rows]
    header = "clip as-is" if factor == 1.0 else "clip retimed {:.2f}x".format(factor)
    print("=== {} ===".format(header))
    print("%-28s %8s %9s %9s %7s" % ("joint", "I_eff", "|qd|peak", "tau_req", "vs max"))
    for row in sorted(scaled, key=lambda r: -r["ratio"]):
        flag = "  OVER" if row["ratio"] > 1.0 else ""
        print("%-28s %8.3f %9.2f %9.1f %6.2fx%s" % (
            row["name"], row["inertia"], row["vel_peak"], row["tau"],
            row["ratio"], flag))
    over = [r for r in scaled if r["ratio"] > 1.0]
    print("\n%d of %d joints over limit" % (len(over), len(scaled)))
    return scaled


def required_factor(rows, exclude):
    """Smallest retime that fits every joint whose name avoids the excluded substrings."""
    considered = [r for r in rows if not any(x in r["name"] for x in exclude)]
    if not considered:
        raise ValueError("--exclude removed every joint")
    worst = max(considered, key=lambda r: r["ratio"])
    return max(np.sqrt(worst["ratio"]), 1.0), worst


# ---------------------------------------------------------------- resampling

def _exp_map_to_quat(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-9, v / np.maximum(angle, 1e-9), np.array([1.0, 0.0, 0.0]))
    half = 0.5 * angle
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def _quat_to_exp_map(q):
    q = np.where(q[..., :1] < 0.0, -q, q)          # shortest arc
    w = np.clip(q[..., :1], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = np.where(s > 1e-9, q[..., 1:] / np.maximum(s, 1e-9), np.array([1.0, 0.0, 0.0]))
    return axis * angle


def _slerp(q0, q1, t):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Below this the two rotations are close enough that lerp and slerp agree to
    # float precision, and dividing by sin_theta would be dividing by nothing.
    near = sin_theta < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(near, 1.0, sin_theta))
    w1 = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin_theta))

    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def retime(frames, factor):
    """Stretch the clip by `factor`, keeping the original frame rate.

    Root translation and joint angles interpolate linearly; the root orientation goes
    through quaternions and slerp, because exp-map coordinates do not interpolate
    linearly once the rotation gets large -- and a spinkick's root spins most of a turn.
    """
    n_src = len(frames)
    n_dst = int(round((n_src - 1) * factor)) + 1
    src = np.linspace(0.0, n_src - 1, n_dst)

    idx0 = np.floor(src).astype(int)
    idx1 = np.minimum(idx0 + 1, n_src - 1)
    blend = (src - idx0)[:, None]

    root_pos = frames[idx0, 0:3] * (1 - blend) + frames[idx1, 0:3] * blend
    quat = _slerp(_exp_map_to_quat(frames[idx0, 3:6]),
                  _exp_map_to_quat(frames[idx1, 3:6]), blend)
    root_rot = _quat_to_exp_map(quat)
    dof = frames[idx0, 6:] * (1 - blend) + frames[idx1, 6:] * blend

    return np.concatenate([root_pos, root_rot, dof], axis=1)


# ---------------------------------------------------------------- entry point

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Where to write the retimed clip.")
    parser.add_argument("--char_file", required=True, help="MJCF the clip is retargeted to.")
    parser.add_argument("--factor", type=float, help="Retime factor; 2.0 runs at half speed.")
    parser.add_argument("--auto", action="store_true",
                        help="Choose --factor from the joints left after --exclude.")
    parser.add_argument("--exclude", nargs="*",
                        default=["shoulder", "elbow", "wrist"],
                        help="Name substrings ignored by --auto. Arms saturating does not "
                             "topple the robot, and demanding they comply makes the clip "
                             "far slower than balance needs.")
    parser.add_argument("--report", action="store_true", help="Print the table and exit.")
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        motion = pickle.load(handle)
    frames = np.array(motion["frames"], dtype=np.float64)
    fps = float(motion["fps"])

    limits = load_joint_limits(args.char_file)
    rows = torque_table(frames, fps, limits)

    print("%s: %d frames @ %g fps = %.3f s\n" % (
        os.path.basename(args.input), len(frames), fps, (len(frames) - 1) / fps))
    print_report(rows, 1.0)

    if args.auto:
        factor, worst = required_factor(rows, args.exclude)
        print("\n--auto: %.2fx, set by %s at %.2fx over" % (
            factor, worst["name"], worst["ratio"]))
    else:
        factor = args.factor

    if args.report or factor is None:
        if factor is None and not args.report:
            print("\nNothing written: pass --factor or --auto.", file=sys.stderr)
            return 1
        return 0

    print()
    print_report(rows, factor)

    out_frames = retime(frames, factor)
    if not args.output:
        print("\nNothing written: pass --output.", file=sys.stderr)
        return 1

    motion_out = dict(motion)
    motion_out["frames"] = out_frames.tolist()
    motion_out["fps"] = fps
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(motion_out, handle)

    print("\nwrote %s: %d frames @ %g fps = %.3f s" % (
        args.output, len(out_frames), fps, (len(out_frames) - 1) / fps))
    print("Retrain the diffusion prior against this file before the RL stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
