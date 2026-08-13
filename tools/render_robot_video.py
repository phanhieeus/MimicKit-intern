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

"""Render a retargeted robot motion to mp4 with MuJoCo's offscreen renderer.

Takes either end of the last pipeline stage:

* a **GMR** file  -- ``{fps, root_pos, root_rot (xyzw), dof_pos}``
* a **MimicKit** file -- ``{fps, loop_mode, frames}`` with
  ``[root_pos(3), root_rot_expmap(3), dof_pos(N)]`` per frame

and plays it back on the robot model, so you see the actual machine rather than
a stick figure.

    python tools/render_robot_video.py \\
        --motion data/motions/vr_m3_1/vr_m3_1_long_walk0.pkl \\
        --robot-xml data/assets/vr_m3_1/vr_m3_1.xml \\
        --output walk.mp4

Headless rendering needs a GL backend. EGL usually works on a machine with
NVIDIA drivers even with no display -- this script selects it by default. If the
renderer fails, try ``--gl osmesa`` (needs ``libosmesa6``).

Note: MuJoCo's EGL context raises an ``EGLError`` from its destructor at exit on
some driver versions. The frames are already written by then; the message is
noise, not a failed render.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The bare robot MJCF has no floor, lights or sky. Wrap it so the render is
# legible instead of a silhouette on black.
SCENE_TEMPLATE = """<mujoco model="render_scene">
  <include file="{robot_xml}"/>
  <statistic center="0 0 0.7" extent="2.2"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-18"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.24 0.31 0.4"
             rgb2="0.05 0.07 0.1" width="512" height="3072"/>
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.22 0.24 0.26" rgb2="0.28 0.30 0.32" markrgb="0.75 0.75 0.75"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="4 4" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (root_pos, root_quat_wxyz, dof_pos, fps) from either format."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)

    if "dof_pos" in d:  # GMR format
        root_pos = np.asarray(d["root_pos"], dtype=np.float64)
        # GMR files store the root quaternion xyzw; MuJoCo wants wxyz.
        root_quat = np.asarray(d["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
        dof = np.asarray(d["dof_pos"], dtype=np.float64)
        return root_pos, root_quat, dof, float(d["fps"])

    if "frames" in d:  # MimicKit format
        import mujoco

        f = np.asarray(d["frames"], dtype=np.float64)
        root_pos = f[:, 0:3]
        exp = f[:, 3:6]
        quat = np.empty((len(f), 4))
        for i in range(len(f)):
            angle = float(np.linalg.norm(exp[i]))
            axis = exp[i] / angle if angle > 1e-9 else np.array([0.0, 0.0, 1.0])
            mujoco.mju_axisAngle2Quat(quat[i], axis, angle)
        return root_pos, quat, f[:, 6:], float(d["fps"])

    raise ValueError(
        f"{path} is neither a GMR file (needs 'dof_pos') nor a MimicKit clip "
        f"(needs 'frames'). Keys: {sorted(d)}"
    )


def writable_scene_dir(robot_xml: Path) -> tuple[Path, Path | None]:
    """Somewhere to drop the scene wrapper that is beside the robot's assets.

    The wrapper has to sit next to the robot XML, not in a temp dir of its own:
    <include> resolves relative to the including file, and so does the robot's
    own `meshdir` -- vr_m3_1.xml declares `meshdir="assets/"`, which MuJoCo reads
    relative to the top-level model file. Move the wrapper and every mesh goes
    missing.

    On Kaggle that directory is a symlink into read-only /kaggle/input, so
    writing there fails with OSError: [Errno 30]. When that happens, mirror the
    robot directory into a temp dir with one symlink per entry and write the
    wrapper there instead -- same relative layout, same meshes, but writable.

    Returns the directory to write into, plus the mirror to clean up (or None).
    """
    parent = robot_xml.parent
    if os.access(parent, os.W_OK):
        return parent, None

    mirror = Path(tempfile.mkdtemp(prefix="render_scene_"))
    for entry in parent.iterdir():
        (mirror / entry.name).symlink_to(entry.resolve())
    return mirror, mirror


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--robot-xml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=3.2)
    ap.add_argument(
        "--free-camera",
        action="store_true",
        help="Keep the camera still instead of following the root.",
    )
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.gl)
    import mujoco  # imported after MUJOCO_GL is set

    root_pos, root_quat, dof, src_fps = load_motion(args.motion)

    scene_dir, mirror = writable_scene_dir(args.robot_xml)
    scene_xml = scene_dir / "_render_scene.xml"
    scene_xml.write_text(SCENE_TEMPLATE.format(robot_xml=args.robot_xml.name))
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
    finally:
        scene_xml.unlink(missing_ok=True)
        if mirror is not None:
            shutil.rmtree(mirror, ignore_errors=True)

    data = mujoco.MjData(model)
    expected = model.nq - 7
    if dof.shape[1] != expected:
        print(
            f"[ERROR] motion has {dof.shape[1]} dofs, {args.robot_xml} expects "
            f"{expected}.",
            file=sys.stderr,
        )
        return 1

    stride = max(1, int(round(src_fps / args.fps)))
    idx = range(0, len(dof), stride)

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
            "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    n = 0
    try:
        for i in idx:
            data.qpos[0:3] = root_pos[i]
            data.qpos[3:7] = root_quat[i]
            data.qpos[7:] = dof[i]
            mujoco.mj_forward(model, data)
            if not args.free_camera:
                cam.lookat[:] = root_pos[i]
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            n += 1
    finally:
        proc.stdin.close()
        proc.wait()

    print(f"[INFO] Wrote {args.output}")
    print(f"  {n} frames @ {args.fps} fps (source {src_fps:g} fps, stride {stride})")
    print(f"  {args.width}x{args.height}, backend {os.environ['MUJOCO_GL']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
