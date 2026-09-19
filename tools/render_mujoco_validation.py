"""Render home and per-joint poses for visual review of a MoIRA MuJoCo model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw


def _qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing joint {name}")
    return int(model.jnt_qposadr[joint_id])


def _render_pose(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    positions: dict[str, float],
) -> Image.Image:
    data.qpos[:] = 0
    for name, angle in positions.items():
        data.qpos[_qpos_address(model, name)] = angle
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render().copy())


def render(model_path: Path, home_path: Path, motion_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.10, 0.10, 0.0)
    camera.distance = 0.43
    camera.azimuth = 125
    camera.elevation = -18

    coupling = float(model.eq_data[0][1])
    poses = (
        ("home", {}),
        ("J1 +30 deg", {"J1_BASE_YAW": math.radians(30)}),
        ("J2 +25 deg", {"J2_SHOULDER": math.radians(25)}),
        ("J3 -25 deg", {"J3_ELBOW": math.radians(-25)}),
        (
            "J4 coupled +35 deg",
            {
                "J4_END_EFFECTOR": math.radians(35),
                "J4_END_EFFECTOR_MIRROR": math.radians(35) / coupling,
            },
        ),
    )
    images = [
        (label, _render_pose(renderer, model, data, camera, positions))
        for label, positions in poses
    ]
    renderer.close()

    home_path.parent.mkdir(parents=True, exist_ok=True)
    images[0][1].save(home_path)
    canvas = Image.new("RGB", (1280, 1440), (15, 15, 17))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        column = index % 2
        row = index // 2
        x = column * 640
        y = row * 480
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 225, y + 25), fill=(15, 15, 17))
        draw.text((x + 8, y + 6), label, fill=(245, 245, 245))
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(motion_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("home", type=Path)
    parser.add_argument("motion", type=Path)
    args = parser.parse_args()
    render(args.model, args.home, args.motion)
    print(f"Wrote visual validation renders to {args.home.resolve()} and {args.motion.resolve()}")


if __name__ == "__main__":
    main()
