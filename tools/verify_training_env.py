"""Verify the Windows GPU training environment with real library operations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    import cv2
    import gymnasium as gym
    import lerobot
    import mujoco
    import numpy as np
    import torch
    import torchvision
    from lerobot.policies.act.configuration_act import ACTConfig
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    require(torch.backends.cuda.is_built(), "PyTorch was installed without CUDA support")
    require(torch.cuda.is_available(), "PyTorch cannot access an NVIDIA CUDA device")
    require(torch.cuda.device_count() == 1, "Expected exactly one training GPU")

    left = torch.randn((1024, 1024), device="cuda")
    product = left @ left
    torch.cuda.synchronize()
    require(product.device.type == "cuda", "Matrix multiplication did not run on CUDA")
    require(bool(torch.isfinite(product).all()), "CUDA matrix multiplication was non-finite")

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    require(bool(encoded) and jpeg.size > 0, "OpenCV JPEG encoding failed")

    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.5">
          <freejoint/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initial_height = float(data.qpos[2])
    for _ in range(100):
        mujoco.mj_step(model, data)
    require(data.time > 0, "MuJoCo simulation time did not advance")
    require(float(data.qpos[2]) < initial_height, "MuJoCo gravity did not move the test body")

    environment = gym.make("CartPole-v1")
    observation, _ = environment.reset(seed=0)
    require(observation.shape == (4,), "Gymnasium returned an unexpected observation")
    environment.step(environment.action_space.sample())
    environment.close()

    act = ACTConfig(device="cuda")
    require(act.type == "act", "LeRobot ACT configuration did not initialize")
    train_command = Path(sys.executable).parent / "lerobot-train.exe"
    require(train_command.is_file(), "lerobot-train is not installed beside the active Python")
    help_result = subprocess.run(
        [str(train_command), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    require(help_result.returncode == 0, f"lerobot-train failed: {help_result.stderr[-1000:]}")

    import moira

    del moira
    report = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "opencv": cv2.__version__,
        "mujoco": mujoco.__version__,
        "gymnasium": gym.__version__,
        "lerobot": lerobot.__version__,
        "lerobot_train": "ok",
        "torchcodec": "ok",
        "moira": "ok",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
