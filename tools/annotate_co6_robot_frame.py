"""Annotate the robot origin and forward direction in one CO6 frame."""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.brand import DISPLAY_NAME  # noqa: E402


LABELS = (
    "base_yaw_center",
    "base_forward_marker",
)


class LandmarkWindow:
    def __init__(self, image_path: Path, output_path: Path, scale: float) -> None:
        self.image_path = image_path
        self.output_path = output_path
        self.scale = scale
        self.image = Image.open(image_path).convert("RGB")
        self.points: list[tuple[float, float]] = []

        self.root = tk.Tk()
        self.root.title(f"{DISPLAY_NAME} | Mark robot frame")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        displayed = self.image.resize(
            (round(self.image.width * scale), round(self.image.height * scale)),
            Image.Resampling.NEAREST,
        )
        self.photo = ImageTk.PhotoImage(displayed)
        self.canvas = tk.Canvas(
            self.root,
            width=displayed.width,
            height=displayed.height,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)

        self.status = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.status,
            font=("Segoe UI", 11, "bold"),
            wraplength=900,
        ).pack(padx=12, pady=8)
        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 10))
        tk.Button(controls, text="Undo", command=self.undo, width=12).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(controls, text="Reset", command=self.reset, width=12).pack(
            side=tk.LEFT, padx=5
        )
        self.save_button = tk.Button(
            controls, text="Save landmarks", command=self.save, width=18, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.update_status()

    def update_status(self) -> None:
        if len(self.points) == 0:
            message = "1/2: Click the center of the base yaw rotation axis."
        elif len(self.points) == 1:
            message = "2/2: Click the black-tape point directly in front of the base."
        else:
            message = "Check both markers, then click Save landmarks."
        self.status.set(message)
        self.save_button.configure(
            state=tk.NORMAL if len(self.points) == len(LABELS) else tk.DISABLED
        )

    def redraw(self) -> None:
        self.canvas.delete("marker")
        colors = ("#00e5ff", "#ff2bd6")
        for index, ((x, y), color) in enumerate(zip(self.points, colors), start=1):
            sx, sy = x * self.scale, y * self.scale
            radius = 10
            self.canvas.create_oval(
                sx - radius,
                sy - radius,
                sx + radius,
                sy + radius,
                outline=color,
                width=3,
                tags="marker",
            )
            self.canvas.create_text(
                sx + 15,
                sy - 15,
                text=str(index),
                fill=color,
                font=("Segoe UI", 14, "bold"),
                tags="marker",
            )
        if len(self.points) == 2:
            (x0, y0), (x1, y1) = self.points
            self.canvas.create_line(
                x0 * self.scale,
                y0 * self.scale,
                x1 * self.scale,
                y1 * self.scale,
                fill="#ff2bd6",
                width=4,
                arrow=tk.LAST,
                tags="marker",
            )

    def click(self, event: tk.Event) -> None:
        if len(self.points) >= len(LABELS):
            return
        x = min(max(float(event.x) / self.scale, 0.0), self.image.width - 1.0)
        y = min(max(float(event.y) / self.scale, 0.0), self.image.height - 1.0)
        self.points.append((x, y))
        self.redraw()
        self.update_status()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        self.redraw()
        self.update_status()

    def reset(self) -> None:
        self.points.clear()
        self.redraw()
        self.update_status()

    def save(self) -> None:
        if len(self.points) != len(LABELS):
            return
        record = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_image": str(self.image_path.resolve()),
            "resolution_px": [self.image.width, self.image.height],
            "base_center_px": list(self.points[0]),
            "base_forward_px": list(self.points[1]),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self.root.destroy()

    def run(self) -> bool:
        self.root.mainloop()
        return self.output_path.exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/camera_workflow/extrinsic_capture.jpg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/camera_workflow/robot_frame_landmarks.json"),
    )
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not 0.5 <= args.scale <= 3.0:
        parser.error("--scale must be between 0.5 and 3.0")
    if args.output.exists():
        args.output.unlink()
    saved = LandmarkWindow(args.image, args.output, args.scale).run()
    if not saved:
        print("No landmarks saved")
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
