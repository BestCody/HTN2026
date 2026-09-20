"""Generate a physically sized checkerboard for CO6 camera calibration."""

from __future__ import annotations

import argparse
from pathlib import Path


def checkerboard_svg(columns: int, rows: int, square_mm: float) -> str:
    if columns < 3 or rows < 3 or square_mm <= 0:
        raise ValueError("Checkerboard dimensions and square size must be positive")
    page_width, page_height = 216.0, 279.0
    board_width, board_height = columns * square_mm, rows * square_mm
    if board_width > 190 or board_height > 210:
        raise ValueError("Checkerboard does not fit on a Letter/A4 page")
    left = (page_width - board_width) / 2
    top = 24.0
    squares = []
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                squares.append(
                    f'<rect x="{left + column * square_mm:.3f}" '
                    f'y="{top + row * square_mm:.3f}" width="{square_mm:.3f}" '
                    f'height="{square_mm:.3f}" fill="black"/>'
                )
    origin_x, origin_y = left + square_mm, top + square_mm
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}mm" '
                f'height="{page_height}mm" viewBox="0 0 {page_width} {page_height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            *squares,
            (
                f'<path d="M {origin_x - 12:.3f} {origin_y:.3f} '
                f'L {origin_x - 3:.3f} {origin_y - 3:.3f} '
                f'L {origin_x - 3:.3f} {origin_y + 3:.3f} Z" fill="#e00000"/>'
            ),
            (
                f'<text x="{origin_x - 13:.3f}" y="{origin_y - 5:.3f}" '
                'font-family="sans-serif" font-size="4" fill="#e00000">ORIGIN</text>'
            ),
            (
                f'<text x="{left:.3f}" y="{top + board_height + 12:.3f}" '
                'font-family="sans-serif" font-size="4" fill="black">'
                f'{columns - 1} x {rows - 1} inner corners; square = {square_mm:.2f} mm. '
                'Print at 100% / Actual Size.</text>'
            ),
            (
                f'<line x1="{left:.3f}" y1="{top + board_height + 22:.3f}" '
                f'x2="{left + 100:.3f}" y2="{top + board_height + 22:.3f}" '
                'stroke="black" stroke-width="0.5"/>'
            ),
            (
                f'<text x="{left:.3f}" y="{top + board_height + 28:.3f}" '
                'font-family="sans-serif" font-size="4">100 mm verification line</text>'
            ),
            "</svg>",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=9, help="Checkerboard squares")
    parser.add_argument("--rows", type=int, default=7, help="Checkerboard squares")
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("calibration/co6_checkerboard.svg")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        checkerboard_svg(args.columns, args.rows, args.square_mm) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {(args.columns - 1, args.rows - 1)} inner corners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
