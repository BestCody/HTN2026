"""List I2C addresses visible to the Raspberry Pi without moving hardware."""

from __future__ import annotations

import json
from time import monotonic, sleep

import board
import busio


def main() -> int:
    i2c = busio.I2C(board.SCL, board.SDA)
    deadline = monotonic() + 2.0
    while not i2c.try_lock():
        if monotonic() >= deadline:
            raise RuntimeError("I2C bus remained locked for two seconds")
        sleep(0.01)
    try:
        addresses = i2c.scan()
    finally:
        i2c.unlock()
        deinit = getattr(i2c, "deinit", None)
        if callable(deinit):
            deinit()
    print(
        json.dumps(
            {
                "addresses": [f"0x{address:02x}" for address in addresses],
                "pca9685_at_0x40": 0x40 in addresses,
            }
        )
    )
    return 0 if 0x40 in addresses else 2


if __name__ == "__main__":
    raise SystemExit(main())
