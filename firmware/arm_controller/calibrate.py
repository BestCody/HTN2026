"""Interactive PCA9685 servo sweep for finding safe pulse-width limits.

Run on the Pi (or with the driver reachable) after wiring one servo at a time.
    python -m arm_controller.calibrate --channel 0
Enter microseconds to command; type `q` to quit. Note the µs where the arm
reaches its safe travel endpoints, then copy them into config.yaml.
"""

from __future__ import annotations

import argparse
import sys

from .config import ArmConfig, default_config_path
from .servo_driver import make_driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--channel", type=int, required=True, help="PCA9685 channel (0-15)")
    parser.add_argument("--start-us", type=int, default=1500, help="Initial pulse width")
    args = parser.parse_args()

    cfg = ArmConfig.load(args.config)
    driver = make_driver(cfg.i2c_address, cfg.pwm_frequency)

    print(f"Driver: {'PCA9685' if driver.hardware else 'MOCK (no hardware)'}")
    print("Enter pulse width in microseconds (typical safe range 500-2500).")
    print("Type 'r' to release the servo, 'q' to quit.")

    us = args.start_us
    try:
        driver.write_us(args.channel, us)
        print(f"channel {args.channel} -> {us} µs")
        while True:
            entry = input("us> ").strip().lower()
            if entry in ("q", "quit", "exit"):
                break
            if entry in ("r", "release"):
                driver.release(args.channel)
                print("released")
                continue
            try:
                new_us = int(entry)
            except ValueError:
                print("expected integer µs, 'r', or 'q'")
                continue
            if not (200 <= new_us <= 2800):
                print("refusing pulse width outside 200-2800 µs")
                continue
            driver.write_us(args.channel, new_us)
            us = new_us
            print(f"channel {args.channel} -> {us} µs")
    finally:
        driver.shutdown()


if __name__ == "__main__":
    sys.exit(main())
