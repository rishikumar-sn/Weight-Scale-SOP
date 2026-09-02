from __future__ import annotations

import sys
import time

import serial


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM12"
    print(f"Opening scale on {port} at 9600 baud...")
    try:
        with serial.Serial(port, 9600, timeout=1.0) as connection:
            connection.reset_input_buffer()
            deadline = time.monotonic() + 8.0
            samples = 0
            while time.monotonic() < deadline and samples < 5:
                raw = connection.readline()
                if raw:
                    samples += 1
                    print(f"  sample {samples}: {raw!r}")
    except serial.SerialException as exc:
        print(f"Scale port failed: {exc}", file=sys.stderr)
        return 1
    if samples == 0:
        print("The port opened, but the scale sent no data in 8 seconds.", file=sys.stderr)
        return 2
    print("Scale serial connection is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
