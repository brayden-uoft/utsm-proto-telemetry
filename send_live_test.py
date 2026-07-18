"""Send fake live rows to the dashboard without telemetry hardware."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/live/telemetry",
        help="Full live ingestion URL",
    )
    parser.add_argument("--api-key", default="change-me")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=0, help="0 sends until Ctrl+C")
    parser.add_argument(
        "--gps",
        action="store_true",
        help="Include a small simulated route near Indianapolis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = 0
    boot_id = int(time.time()) & 0xFFFFFFFF
    print(f"Sending test rows to {args.url}; Ctrl+C to stop")

    while args.count == 0 or sequence < args.count:
        phase = sequence / 8.0
        current_ma = round(6500 + 4500 * (0.5 + 0.5 * math.sin(phase)))
        voltage_mv = round(24100 - current_ma * 0.045)
        payload = {
            "device_id": "dashboard-test",
            "source_boot_id": boot_id,
            "sequence": sequence,
            "timestamp_ms": sequence * round(args.interval * 1000),
            "current_mA": current_ma,
            "voltage_mV": voltage_mv,
            "ax_x100": round(80 * math.sin(phase * 1.7)),
            "ay_x100": round(55 * math.cos(phase * 1.3)),
            "az_x100": 981,
            "amag_x100": round(985 + 25 * math.sin(phase)),
        }
        if args.gps:
            payload["latitude"] = 39.79917 + 0.00055 * math.sin(phase / 5)
            payload["longitude"] = -86.23801 + 0.00075 * math.cos(phase / 5)

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            args.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-Telemetry-Key": args.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            print(f"seq={sequence} status={response.status}")
        sequence += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

