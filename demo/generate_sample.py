"""Generate the deterministic, non-participant IMU example bundled with the demo."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "examples" / "synthetic_activity_imu.tsv"
RATE_HZ = 100
DURATION_SEC = 120
START_MS = 1_700_000_000_000
BASE = (1_915.0, -2_223.0, -94.0, -69.0, -49.0, -1.0)


def _motion(t: float) -> tuple[float, float, float, float, float, float]:
    """Create quiet, periodic, quiet, and mixed-motion phases."""

    if 20.0 <= t < 58.0:
        phase = t - 20.0
        return (
            2_250.0 * math.sin(2 * math.pi * 2.15 * phase),
            1_180.0 * math.sin(2 * math.pi * 2.15 * phase + 0.8),
            3_350.0 * math.sin(2 * math.pi * 4.30 * phase + 0.2),
            1_050.0 * math.sin(2 * math.pi * 2.15 * phase + 1.1),
            720.0 * math.sin(2 * math.pi * 2.15 * phase + 0.1),
            1_380.0 * math.sin(2 * math.pi * 4.30 * phase + 0.5),
        )
    if 72.0 <= t < 112.0:
        phase = t - 72.0
        stride = math.sin(2 * math.pi * 2.75 * phase)
        harmonic = math.sin(2 * math.pi * 5.50 * phase + 0.35)
        return (
            1_650.0 * stride + 420.0 * harmonic,
            1_300.0 * math.sin(2 * math.pi * 2.75 * phase + 1.7),
            2_050.0 * abs(stride) - 1_000.0,
            860.0 * math.sin(2 * math.pi * 2.75 * phase + 0.6),
            1_120.0 * math.sin(2 * math.pi * 2.75 * phase + 1.9),
            690.0 * harmonic,
        )
    return (
        45.0 * math.sin(2 * math.pi * 0.20 * t),
        38.0 * math.sin(2 * math.pi * 0.17 * t + 0.4),
        55.0 * math.sin(2 * math.pi * 0.13 * t + 1.2),
        22.0 * math.sin(2 * math.pi * 0.18 * t),
        18.0 * math.sin(2 * math.pi * 0.11 * t + 0.7),
        16.0 * math.sin(2 * math.pi * 0.15 * t + 1.1),
    )


def main() -> None:
    """Write a reproducible 120-second, seven-column TSV recording."""

    rng = random.Random(20260821)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("ACC_TIME", "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"))
        for index in range(RATE_HZ * DURATION_SEC):
            time_sec = index / RATE_HZ
            motion = _motion(time_sec)
            noise_scale = 22.0 if 20.0 <= time_sec < 112.0 else 6.0
            values = [
                base + delta + rng.gauss(0.0, noise_scale)
                for base, delta in zip(BASE, motion)
            ]
            writer.writerow(
                [START_MS + index * 10] + [str(round(value)) for value in values]
            )
    print(f"wrote {OUTPUT} ({RATE_HZ * DURATION_SEC:,} samples)")


if __name__ == "__main__":
    main()
