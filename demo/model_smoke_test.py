"""Run the bundled example through all public HF checkpoints on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.runtime import run_pipeline, segments_dataframe


def main() -> None:
    """Verify real checkpoint loading, inference, fusion, and post-processing."""

    sample = ROOT / "demo" / "examples" / "synthetic_activity_imu.tsv"
    result = run_pipeline(
        sample,
        fusion_mode="local_boundary",
        min_duration_sec=5,
        confidence_min=0.30,
        top_k=5,
    )
    assert result.model_scales == ("3s", "5s", "8s")
    assert result.probabilities.shape == (118, 6)
    assert result.decoded_path.shape == (118,)
    print(f"model smoke test passed on {result.device}: {len(result.segments)} segments")
    if result.segments:
        print(segments_dataframe(result).to_string(index=False))


if __name__ == "__main__":
    main()
