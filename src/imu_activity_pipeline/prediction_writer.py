"""Output helper for writing activity-segment predictions.

Purpose:
    Validates prediction rows and saves them in the stable workbook schema used
    by evaluation and downstream submission/analysis tools.
Inputs:
    A list of `[user_id, category, start, end]` rows and an output `.xlsx` path.
Outputs:
    Writes an Excel workbook with normalized string labels and int64 timestamps.
"""
from pathlib import Path
from typing import List

import pandas as pd


class DataOutput:
    """Validate and save predicted activity segments as an Excel workbook."""

    def __init__(self, results: List[List], output_file: str = "./predictions_external_test.xlsx"):
        """Store prediction rows and the destination workbook path."""
        self.results = results
        self.output_file = output_file

    def save_predictions(self) -> None:
        """Write rows in ``[user_id, category, start, end]`` format to ``.xlsx``.

        ``start`` and ``end`` are stored as int64 timestamps to preserve
        millisecond-resolution values without scientific-notation drift.
        """
        if not self.results:
            raise ValueError("prediction rows must not be empty")

        if not all(len(row) == 4 for row in self.results):
            raise ValueError("each row must contain [user_id, category, start, end]")

        df = pd.DataFrame(self.results, columns=["user_id", "category", "start", "end"])

        # Normalize dtypes before writing so downstream evaluation sees a stable schema.
        df["user_id"] = df["user_id"].astype(str)
        df["category"] = df["category"].astype(str)
        df["start"] = df["start"].astype("int64")
        df["end"] = df["end"].astype("int64")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.output_file, index=False, engine="openpyxl")

        print(f"Saved {len(df)} prediction rows to {self.output_file}")


if __name__ == "__main__":
    example_rows = [
        ["HNU22001", "A", 10, 1111111111112],
        ["HNU22002", "B", 5, 15],
        ["HNU22003", "C", 0, 30],
    ]
    DataOutput(example_rows).save_predictions()
