"""Input helper for reading raw signal text files.

Purpose:
    Provides a small `DataReader` abstraction used by tests and lightweight
    workflows that need to read all `.txt` signal files from one directory.
Inputs:
    A directory path containing per-user tab-separated sensor text files.
Outputs:
    Returns a dictionary mapping each file stem to its raw text contents.
"""
from pathlib import Path
from typing import Dict


class DataReader:
    """Read text-format signal files from a directory."""

    def __init__(self, dst: str):
        """Store the directory that contains per-user ``.txt`` signal files."""
        self.dst = dst

    def read_data(self) -> Dict[str, str]:
        """Return a ``{file_id: file_text}`` mapping for all ``.txt`` files."""
        out = Path(self.dst)
        result = {}

        if not out.exists():
            print(f"Directory does not exist: {out}")
            return result

        # File IDs are derived from file stems, for example HNU22001.txt -> HNU22001.
        for txt_file in sorted(out.glob("*.txt")):
            file_id = txt_file.stem
            try:
                text = txt_file.read_text(encoding="utf-8")
                result[file_id] = text
            except Exception as e:
                print(f"Failed to read {txt_file}: {e}")

        return result


if __name__ == "__main__":
    reader = DataReader("./data/signals/external_test")
    data = reader.read_data()
    print(data)
