#!/usr/bin/env python3
# utility script to create a "head" version of a dataset file
# by copying the first N lines and backing up the original
# usage: python make_head_copy.py <dataset.jsonl> [N]
# designed for testing and quick iterations on smaller datasets

import shutil
import sys
from pathlib import Path

def make_head_copy(path: str, n: int = 500):
    """Create a head copy of the dataset file with first N lines."""
    orig = Path(path)
    bak = orig.with_suffix(orig.suffix + ".bak")
    if not orig.exists():
        print(f"❌ File not found: {orig}")
        return

    # Move original to .bak if not already
    if not bak.exists():
        shutil.move(orig, bak)
        print(f"📦 Moved {orig} → {bak}")
    else:
        print(f"⚠️ Backup already exists at {bak}, not overwriting.")

    # Copy first N lines into new dataset file
    with open(bak, "r") as fin, open(orig, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)

    print(f"✅ Created head dataset with first {n} rows at {orig}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_head_copy.py <dataset.jsonl> [N]")
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    make_head_copy(path, n)
