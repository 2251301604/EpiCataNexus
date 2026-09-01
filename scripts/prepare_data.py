#!/usr/bin/env python3
"""Validate an enzyme-substrate table and create the nested paper split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"protein_id", "sequence", "smiles", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input TSV manifest.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--outer-test-fraction", type=float, default=0.10)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, sep="\t")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if "row_id" not in frame:
        frame.insert(0, "row_id", range(len(frame)))

    development, test = train_test_split(
        frame,
        test_size=args.outer_test_fraction,
        random_state=args.seed,
        shuffle=True,
    )
    train, valid = train_test_split(
        development,
        test_size=args.inner_validation_fraction,
        random_state=args.seed,
        shuffle=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        split.sort_values("row_id").to_csv(args.output_dir / f"{name}.tsv", sep="\t", index=False)
    print(
        f"Wrote nested split to {args.output_dir}: "
        f"train={len(train)}, valid={len(valid)}, test={len(test)}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()

