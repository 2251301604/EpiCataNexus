#!/usr/bin/env python3
"""Rank enzyme variants relative to a wild-type prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-column", default="prediction_log10")
    parser.add_argument("--variant-column", default="variant_id")
    parser.add_argument("--wild-type", required=True, help="Variant ID representing wild type.")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    wild_type = frame.loc[frame[args.variant_column] == args.wild_type, args.score_column]
    if len(wild_type) != 1:
        raise ValueError("Exactly one wild-type row is required.")
    frame["delta_log10"] = frame[args.score_column] - float(wild_type.iloc[0])
    frame["predicted_fold_change"] = np.power(10.0, frame["delta_log10"])
    frame = frame.sort_values("delta_log10", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Ranked {len(frame)} variants: {args.output}")


if __name__ == "__main__":
    main()

