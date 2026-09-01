#!/usr/bin/env python3
"""Compute predicted double-mutant epistasis on the log10 activity scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"double_score", "single_a_score", "single_b_score", "wild_type_score"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    frame["predicted_epistasis"] = (
        frame["double_score"]
        - frame["single_a_score"]
        - frame["single_b_score"]
        + frame["wild_type_score"]
    )
    frame = frame.sort_values("predicted_epistasis", ascending=False).reset_index(drop=True)
    frame.insert(0, "epistasis_rank", frame.index + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Scored {len(frame)} double mutants: {args.output}")


if __name__ == "__main__":
    main()

