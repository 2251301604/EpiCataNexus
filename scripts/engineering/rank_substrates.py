#!/usr/bin/env python3
"""Rank candidate substrates independently for each enzyme query."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-column", default="query_id")
    parser.add_argument("--score-column", default="score")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    frame["rank"] = frame.groupby(args.query_column)[args.score_column].rank(
        method="first", ascending=False
    ).astype(int)
    frame = frame.sort_values([args.query_column, "rank"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Ranked {len(frame)} candidates across {frame[args.query_column].nunique()} queries.")


if __name__ == "__main__":
    main()

