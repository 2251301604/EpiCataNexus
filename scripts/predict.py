#!/usr/bin/env python3
"""Predict prepared enzyme-substrate batches with a released checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from epicatanexus.io import MODEL_INPUT_KEYS, load_prepared_batches, move_batch
from epicatanexus.models import EpiCataNexus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EpiCataNexus.from_mapping(state["model_config"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(load_prepared_batches(args.batches)):
            batch = move_batch(raw_batch, device)
            prediction = model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
            values = prediction.detach().cpu().view(-1).tolist()
            pair_ids = raw_batch.get("pair_id", [f"batch{batch_index}_{i}" for i in range(len(values))])
            rows.extend({"pair_id": pair_id, "prediction_log10": value} for pair_id, value in zip(pair_ids, values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

