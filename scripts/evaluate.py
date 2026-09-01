#!/usr/bin/env python3
"""Evaluate a checkpoint on prepared tensor batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from epicatanexus.io import MODEL_INPUT_KEYS, load_prepared_batches, move_batch
from epicatanexus.metrics import regression_metrics
from epicatanexus.models import EpiCataNexus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
            inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
            predictions = model(**inputs).detach().cpu().view(-1).tolist()
            targets = batch["target"].detach().cpu().view(-1).tolist()
            pair_ids = raw_batch.get("pair_id", [f"batch{batch_index}_{i}" for i in range(len(predictions))])
            rows.extend(
                {"pair_id": pair_id, "target": target, "prediction": prediction}
                for pair_id, target, prediction in zip(pair_ids, targets, predictions)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    metrics = regression_metrics(frame["target"], frame["prediction"])
    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

