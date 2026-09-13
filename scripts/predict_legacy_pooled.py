#!/usr/bin/env python3
"""Predict legacy pooled-feature batches with released .safetensors checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from epicatanexus.legacy_pooled import load_legacy_checkpoint


LEGACY_INPUT_KEYS = {
    "node_features",
    "coordinates",
    "edge_index",
    "edge_features",
    "node_batch",
    "smiles_tokens",
    "t5_features",
    "trfm_features",
    "esm_features",
    "pst_features",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Released legacy pooled checkpoint, e.g. epicatanexus_kcat_pooled.safetensors.",
    )
    parser.add_argument(
        "--batches",
        required=True,
        type=Path,
        help="torch.save payload containing legacy pooled-feature batch dictionaries.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_legacy_batches(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Legacy pooled batch file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    batches = payload.get("batches") if isinstance(payload, dict) and "batches" in payload else payload
    if not isinstance(batches, list) or not batches:
        raise ValueError("Batch payload must be a non-empty list or {'batches': [...]} mapping.")
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise TypeError(f"Batch {index} must be a mapping, got {type(batch).__name__}.")
        missing = sorted(LEGACY_INPUT_KEYS - set(batch))
        if missing:
            raise KeyError(f"Batch {index} is missing legacy pooled inputs: {', '.join(missing)}")
    return batches


def move_tensor_values(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Legacy pooled inference uses the Mamba SMILES branch and normally requires "
            "a CUDA-compatible mamba-ssm runtime. Re-run with --device cuda on a CUDA host."
        )
    model = load_legacy_checkpoint(args.checkpoint, device=device)
    rows = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(load_legacy_batches(args.batches)):
            batch = move_tensor_values(raw_batch, device)
            prediction = model(**{key: batch[key] for key in LEGACY_INPUT_KEYS})
            values = prediction.detach().cpu().view(-1).tolist()
            pair_ids = raw_batch.get(
                "pair_id", [f"batch{batch_index}_{item_index}" for item_index in range(len(values))]
            )
            rows.extend(
                {"pair_id": pair_id, "prediction_log10": value}
                for pair_id, value in zip(pair_ids, values)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} legacy pooled predictions to {args.output}")


if __name__ == "__main__":
    main()
