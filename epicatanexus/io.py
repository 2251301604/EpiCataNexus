"""Prepared-batch I/O shared by training, evaluation, and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


MODEL_INPUT_KEYS = {
    "node_features",
    "coordinates",
    "edge_index",
    "edge_features",
    "node_batch",
    "t5_states",
    "esm_states",
    "sequence_mask",
    "smiles_tokens",
    "smiles_mask",
    "trfm_features",
    "pst_features",
}


def load_prepared_batches(path: str | Path) -> list[dict[str, Any]]:
    batch_path = Path(path)
    if not batch_path.exists():
        raise FileNotFoundError(
            f"Prepared batch file not found: {batch_path}. See docs/DATA.md for the schema."
        )
    payload = torch.load(batch_path, map_location="cpu", weights_only=False)
    batches = payload.get("batches") if isinstance(payload, dict) and "batches" in payload else payload
    if not isinstance(batches, list) or not batches:
        raise ValueError("Prepared batch payload must be a non-empty list or {'batches': [...]} mapping.")
    for index, batch in enumerate(batches):
        validate_batch(batch, index=index)
    return batches


def validate_batch(batch: dict[str, Any], index: int | None = None, require_target: bool = False) -> None:
    if not isinstance(batch, dict):
        raise TypeError(f"Batch {index} must be a mapping, got {type(batch).__name__}.")
    missing = sorted(MODEL_INPUT_KEYS - set(batch))
    if missing:
        raise KeyError(f"Batch {index} is missing model inputs: {', '.join(missing)}")
    if require_target and "target" not in batch:
        raise KeyError(f"Batch {index} is missing 'target'.")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

