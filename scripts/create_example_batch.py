#!/usr/bin/env python3
"""Create a tiny synthetic prepared batch for interface validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from epicatanexus.io import validate_batch


def create_example_batch(seed: int = 3407) -> dict:
    generator = torch.Generator().manual_seed(seed)
    node_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
            [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5],
        ],
        dtype=torch.long,
    )
    batch = {
        "node_features": torch.randn(6, 51, generator=generator),
        "coordinates": torch.randn(6, 3, generator=generator),
        "edge_index": edge_index,
        "edge_features": torch.randn(edge_index.size(1), 92, generator=generator),
        "node_batch": node_batch,
        "t5_states": torch.randn(2, 5, 1024, generator=generator),
        "esm_states": torch.randn(2, 5, 1280, generator=generator),
        "sequence_mask": torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
        ),
        "smiles_tokens": torch.randint(0, 600, (2, 12), generator=generator),
        "smiles_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1] * 12], dtype=torch.bool
        ),
        "trfm_features": torch.randn(2, 1024, generator=generator),
        "pst_features": torch.randn(2, 1280, generator=generator),
        "target": torch.tensor([0.0, 1.0]),
        "pair_id": ["synthetic_001", "synthetic_002"],
    }
    validate_batch(batch, require_target=True)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/example_batches.pt"))
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"batches": [create_example_batch(args.seed)]}, args.output)
    print(f"Synthetic interface example written to: {args.output}")


if __name__ == "__main__":
    main()
