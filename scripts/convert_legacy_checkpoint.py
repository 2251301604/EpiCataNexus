#!/usr/bin/env python3
"""Convert a trusted pooled-feature state dict to safetensors and verify it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from epicatanexus.legacy_pooled import load_legacy_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trusted tensor-only .pkl checkpoint")
    parser.add_argument("output", type=Path, help="Destination .safetensors file")
    args = parser.parse_args()
    if args.output.suffix != ".safetensors":
        raise ValueError("Output filename must end with .safetensors")

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Install the model extras: pip install -e '.[model]'") from exc

    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("Expected a non-empty tensor state dict")
    tensors = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor state entry: {key}")
        # The old state dict has two names for one shared attention gate. Clone each
        # entry because safetensors requires independent storage for the two keys.
        tensors[key] = value.detach().cpu().contiguous().clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={"format": "pt", "architecture": "legacy-pooled-epicatanexus"},
    )
    load_legacy_checkpoint(args.output)
    print(f"Converted and strictly validated: {args.output}")
    print(f"SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
