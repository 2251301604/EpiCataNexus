#!/usr/bin/env python3
"""Strictly validate a released pooled-feature kcat or Km checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from epicatanexus.legacy_pooled import load_legacy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    model = load_legacy_checkpoint(args.checkpoint)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Strict checkpoint validation passed: {args.checkpoint} ({count:,} parameters)")


if __name__ == "__main__":
    main()
