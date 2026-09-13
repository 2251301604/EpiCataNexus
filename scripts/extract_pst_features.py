#!/usr/bin/env python3
"""Standardize externally computed PST structural features for EpiCataNexus."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="External PST features as .pkl/.npz/.csv/.tsv mapping protein_id to vector.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protein-id-column", default="protein_id")
    parser.add_argument(
        "--feature-prefix",
        default="pst_",
        help="Prefix for vector columns when reading CSV/TSV tables.",
    )
    parser.add_argument("--pst-model-id", required=True, help="Model/checkpoint identifier to record.")
    parser.add_argument("--pooling", default="documented-external", help="Pooling rule to record.")
    parser.add_argument("--expected-dim", type=int, default=1280)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def flatten_vector(value: Any, *, expected_dim: int, protein_id: str) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != expected_dim:
        raise ValueError(
            f"PST dimension mismatch for {protein_id}: expected {expected_dim}, got {array.size}"
        )
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(array, dtype=torch.float32)


def load_feature_mapping(path: Path, *, id_column: str, prefix: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            raise TypeError(".pkl PST features must be a mapping from protein_id to vector.")
        return {str(key): value for key, value in data.items()}
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {str(key): data[key] for key in data.files}
    if suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        if id_column not in frame.columns:
            raise ValueError(f"Feature table is missing id column: {id_column}")
        feature_columns = [column for column in frame.columns if column.startswith(prefix)]
        if not feature_columns:
            raise ValueError(f"No feature columns found with prefix: {prefix}")
        return {
            str(row[id_column]): row[feature_columns].to_numpy(dtype=np.float32)
            for _, row in frame.iterrows()
        }
    raise ValueError("Unsupported PST feature format. Use .pkl, .npz, .csv, or .tsv.")


def main() -> None:
    args = parse_args()
    storage_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    features = load_feature_mapping(
        args.features, id_column=args.protein_id_column, prefix=args.feature_prefix
    )
    tensor_dir = args.output_dir / "pst_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for protein_id, vector in sorted(features.items()):
        key = stable_key(protein_id)
        output = tensor_dir / f"{key}.pt"
        if output.exists() and not args.overwrite:
            payload = torch.load(output, map_location="cpu", weights_only=False)
            feature = payload["pst_features"]
        else:
            feature = flatten_vector(vector, expected_dim=args.expected_dim, protein_id=protein_id)
            payload = {
                "schema_version": 1,
                "protein_id": protein_id,
                "protein_id_sha256": key,
                "pst_model_id": args.pst_model_id,
                "pooling": args.pooling,
                "pst_features": feature.to(dtype=storage_dtype),
            }
            torch.save(payload, output)
        rows.append(
            {
                "protein_id": protein_id,
                "protein_id_sha256": key,
                "feature_dim": int(feature.numel()),
                "tensor_path": str(output.relative_to(args.output_dir)),
            }
        )
        print(f"{protein_id}: PST dim={feature.numel()} -> {output.name}")

    pd.DataFrame(rows).to_csv(args.output_dir / "pst_feature_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(rows)} PST feature records to {args.output_dir}")


if __name__ == "__main__":
    main()
