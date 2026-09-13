#!/usr/bin/env python3
"""Extract pooled SMILES Transformer (TRFM) features for EpiCataNexus batches."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def canonicalize_smiles(smiles: str, *, use_rdkit: bool) -> str:
    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("Encountered an empty SMILES string.")
    if not use_rdkit:
        return smiles
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("Install RDKit or omit --canonicalize-rdkit.") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="TSV/CSV manifest with SMILES.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--compound-id-column", default=None)
    parser.add_argument(
        "--trfm-model",
        required=True,
        help="Hugging Face model id or local path for the SMILES Transformer encoder.",
    )
    parser.add_argument("--separator", default="\t", help="Input delimiter; default is TSV.")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--allow-dim-mismatch", action="store_true")
    parser.add_argument("--canonicalize-rdkit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def encode_smiles(smiles: str, tokenizer, model, device: torch.device, args) -> torch.Tensor:
    encoded = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded)
    states = output.last_hidden_state[0]
    if args.pooling == "cls":
        feature = states[0]
    else:
        mask = encoded.get("attention_mask")
        if mask is None:
            feature = states.mean(dim=0)
        else:
            weights = mask[0].to(states.dtype).unsqueeze(-1)
            feature = (states * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
    return feature.detach()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, sep=args.separator)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"Input is missing SMILES column: {args.smiles_column}")
    if args.compound_id_column and args.compound_id_column not in frame.columns:
        raise ValueError(f"Input is missing compound id column: {args.compound_id_column}")

    device = resolve_device(args.device)
    storage_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.trfm_model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        args.trfm_model, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    tensor_dir = args.output_dir / "trfm_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    unique_columns = [args.smiles_column]
    if args.compound_id_column:
        unique_columns.insert(0, args.compound_id_column)
    unique = frame[unique_columns].drop_duplicates()
    rows = []
    for values in unique.itertuples(index=False, name=None):
        if args.compound_id_column:
            compound_id, raw_smiles = values
            compound_id = str(compound_id)
        else:
            raw_smiles = values[0]
            compound_id = stable_key(str(raw_smiles))[:16]
        smiles = canonicalize_smiles(raw_smiles, use_rdkit=args.canonicalize_rdkit)
        key = stable_key(smiles)
        output = tensor_dir / f"{key}.pt"
        if output.exists() and not args.overwrite:
            payload = torch.load(output, map_location="cpu", weights_only=False)
            feature = payload["trfm_features"]
        else:
            feature = encode_smiles(smiles, tokenizer, model, device, args).to("cpu")
            if feature.ndim != 1:
                raise RuntimeError(f"Expected a flat TRFM vector, got shape {tuple(feature.shape)}")
            if feature.numel() != args.expected_dim and not args.allow_dim_mismatch:
                raise RuntimeError(
                    f"TRFM dimension mismatch for {compound_id}: expected "
                    f"{args.expected_dim}, got {feature.numel()}. Use --expected-dim or "
                    "--allow-dim-mismatch only if the downstream model was configured for it."
                )
            payload = {
                "schema_version": 1,
                "compound_id": compound_id,
                "smiles": smiles,
                "smiles_sha256": key,
                "trfm_model": args.trfm_model,
                "pooling": args.pooling,
                "trfm_features": feature.to(dtype=storage_dtype),
            }
            torch.save(payload, output)
        rows.append(
            {
                "compound_id": compound_id,
                "smiles_sha256": key,
                "feature_dim": int(feature.numel()),
                "tensor_path": str(output.relative_to(args.output_dir)),
            }
        )
        print(f"{compound_id}: TRFM dim={feature.numel()} -> {output.name}")

    pd.DataFrame(rows).to_csv(args.output_dir / "trfm_feature_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(rows)} TRFM feature records to {args.output_dir}")


if __name__ == "__main__":
    main()
