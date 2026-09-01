#!/usr/bin/env python3
"""Extract residue-aligned ProtT5 and ESM-2 states required by EpiCataNexus."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer


NON_STANDARD_RESIDUES = re.compile(r"[UZOB*]")


def normalize_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).split()).upper()
    if not sequence:
        raise ValueError("Encountered an empty protein sequence.")
    return NON_STANDARD_RESIDUES.sub("X", sequence)


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="TSV with protein_id and sequence.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protein-id-column", default="protein_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--prott5-model", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--esm2-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def extract_prott5(sequence, tokenizer, model, device):
    encoded = tokenizer(" ".join(sequence), add_special_tokens=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).last_hidden_state[0][: len(sequence)]


@torch.inference_mode()
def extract_esm2(sequence, tokenizer, model, device):
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).last_hidden_state[0][1 : 1 + len(sequence)]


def main() -> None:
    args = parse_args()
    if args.max_residues <= 0:
        raise ValueError("--max-residues must be positive.")
    frame = pd.read_csv(args.input, sep="\t")
    required = {args.protein_id_column, args.sequence_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    storage_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    t5_tokenizer = T5Tokenizer.from_pretrained(args.prott5_model, do_lower_case=False)
    t5_model = T5EncoderModel.from_pretrained(args.prott5_model).to(device).eval()
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    esm_model = AutoModel.from_pretrained(args.esm2_model).to(device).eval()

    tensor_dir = args.output_dir / "residue_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    unique = frame[[args.protein_id_column, args.sequence_column]].drop_duplicates()
    for protein_id, raw_sequence in unique.itertuples(index=False, name=None):
        protein_id = str(protein_id)
        full_sequence = normalize_sequence(str(raw_sequence))
        sequence = full_sequence[: args.max_residues]
        key = sequence_key(full_sequence)
        output = tensor_dir / f"{key}.pt"
        if output.exists() and not args.overwrite:
            payload = torch.load(output, map_location="cpu", weights_only=False)
        else:
            t5_states = extract_prott5(sequence, t5_tokenizer, t5_model, device)
            esm_states = extract_esm2(sequence, esm_tokenizer, esm_model, device)
            aligned_length = min(len(sequence), t5_states.size(0), esm_states.size(0))
            if aligned_length != len(sequence):
                raise RuntimeError(
                    f"Residue alignment failed for {protein_id}: sequence={len(sequence)}, "
                    f"ProtT5={t5_states.size(0)}, ESM-2={esm_states.size(0)}"
                )
            payload = {
                "schema_version": 1,
                "protein_id": protein_id,
                "sequence_sha256": key,
                "sequence_length": len(full_sequence),
                "encoded_residues": aligned_length,
                "truncated": len(full_sequence) > args.max_residues,
                "t5_states": t5_states.to("cpu", dtype=storage_dtype),
                "esm_states": esm_states.to("cpu", dtype=storage_dtype),
                "sequence_mask": torch.ones(aligned_length, dtype=torch.bool),
            }
            torch.save(payload, output)
        rows.append(
            {
                "protein_id": protein_id,
                "sequence_sha256": key,
                "sequence_length": payload["sequence_length"],
                "encoded_residues": payload["encoded_residues"],
                "truncated": payload["truncated"],
                "tensor_path": str(output.relative_to(args.output_dir)),
            }
        )
        print(f"{protein_id}: {payload['encoded_residues']} aligned residues -> {output.name}")

    pd.DataFrame(rows).to_csv(args.output_dir / "residue_feature_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(rows)} residue feature records to {args.output_dir}")


if __name__ == "__main__":
    main()
