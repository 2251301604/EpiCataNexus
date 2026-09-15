#!/usr/bin/env python3
"""Infer kcat and Km from a protein structure PDB and a substrate SMILES string.

This script targets the released legacy pooled-feature checkpoints. It performs the
feature steps needed by that compatibility path:

1. pooled ProtT5 and ESM-2 features from a protein sequence;
2. a structure graph from a PDB, or from an optional pocket-only PDB;
3. pooled PST features from the full protein PDB;
4. SMILES-Mamba tokens and TRFM features from the substrate SMILES;
5. kcat and Km prediction with the released `.safetensors` checkpoints.

Sequence-only inference is intentionally not supported. If only a sequence is
available, first generate or provide a structure, for example with AlphaFold, then
run this script with `--pdb`.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, T5EncoderModel, T5Tokenizer

from epicatanexus.legacy_pooled import load_legacy_checkpoint


NON_STANDARD_RESIDUES = re.compile(r"[UZOB*]")
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


class MolTranBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str = "",
        do_lower_case: bool = False,
        unk_token: str = "<pad>",
        sep_token: str = "<eos>",
        pad_token: str = "<pad>",
        cls_token: str = "<bos>",
        mask_token: str = "<mask>",
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.regex_tokenizer = SMILES_PATTERN
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text: str) -> list[str]:
        return self.regex_tokenizer.findall(text)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).strip()


class WordVocab:
    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def load_vocab(path: str | Path) -> "WordVocab":
        import pickle

        with open(path, "rb") as handle:
            return pickle.load(handle)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(0)].transpose(0, 1)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(src) * math.sqrt(self.hidden_size)
        embedded = self.pe(embedded)
        memory = self.trfm.encoder(embedded)
        return torch.cat(
            [memory[0], torch.max(memory, dim=0).values, torch.mean(memory, dim=0), memory[-1]],
            dim=1,
        )


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).split()).upper()
    if not sequence:
        raise ValueError("Encountered an empty protein sequence.")
    return NON_STANDARD_RESIDUES.sub("X", sequence)


def truncate_sequence(sequence: str, max_residues: int, mode: str) -> str:
    if max_residues <= 0:
        raise ValueError("--max-residues must be positive.")
    if len(sequence) <= max_residues:
        return sequence
    if mode == "head":
        return sequence[:max_residues]
    if mode == "balanced":
        left = max_residues // 2
        right = max_residues - left
        return sequence[:left] + sequence[-right:]
    raise ValueError(f"Unsupported truncate mode: {mode}")


def read_sequence_file(path: Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                lines.append(line)
    return normalize_sequence("".join(lines))


def infer_sequence_from_pdb(path: Path, chain: str | None) -> str:
    residues = []
    seen = set()
    with open_text(path) as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue
            if chain and len(line) > 21 and line[21].strip() != chain:
                continue
            residue_key = (line[21].strip(), line[22:27].strip(), line[17:20].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            residues.append(THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
    if not residues:
        raise ValueError(f"No protein ATOM residues found in {path} for chain={chain!r}")
    return normalize_sequence("".join(residues))


@torch.inference_mode()
def extract_t5_pooled(sequence: str, tokenizer, model, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(" ".join(sequence), add_special_tokens=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = model(**encoded).last_hidden_state[0]
    return hidden[: len(sequence)].float().mean(dim=0).cpu()


@torch.inference_mode()
def extract_esm_pooled(sequence: str, tokenizer, model, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = model(**encoded).last_hidden_state[0]
    return hidden[1 : 1 + len(sequence)].float().mean(dim=0).cpu()


def load_protein_language_models(args, device: torch.device):
    t5_tokenizer = T5Tokenizer.from_pretrained(args.prott5_model, do_lower_case=False)
    t5_model = T5EncoderModel.from_pretrained(args.prott5_model).to(device).eval()
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = AutoModel.from_pretrained(args.esm_model).to(device).eval()
    return t5_tokenizer, t5_model, esm_tokenizer, esm_model


def get_cb(n, ca, c):
    b = ca - n
    c_vec = c - ca
    a = torch.linalg.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def parse_pdb_atoms(pdb_file: Path, chain: str | None = None, cal_cb: bool = True) -> torch.Tensor:
    fillna = torch.zeros(3, dtype=torch.float32)
    current_key = None
    current_aa: dict[str, torch.Tensor] = {}
    residues = []

    def flush_current() -> None:
        if not current_aa:
            return
        r_group = [value for atom, value in current_aa.items() if atom not in {"N", "CA", "C", "O"}]
        r_group_tensor = torch.stack(r_group).mean(dim=0) if r_group else fillna
        residues.append(
            torch.stack(
                [
                    current_aa.get("N", fillna),
                    current_aa.get("CA", fillna),
                    current_aa.get("C", fillna),
                    current_aa.get("O", fillna),
                    r_group_tensor,
                ]
            )
        )

    with open_text(pdb_file) as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "TER":
                flush_current()
                current_aa = {}
                current_key = None
                continue
            if record != "ATOM":
                continue
            if chain and len(line) > 21 and line[21].strip() != chain:
                continue
            key = (line[21].strip(), line[22:27].strip())
            if current_key is not None and key != current_key:
                flush_current()
                current_aa = {}
            current_key = key
            atom = line[12:16].strip()
            if atom.startswith("H"):
                continue
            try:
                xyz = torch.tensor(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=torch.float32,
                )
            except ValueError:
                xyz = fillna
            current_aa[atom] = xyz
    flush_current()

    if not residues:
        raise ValueError(f"No ATOM residues found in {pdb_file} for chain={chain!r}")
    x = torch.stack(residues)
    if cal_cb:
        cb = get_cb(x[:, 0], x[:, 1], x[:, 2]).unsqueeze(1)
        x = torch.cat([x, cb], dim=1)
    return x


def positional_encodings(edge_index: torch.Tensor, num_embeddings: int = 16) -> torch.Tensor:
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(math.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


def get_angle(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x_orig = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
    dx = x_orig[1:] - x_orig[:-1]
    u = F.normalize(dx, dim=-1)
    u_2, u_1, u_0 = u[:-2], u[1:-1], u[2:]
    n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)
    cos_d = torch.clamp(torch.sum(n_2 * n_1, -1), -1 + eps, 1 - eps)
    d = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cos_d)
    dihedral = torch.cat(
        [torch.cos(F.pad(d, [1, 2]).reshape([-1, 3])), torch.sin(F.pad(d, [1, 2]).reshape([-1, 3]))],
        1,
    )
    cos_bond = torch.clamp((u_2 * u_1).sum(-1), -1 + eps, 1 - eps)
    bond = torch.acos(cos_bond)
    bond_angles = torch.cat(
        (
            torch.cos(F.pad(bond, [1, 2]).reshape([-1, 3])),
            torch.sin(F.pad(bond, [1, 2]).reshape([-1, 3])),
        ),
        1,
    )
    return torch.cat((dihedral, bond_angles), 1)


def rbf(distance: torch.Tensor, d_min: float = 0.0, d_max: float = 20.0, count: int = 1):
    mu = torch.linspace(d_min, d_max, count).view([1, -1])
    return torch.exp(-((torch.unsqueeze(distance, -1) - mu) / ((d_max - d_min) / count)) ** 2)


def get_distance(x: torch.Tensor, edge_index: torch.Tensor, count: int = 1):
    atom_coords = [x[:, i] for i in range(6)]
    node_dist = []
    for i in range(6):
        for j in range(i + 1, 6):
            node_dist.append(rbf((atom_coords[i] - atom_coords[j]).norm(dim=-1), count=count))
    edge_dist = []
    for i in range(6):
        for j in range(6):
            edge_dist.append(
                rbf((atom_coords[i][edge_index[0]] - atom_coords[j][edge_index[1]]).norm(dim=-1), count=count)
            )
    return torch.cat(node_dist, -1), torch.cat(edge_dist, -1)


def get_direction_orientation(x: torch.Tensor, edge_index: torch.Tensor):
    x_n, x_ca, x_c = x[:, 0], x[:, 1], x[:, 2]
    u = F.normalize(x_ca - x_n, dim=-1)
    v = F.normalize(x_c - x_ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.linalg.cross(u, v), dim=-1)
    q = torch.stack([b, n, torch.linalg.cross(b, n)], -1)
    node_dir = torch.matmul(F.normalize(x[:, [0, 2, 3, 4, 5]] - x_ca.unsqueeze(1), dim=-1), q).reshape(x.shape[0], -1)
    node_j, node_i = edge_index
    edge_dir = torch.cat(
        [
            torch.matmul(F.normalize(x[node_j] - x_ca[node_i].unsqueeze(1), dim=-1), q[node_i]).reshape(node_j.shape[0], -1),
            torch.matmul(F.normalize(x[node_i] - x_ca[node_j].unsqueeze(1), dim=-1), q[node_j]).reshape(node_j.shape[0], -1),
        ],
        -1,
    )
    rot = torch.matmul(q[node_i].transpose(-1, -2), q[node_j])
    diag = torch.diagonal(rot, dim1=-2, dim2=-1)
    xyz = torch.sign(
        torch.stack(
            [rot[:, 2, 1] - rot[:, 1, 2], rot[:, 0, 2] - rot[:, 2, 0], rot[:, 1, 0] - rot[:, 0, 1]],
            -1,
        )
    ) * (
        0.5
        * torch.sqrt(
            torch.abs(
                1
                + torch.stack(
                    [
                        diag[:, 0] - diag[:, 1] - diag[:, 2],
                        -diag[:, 0] + diag[:, 1] - diag[:, 2],
                        -diag[:, 0] - diag[:, 1] + diag[:, 2],
                    ],
                    -1,
                )
            )
        )
    )
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    return node_dir, edge_dir, F.normalize(torch.cat((xyz, w), -1), dim=-1)


def get_dssp_features(pdb_path: Path, n_residues: int, dssp_bin: str | None) -> tuple[torch.Tensor, str]:
    if not dssp_bin:
        return torch.zeros((n_residues, 9), dtype=torch.float32), "zero_filled_no_dssp_bin"
    dssp_path = Path(dssp_bin).expanduser()
    if not dssp_path.exists():
        return torch.zeros((n_residues, 9), dtype=torch.float32), f"zero_filled_missing_dssp:{dssp_bin}"

    ss_map = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7}
    try:
        result = subprocess.run(
            [str(dssp_path), "--output-format", "dssp", str(pdb_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        header_idx = next(
            (i for i, line in enumerate(lines) if "#" in line and "RESIDUE" in line and "AA" in line),
            -1,
        )
        if header_idx < 0:
            return torch.zeros((n_residues, 9), dtype=torch.float32), "zero_filled_unparsed_dssp"
        acc_pos = lines[header_idx].find("ACC")
        features = torch.zeros((n_residues, 9), dtype=torch.float32)
        for i, line in enumerate(lines[header_idx + 1 :]):
            if i >= n_residues:
                break
            if len(line) < 30:
                continue
            ss_char = line[16] if len(line) > 16 else "-"
            if ss_char == " ":
                ss_char = "-"
            try:
                abs_asa = float(line[acc_pos - 2 : acc_pos + 3].strip())
                rel_asa = min(abs_asa / 200.0, 1.0)
            except Exception:
                rel_asa = 0.0
            features[i, ss_map.get(ss_char, 7)] = 1.0
            features[i, 8] = rel_asa
        return features, "dssp"
    except Exception as exc:
        return torch.zeros((n_residues, 9), dtype=torch.float32), f"zero_filled_dssp_error:{exc}"


def build_graph_from_pdb(pdb_path: Path, chain: str | None, radius: float, dssp_bin: str | None = None) -> Data:
    x = parse_pdb_atoms(pdb_path, chain=chain).float()
    ca = x[:, 1]
    dist_matrix = torch.cdist(ca, ca)
    edge_index = ((dist_matrix <= radius) & (dist_matrix > 0)).nonzero().t().contiguous()
    if edge_index.numel() == 0:
        raise RuntimeError(f"No graph edges generated for {pdb_path}; check structure or radius.")
    node_angles = get_angle(x)
    node_dist, edge_dist = get_distance(x, edge_index)
    node_dir, edge_dir, edge_ori = get_direction_orientation(x, edge_index)
    dssp_features, dssp_status = get_dssp_features(pdb_path, x.size(0), dssp_bin)
    node_features = torch.cat([node_angles, node_dist, node_dir, dssp_features], dim=-1)
    edge_features = torch.cat([positional_encodings(edge_index), edge_ori, edge_dist, edge_dir], dim=-1)
    if node_features.size(1) != 51 or edge_features.size(1) != 92:
        raise RuntimeError(
            f"Unexpected graph dimensions: node={node_features.size(1)}, edge={edge_features.size(1)}"
        )
    graph = Data(
        x=torch.nan_to_num(node_features, 0.0),
        pos=torch.nan_to_num(ca, 0.0),
        edge_index=edge_index,
        edge_attr=torch.nan_to_num(edge_features, 0.0),
        name=pdb_path.stem,
    )
    graph.dssp_status = dssp_status
    return graph


def split_smiles(smiles: str) -> str:
    bracket_regex = r"(\[[^\[\]]{1,10}\])"
    parts = re.split(bracket_regex, str(smiles).strip())
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            tokens.append(part)
        else:
            tokens.extend(list(part))
    return " ".join(tokens)


def trfm_inputs(smiles_batch: list[str], vocab: WordVocab, seq_len: int = 220) -> torch.Tensor:
    pad_index, unk_index, eos_index, sos_index = 0, 1, 2, 3
    ids_batch: list[list[int]] = []
    for smiles in smiles_batch:
        tokens = split_smiles(smiles).split()
        if len(tokens) > seq_len - 2:
            keep = (seq_len - 2) // 2
            tokens = tokens[:keep] + tokens[-keep:]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids + [eos_index]
        ids.extend([pad_index] * (seq_len - len(ids)))
        ids_batch.append(ids[:seq_len])
    return torch.tensor(ids_batch, dtype=torch.long)


def load_trfm_model(model_path: Path, vocab: WordVocab, device: torch.device) -> TrfmSeq2seq:
    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def extract_smiles_features(args, device: torch.device):
    smiles_tokenizer = MolTranBertTokenizer(str(args.bert_vocab))
    encoded = smiles_tokenizer.encode_plus(
        " ".join(list(args.smiles)),
        padding="max_length",
        truncation=True,
        max_length=args.smiles_token_length,
        return_tensors="pt",
    )
    smiles_tokens = encoded["input_ids"].long()

    trfm_vocab = WordVocab.load_vocab(args.trfm_vocab)
    trfm_model = load_trfm_model(args.trfm_model, trfm_vocab, device)
    xid = trfm_inputs([args.smiles], trfm_vocab, seq_len=args.trfm_seq_length).to(device)
    trfm_features = trfm_model.encode(xid.t()).detach().cpu().float()
    if trfm_features.shape != (1, 1024):
        raise RuntimeError(f"Expected TRFM features shape (1, 1024), got {tuple(trfm_features.shape)}")
    return smiles_tokens, trfm_features


def prepare_pst_imports(pst_root: str | None) -> None:
    if pst_root:
        root = Path(pst_root).expanduser().resolve()
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "scripts"))
    try:
        import example_dataset  # noqa: F401
        import pst.esm2  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import PST modules. Install PST or pass --pst-root /path/to/PST-main."
        ) from exc


def write_clean_pdb_for_pst(source_pdb: Path, output_pdb: Path, chain: str | None) -> None:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with open_text(source_pdb) as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"HEADER", "TITLE"}:
                lines.append(line.rstrip("\n"))
            elif record in {"ATOM", "TER"}:
                if chain and len(line) > 21 and line[21].strip() != chain:
                    continue
                lines.append(line.rstrip("\n"))
    if not any(line.startswith("ATOM") for line in lines):
        raise ValueError(f"No ATOM records found in {source_pdb} for chain={chain!r}")
    if not lines[0].startswith("HEADER"):
        lines.insert(0, "HEADER    PROTEIN                                 01-JAN-00   XXXX")
    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pst_model(model_name: str, checkpoint: Path, device: torch.device):
    from pst.esm2 import PST

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        model, model_cfg = PST.from_pretrained_url(model_name, checkpoint)
    except Exception:
        model, model_cfg = PST.from_pretrained_url(
            model_name,
            checkpoint,
            map_location=torch.device("cpu"),
        )
    return model.to(device).eval(), model_cfg


@torch.no_grad()
def compute_pst_representations(data_loader, model, device: torch.device, model_name: str, aggr: str | None):
    embeddings = []
    for data in data_loader:
        data = data.to(device)
        out = model(data, return_repr=True, aggr=aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        embeddings.extend(list(unbatch(out, batch)))
    return embeddings


def extract_pst_feature(args, device: torch.device, work_dir: Path) -> torch.Tensor:
    prepare_pst_imports(args.pst_root.strip() or None)
    from example_dataset import ExampleDataset

    dataset_root = work_dir / "pst_dataset"
    raw_pdb = dataset_root / "raw" / f"{args.protein_id}.pdb"
    write_clean_pdb_for_pst(args.pdb, raw_pdb, args.chain or None)
    model, _ = load_pst_model(args.pst_model, args.pst_checkpoint, device)
    dataset = ExampleDataset(root=str(dataset_root))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    aggr = None if args.pst_aggr == "none" else args.pst_aggr
    residue_embeddings = compute_pst_representations(loader, model, device, args.pst_model, aggr)
    if len(residue_embeddings) != 1:
        raise RuntimeError(f"Expected one PST embedding for {raw_pdb}, got {len(residue_embeddings)}")
    residue_repr = residue_embeddings[0].detach().float().cpu()
    pooled = residue_repr.mean(dim=0).view(1, -1)
    if pooled.shape != (1, 1280):
        raise RuntimeError(f"Expected PST features shape (1, 1280), got {tuple(pooled.shape)}")
    return pooled


def build_legacy_batch(graph: Data, smiles_tokens, t5_features, trfm_features, esm_features, pst_features, pair_id: str):
    graph_batch = Batch.from_data_list([graph])
    return {
        "node_features": graph_batch.x.float(),
        "coordinates": graph_batch.pos.float(),
        "edge_index": graph_batch.edge_index.long(),
        "edge_features": graph_batch.edge_attr.float(),
        "node_batch": graph_batch.batch.long(),
        "smiles_tokens": smiles_tokens.long(),
        "t5_features": t5_features.float().view(1, -1),
        "trfm_features": trfm_features.float().view(1, -1),
        "esm_features": esm_features.float().view(1, -1),
        "pst_features": pst_features.float().view(1, -1),
        "pair_id": [pair_id],
    }


@torch.no_grad()
def predict_one(checkpoint: Path, batch: dict, device: torch.device) -> float:
    model = load_legacy_checkpoint(checkpoint, device=device)
    moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    prediction = model(
        **{
            key: moved[key]
            for key in {
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
        }
    )
    return float(prediction.detach().cpu().view(-1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, help="Full protein PDB file.")
    parser.add_argument(
        "--pocket-pdb",
        type=Path,
        default=None,
        help="Optional pocket-only PDB. If omitted, --pdb is used for the graph.",
    )
    parser.add_argument("--chain", default="", help="Optional chain ID to retain from --pdb.")
    parser.add_argument("--sequence", default="", help="Protein sequence. If omitted, inferred from --pdb ATOM records.")
    parser.add_argument("--sequence-file", type=Path, default=None, help="FASTA/plain sequence file.")
    parser.add_argument("--protein-id", default="query_protein")
    parser.add_argument("--pair-id", default="query_pair")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--kcat-checkpoint", required=True, type=Path)
    parser.add_argument("--km-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-batch", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prott5-model", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--bert-vocab", type=Path, required=True)
    parser.add_argument("--trfm-vocab", type=Path, required=True)
    parser.add_argument("--trfm-model", type=Path, required=True)
    parser.add_argument("--smiles-token-length", type=int, default=500)
    parser.add_argument("--trfm-seq-length", type=int, default=220)
    parser.add_argument("--pst-root", default="")
    parser.add_argument("--pst-model", default="pst_t33_so")
    parser.add_argument("--pst-checkpoint", type=Path, required=True)
    parser.add_argument("--pst-aggr", choices=["none", "mean", "concat"], default="none")
    parser.add_argument("--graph-radius", type=float, default=10.0)
    parser.add_argument("--dssp-bin", default="", help="Optional mkdssp/DSSP binary. Empty zero-fills the last 9 graph node features.")
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--truncate-mode", choices=["head", "balanced"], default="head")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pdb is None:
        raise ValueError(
            "Sequence-only inference is not supported because EpiCataNexus needs a "
            "structure-derived pocket graph and PST features. Generate or provide a PDB "
            "structure first, then re-run with --pdb."
        )
    if not args.pdb.exists():
        raise FileNotFoundError(f"PDB file not found: {args.pdb}")
    if args.pocket_pdb is not None and not args.pocket_pdb.exists():
        raise FileNotFoundError(f"Pocket PDB file not found: {args.pocket_pdb}")
    required_files = {
        "kcat checkpoint": args.kcat_checkpoint,
        "Km checkpoint": args.km_checkpoint,
        "SMILES BERT vocab": args.bert_vocab,
        "TRFM vocab": args.trfm_vocab,
        "TRFM model": args.trfm_model,
        "PST checkpoint": args.pst_checkpoint,
    }
    for label, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Released legacy pooled inference normally requires CUDA because of the Mamba "
            "SMILES branch. Re-run with --device cuda on a CUDA host."
        )

    if args.sequence_file:
        sequence = read_sequence_file(args.sequence_file)
    elif args.sequence.strip():
        sequence = normalize_sequence(args.sequence)
    else:
        sequence = infer_sequence_from_pdb(args.pdb, args.chain or None)

    graph_source = args.pocket_pdb or args.pdb
    if args.pocket_pdb is None:
        print(
            "Warning: --pocket-pdb was not provided. Building the graph from the full --pdb. "
            "For manuscript-consistent inference, provide a PDB cropped to the fpocket-selected pocket residues.",
            file=sys.stderr,
        )
    dssp_bin = args.dssp_bin.strip() or None
    if dssp_bin:
        with tempfile.TemporaryDirectory() as graph_tmp:
            clean_graph_pdb = Path(graph_tmp) / f"{args.protein_id}_graph.pdb"
            write_clean_pdb_for_pst(graph_source, clean_graph_pdb, args.chain or None)
            graph = build_graph_from_pdb(clean_graph_pdb, None, args.graph_radius, dssp_bin)
    else:
        graph = build_graph_from_pdb(graph_source, args.chain or None, args.graph_radius, None)

    encoded_sequence = truncate_sequence(sequence, args.max_residues, args.truncate_mode)
    if len(encoded_sequence) != len(sequence):
        print(
            f"Warning: sequence length {len(sequence)} exceeds --max-residues {args.max_residues}; "
            f"using {args.truncate_mode} truncation for ProtT5/ESM-2 pooled features.",
            file=sys.stderr,
        )

    t5_tokenizer, t5_model, esm_tokenizer, esm_model = load_protein_language_models(args, device)
    t5_features = extract_t5_pooled(encoded_sequence, t5_tokenizer, t5_model, device)
    esm_features = extract_esm_pooled(encoded_sequence, esm_tokenizer, esm_model, device)
    smiles_tokens, trfm_features = extract_smiles_features(args, device)

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        pst_features = extract_pst_feature(args, device, args.work_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            pst_features = extract_pst_feature(args, device, Path(tmp))

    batch = build_legacy_batch(
        graph=graph,
        smiles_tokens=smiles_tokens,
        t5_features=t5_features,
        trfm_features=trfm_features,
        esm_features=esm_features,
        pst_features=pst_features,
        pair_id=args.pair_id,
    )
    if args.save_batch:
        args.save_batch.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"batches": [batch], "schema": "epicatanexus_legacy_pooled_prepared_batch_v1"}, args.save_batch)

    log10_kcat = predict_one(args.kcat_checkpoint, batch, device)
    log10_km = predict_one(args.km_checkpoint, batch, device)
    output = pd.DataFrame(
        [
            {
                "pair_id": args.pair_id,
                "protein_id": args.protein_id,
                "smiles": args.smiles,
                "sequence_length": len(sequence),
                "encoded_residues": len(encoded_sequence),
                "truncated": len(encoded_sequence) != len(sequence),
                "graph_source": str(graph_source),
                "dssp_status": getattr(graph, "dssp_status", ""),
                "log10_kcat": log10_kcat,
                "kcat": 10.0**log10_kcat,
                "log10_Km": log10_km,
                "Km": 10.0**log10_km,
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote kcat/Km prediction to {args.output}")


if __name__ == "__main__":
    main()
