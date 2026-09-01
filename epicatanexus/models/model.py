"""End-to-end EpiCataNexus network matching the manuscript-level architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .modules import (
    GatedPocketReadout,
    PocketEGNNEncoder,
    ResidueCrossAttention,
    SMILESMambaEncoder,
    SubstrateGuidedGating,
)


@dataclass(frozen=True)
class EpiCataNexusConfig:
    node_dim: int = 51
    edge_dim: int = 92
    pocket_dim: int = 512
    sequence_dim: int = 512
    substrate_dim: int = 256
    pretrained_dim: int = 256
    t5_dim: int = 1024
    esm_dim: int = 1280
    trfm_dim: int = 1024
    pst_dim: int = 1280
    smiles_vocab_size: int = 600
    attention_heads: int = 8
    egnn_layers: int = 6
    sggn_temperature: float = 0.1
    dropout: float = 0.2


class EpiCataNexus(nn.Module):
    """Predict log10 kinetic values from a prepared enzyme-substrate batch."""

    def __init__(self, config: EpiCataNexusConfig | None = None) -> None:
        super().__init__()
        self.config = config or EpiCataNexusConfig()
        cfg = self.config
        self.sequence_encoder = ResidueCrossAttention(
            cfg.t5_dim, cfg.esm_dim, cfg.sequence_dim, cfg.attention_heads
        )
        self.pocket_encoder = PocketEGNNEncoder(
            cfg.node_dim, cfg.edge_dim, cfg.pocket_dim, cfg.egnn_layers
        )
        self.pocket_readout = GatedPocketReadout(cfg.pocket_dim)
        self.smiles_encoder = SMILESMambaEncoder(cfg.smiles_vocab_size, cfg.substrate_dim)
        self.trfm_projection = nn.Sequential(
            nn.Linear(cfg.trfm_dim, cfg.pretrained_dim), nn.ReLU()
        )
        self.pst_projection = nn.Sequential(nn.Linear(cfg.pst_dim, cfg.pretrained_dim), nn.ReLU())

        protein_dim = cfg.pocket_dim + cfg.sequence_dim + cfg.pretrained_dim
        substrate_dim = cfg.substrate_dim + cfg.pretrained_dim
        self.sggn = SubstrateGuidedGating(substrate_dim, protein_dim, cfg.sggn_temperature)
        joint_dim = protein_dim + substrate_dim
        self.regression_head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, 1024),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(1024, 1),
        )

    @classmethod
    def from_mapping(cls, values: dict) -> "EpiCataNexus":
        fields = EpiCataNexusConfig.__dataclass_fields__
        return cls(EpiCataNexusConfig(**{key: value for key, value in values.items() if key in fields}))

    def encode(self, **batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = batch["t5_states"].size(0)
        sequence = self.sequence_encoder(
            batch["t5_states"], batch["esm_states"], batch["sequence_mask"]
        )
        pocket_nodes = self.pocket_encoder(
            batch["node_features"],
            batch["coordinates"],
            batch["edge_index"],
            batch["edge_features"],
        )
        pocket = self.pocket_readout(pocket_nodes, batch["node_batch"], batch_size)
        smiles = self.smiles_encoder(batch["smiles_tokens"], batch["smiles_mask"])
        trfm = self.trfm_projection(batch["trfm_features"])
        pst = self.pst_projection(batch["pst_features"])
        protein = torch.cat([sequence, pocket, pst], dim=-1)
        substrate = torch.cat([smiles, trfm], dim=-1)
        joint, gates = self.sggn(substrate, protein)
        return joint, {
            "sequence": sequence,
            "pocket": pocket,
            "pst": pst,
            "smiles": smiles,
            "trfm": trfm,
            "sggn_gates": gates,
        }

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        joint, _ = self.encode(**batch)
        return self.regression_head(joint).squeeze(-1)

