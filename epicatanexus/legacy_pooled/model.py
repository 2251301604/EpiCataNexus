"""Legacy pooled-feature model used by the separately released kcat and Km checkpoints.

This compatibility module is intentionally separate from ``epicatanexus.models``.
The canonical manuscript implementation consumes residue-level ProtT5 and ESM-2
states, whereas these historical checkpoints consume one pooled vector per encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class LegacyPooledConfig:
    node_dim: int = 51
    edge_dim: int = 92
    pocket_dim: int = 512
    t5_dim: int = 1024
    esm_dim: int = 1280
    trfm_dim: int = 1024
    pst_dim: int = 1280
    hidden_dim: int = 256
    fusion_dim: int = 512
    smiles_vocab_size: int = 600
    egnn_layers: int = 6
    sggn_temperature: float = 0.1


class LegacyEGNNLayer(nn.Module):
    """Pure-PyTorch equivalent of the historical mean-aggregation EGNN layer."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1 + edge_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        states: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index.long()
        squared_distance = ((coordinates[target] - coordinates[source]) ** 2).sum(
            dim=-1, keepdim=True
        )
        squared_distance = squared_distance.clamp(max=100.0) + 1e-8
        messages = self.edge_mlp(
            torch.cat(
                [states[target], states[source], squared_distance, edge_features], dim=-1
            )
        )
        aggregated = states.new_zeros((states.size(0), messages.size(-1)))
        counts = states.new_zeros((states.size(0), 1))
        aggregated.index_add_(0, target, messages)
        counts.index_add_(0, target, messages.new_ones((messages.size(0), 1)))
        aggregated = aggregated / counts.clamp_min(1.0)
        return self.node_mlp(torch.cat([states, aggregated], dim=-1))


class LegacyEGNNEncoder(nn.Module):
    def __init__(
        self, node_in_dim: int, edge_in_dim: int, hidden_dim: int, num_layers: int = 6
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LegacyEGNNLayer(
                    node_in_dim if index == 0 else hidden_dim,
                    hidden_dim,
                    edge_in_dim,
                )
                for index in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, states, coordinates, edge_index, edge_features):
        for layer, norm in zip(self.layers, self.norms):
            updated = layer(states, coordinates, edge_index, edge_features)
            states = states + updated if states.shape == updated.shape else updated
            states = norm(states)
        return states


class LegacyCrossAttentionFusion(nn.Module):
    """Historical attention over one pooled ProtT5/ESM-2 token per protein."""

    def __init__(self, dim_t5: int = 1024, dim_esm: int = 1280, output_dim: int = 512):
        super().__init__()
        hidden_dim = 256
        self.q_proj = nn.Linear(dim_t5, hidden_dim)
        self.k_proj = nn.Linear(dim_esm, hidden_dim)
        self.v_proj = nn.Linear(dim_esm, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.SiLU()
        )

    def forward(self, t5_features: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        batch_size = t5_features.size(0)
        t5_features = torch.nan_to_num(t5_features).clamp(-50.0, 50.0)
        esm_features = torch.nan_to_num(esm_features).clamp(-50.0, 50.0)
        query = self.q_norm(self.q_proj(t5_features)).view(batch_size, 1, -1)
        key = self.k_norm(self.k_proj(esm_features)).view(batch_size, 1, -1)
        value = self.v_norm(self.v_proj(esm_features)).view(batch_size, 1, -1)
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        key = torch.nn.functional.normalize(key, p=2, dim=-1)
        attended, _ = self.attn(query, key, value)
        return self.out_proj(attended).view(batch_size, -1)


class LegacyAttentionalPool(nn.Module):
    """Graph-wise softmax pooling with historical ``gate_nn`` state-dict names."""

    def __init__(self, gate_nn: nn.Module) -> None:
        super().__init__()
        self.gate_nn = gate_nn

    def forward(
        self, node_states: torch.Tensor, node_batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        pooled = node_states.new_zeros((batch_size, node_states.size(-1)))
        logits = self.gate_nn(node_states).squeeze(-1)
        for graph_index in range(batch_size):
            selected = node_batch == graph_index
            if selected.any():
                weights = torch.softmax(logits[selected], dim=0).unsqueeze(-1)
                pooled[graph_index] = (node_states[selected] * weights).sum(dim=0)
        return pooled


class LegacySubstrateGuidedGating(nn.Module):
    def __init__(self, substrate_dim: int, protein_dim: int, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.gate_generator = nn.Sequential(
            nn.Linear(substrate_dim, substrate_dim),
            nn.LayerNorm(substrate_dim),
            nn.SiLU(),
            nn.Linear(substrate_dim, protein_dim),
        )
        nn.init.constant_(self.gate_generator[-1].bias, 2.0)

    def forward(self, substrate: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_generator(substrate) / self.temperature)
        return torch.cat([protein * gate, substrate], dim=-1)


class LegacyPooledEpiCataNexus(nn.Module):
    """Exact neural architecture of the released pooled-feature task checkpoints."""

    def __init__(self, config: LegacyPooledConfig | None = None) -> None:
        super().__init__()
        cfg = config or LegacyPooledConfig()
        self.config = cfg
        self.node_proj = nn.Sequential(
            nn.Linear(cfg.node_dim, cfg.pocket_dim),
            nn.SiLU(),
            nn.Linear(cfg.pocket_dim, cfg.pocket_dim),
            nn.LayerNorm(cfg.pocket_dim),
        )
        self.gnn_encoder = LegacyEGNNEncoder(
            cfg.pocket_dim, cfg.edge_dim, cfg.pocket_dim, cfg.egnn_layers
        )
        self.pocket_finder = nn.Sequential(
            nn.Linear(cfg.pocket_dim, cfg.pocket_dim // 2),
            nn.SiLU(),
            nn.Linear(cfg.pocket_dim // 2, 1),
        )
        self.pocket_pool = LegacyAttentionalPool(self.pocket_finder)
        self.fusion_engine = LegacyCrossAttentionFusion(
            cfg.t5_dim, cfg.esm_dim, cfg.fusion_dim
        )
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - optional CUDA dependency
            raise RuntimeError(
                "Legacy pooled-feature inference requires a CUDA-compatible mamba-ssm build."
            ) from exc
        self.smi_embedding = nn.Embedding(cfg.smiles_vocab_size, cfg.hidden_dim)
        self.smi_mamba = Mamba(
            d_model=cfg.hidden_dim, d_state=64, d_conv=4, expand=2
        )
        self.trfm_proj = nn.Sequential(nn.Linear(cfg.trfm_dim, cfg.hidden_dim), nn.ReLU())
        self.pst_proj = nn.Sequential(nn.Linear(cfg.pst_dim, cfg.hidden_dim), nn.ReLU())
        protein_dim = cfg.pocket_dim + cfg.fusion_dim + cfg.hidden_dim
        substrate_dim = cfg.hidden_dim * 2
        self.sggn = LegacySubstrateGuidedGating(
            substrate_dim, protein_dim, cfg.sggn_temperature
        )
        joint_dim = protein_dim + substrate_dim
        self.norm = nn.LayerNorm(joint_dim)
        self.fc = nn.Sequential(
            nn.Linear(joint_dim, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1)
        )

    def extract_features(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch: torch.Tensor,
        smiles_tokens: torch.Tensor,
        t5_features: torch.Tensor,
        trfm_features: torch.Tensor,
        esm_features: torch.Tensor,
        pst_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = t5_features.size(0)
        node_states = self.node_proj(torch.nan_to_num(node_features).clamp(-100.0, 100.0))
        node_states = self.gnn_encoder(
            node_states,
            coordinates,
            edge_index,
            torch.nan_to_num(edge_features).clamp(-50.0, 50.0),
        )
        pocket = self.pocket_pool(
            torch.nan_to_num(node_states).clamp(-100.0, 100.0),
            node_batch.long(),
            batch_size,
        )
        sequence = self.fusion_engine(t5_features, esm_features)
        smiles = self.smi_mamba(self.smi_embedding(smiles_tokens.long())).mean(dim=1)
        trfm = self.trfm_proj(torch.nan_to_num(trfm_features).clamp(-50.0, 50.0))
        pst = self.pst_proj(torch.nan_to_num(pst_features).clamp(-50.0, 50.0))
        protein = torch.cat([pocket, sequence, pst], dim=-1)
        substrate = torch.cat([smiles, trfm], dim=-1)
        return torch.nan_to_num(self.sggn(substrate, protein))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(**inputs)
        return self.fc(self.norm(features)).squeeze(-1)


def load_legacy_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LegacyPooledEpiCataNexus:
    """Load a trusted pooled-feature state dict with strict key validation."""

    path = Path(checkpoint)
    model = LegacyPooledEpiCataNexus()
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install safetensors to load .safetensors checkpoints.") from exc
        state_dict = load_file(str(path), device=str(device))
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
