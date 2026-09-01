"""Neural building blocks described in the EpiCataNexus manuscript."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


def segment_mean(values: torch.Tensor, batch: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    if batch_size is None:
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
    output = values.new_zeros((batch_size, values.size(-1)))
    counts = values.new_zeros((batch_size, 1))
    output.index_add_(0, batch, values)
    counts.index_add_(0, batch, values.new_ones((values.size(0), 1)))
    return output / counts.clamp_min(1.0)


class ResidueCrossAttention(nn.Module):
    """ProtT5 queries aligned ESM-2 keys/values before masked pooling."""

    def __init__(
        self,
        t5_dim: int = 1024,
        esm_dim: int = 1280,
        output_dim: int = 512,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(t5_dim, output_dim)
        self.key = nn.Linear(esm_dim, output_dim)
        self.value = nn.Linear(esm_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        t5_states: torch.Tensor,
        esm_states: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if t5_states.shape[:2] != esm_states.shape[:2]:
            raise ValueError("ProtT5 and ESM-2 states must be residue-aligned with equal [B, L].")
        query = self.query(t5_states)
        attended, weights = self.attention(
            query,
            self.key(esm_states),
            self.value(esm_states),
            key_padding_mask=~mask.bool(),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        states = self.norm(query + attended)
        pooled = masked_mean(states, mask.bool())
        return (pooled, weights) if return_attention else pooled


class PocketEGNNLayer(nn.Module):
    """Distance-based pocket message passing with E(n)-invariant node states."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index.long()
        squared_distance = ((coordinates[source] - coordinates[target]) ** 2).sum(-1, keepdim=True)
        messages = self.message(
            torch.cat(
                [node_states[target], node_states[source], squared_distance, edge_features], dim=-1
            )
        )
        aggregated = node_states.new_zeros(node_states.shape)
        counts = node_states.new_zeros((node_states.size(0), 1))
        aggregated.index_add_(0, target, messages)
        counts.index_add_(0, target, messages.new_ones((messages.size(0), 1)))
        aggregated = aggregated / counts.clamp_min(1.0)
        updated = self.update(torch.cat([node_states, aggregated], dim=-1))
        return self.norm(node_states + updated)


class PocketEGNNEncoder(nn.Module):
    def __init__(self, node_dim: int = 51, edge_dim: int = 92, hidden_dim: int = 512, layers: int = 6):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([PocketEGNNLayer(hidden_dim, edge_dim) for _ in range(layers)])

    def forward(self, nodes, coordinates, edge_index, edge_features):
        states = self.node_projection(torch.nan_to_num(nodes).clamp(-100.0, 100.0))
        edge_features = torch.nan_to_num(edge_features).clamp(-50.0, 50.0)
        for layer in self.layers:
            states = layer(states, coordinates, edge_index, edge_features)
        return states


class GatedPocketReadout(nn.Module):
    """Pocket-restricted mean readout followed by feature-wise self-gating."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_states: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        pooled = segment_mean(node_states, batch.long(), batch_size)
        return pooled * torch.sigmoid(self.gate(pooled))


class SMILESMambaEncoder(nn.Module):
    def __init__(self, vocab_size: int = 600, hidden_dim: int = 256) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - depends on CUDA-specific optional package
            raise RuntimeError(
                "The full substrate encoder requires mamba-ssm. Follow docs/INSTALL.md "
                "and install a build matching the local PyTorch/CUDA versions."
            ) from exc
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        states = self.embedding(tokens.long())
        states = states * mask.to(states.dtype).unsqueeze(-1)
        return masked_mean(self.mamba(states), mask.bool())


class SubstrateGuidedGating(nn.Module):
    def __init__(self, substrate_dim: int = 512, protein_dim: int = 1280, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.gate_generator = nn.Sequential(
            nn.Linear(substrate_dim, substrate_dim),
            nn.LayerNorm(substrate_dim),
            nn.SiLU(),
            nn.Linear(substrate_dim, protein_dim),
        )
        nn.init.constant_(self.gate_generator[-1].bias, 2.0)

    def forward(self, substrate: torch.Tensor, protein: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_generator(substrate) / self.temperature)
        return torch.cat([protein * gate, substrate], dim=-1), gate

