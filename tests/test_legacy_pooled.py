import torch

from epicatanexus.legacy_pooled.model import LegacyCrossAttentionFusion, LegacyEGNNEncoder


def test_legacy_pooled_attention_shape():
    module = LegacyCrossAttentionFusion(dim_t5=8, dim_esm=10, output_dim=12)
    output = module(torch.randn(2, 8), torch.randn(2, 10))
    assert output.shape == (2, 12)


def test_legacy_egnn_shape():
    module = LegacyEGNNEncoder(node_in_dim=4, edge_in_dim=3, hidden_dim=8, num_layers=2)
    output = module(
        torch.randn(4, 4),
        torch.randn(4, 3),
        torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]]),
        torch.randn(6, 3),
    )
    assert output.shape == (4, 8)
