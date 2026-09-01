import torch

from epicatanexus.models.modules import PocketEGNNEncoder, ResidueCrossAttention


def test_residue_cross_attention_shape():
    module = ResidueCrossAttention(t5_dim=8, esm_dim=10, output_dim=12, heads=3)
    output = module(
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 10),
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
    )
    assert output.shape == (2, 12)


def test_pocket_encoder_is_rotation_invariant():
    torch.manual_seed(7)
    module = PocketEGNNEncoder(node_dim=4, edge_dim=3, hidden_dim=8, layers=2).eval()
    nodes = torch.randn(4, 4)
    coordinates = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]])
    edges = torch.randn(edge_index.size(1), 3)
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    with torch.no_grad():
        first = module(nodes, coordinates, edge_index, edges)
        second = module(nodes, coordinates @ rotation, edge_index, edges)
    assert torch.allclose(first, second, atol=1e-5)

