import pytest
import torch

from scripts.predict_legacy_pooled import LEGACY_INPUT_KEYS, load_legacy_batches


def _legacy_batch():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return {
        "node_features": torch.randn(2, 51),
        "coordinates": torch.randn(2, 3),
        "edge_index": edge_index,
        "edge_features": torch.randn(edge_index.size(1), 92),
        "node_batch": torch.zeros(2, dtype=torch.long),
        "smiles_tokens": torch.randint(0, 600, (1, 8)),
        "t5_features": torch.randn(1, 1024),
        "trfm_features": torch.randn(1, 1024),
        "esm_features": torch.randn(1, 1280),
        "pst_features": torch.randn(1, 1280),
        "pair_id": ["example"],
    }


def test_load_legacy_batches_accepts_valid_schema(tmp_path):
    path = tmp_path / "legacy_batches.pt"
    torch.save({"batches": [_legacy_batch()]}, path)
    batches = load_legacy_batches(path)
    assert len(batches) == 1
    assert LEGACY_INPUT_KEYS <= set(batches[0])


def test_load_legacy_batches_rejects_missing_key(tmp_path):
    batch = _legacy_batch()
    batch.pop("t5_features")
    path = tmp_path / "legacy_batches.pt"
    torch.save([batch], path)
    with pytest.raises(KeyError, match="t5_features"):
        load_legacy_batches(path)
