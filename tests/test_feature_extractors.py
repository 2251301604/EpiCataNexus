import numpy as np
import pandas as pd
import pytest

from scripts.extract_pst_features import flatten_vector, load_feature_mapping


def test_load_pst_feature_table_from_tsv(tmp_path):
    path = tmp_path / "pst.tsv"
    frame = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "pst_0": [1.0, 3.0],
            "pst_1": [2.0, 4.0],
        }
    )
    frame.to_csv(path, sep="\t", index=False)
    mapping = load_feature_mapping(path, id_column="protein_id", prefix="pst_")
    assert sorted(mapping) == ["P1", "P2"]
    assert np.allclose(mapping["P1"], [1.0, 2.0])


def test_flatten_pst_vector_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="expected 3, got 2"):
        flatten_vector([1.0, 2.0], expected_dim=3, protein_id="P1")
