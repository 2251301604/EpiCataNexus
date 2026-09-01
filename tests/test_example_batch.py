from epicatanexus.io import MODEL_INPUT_KEYS, validate_batch
from scripts.create_example_batch import create_example_batch


def test_synthetic_example_matches_prepared_batch_contract():
    batch = create_example_batch()
    validate_batch(batch, require_target=True)
    assert MODEL_INPUT_KEYS <= set(batch)
    assert batch["t5_states"].shape[:2] == batch["esm_states"].shape[:2]
