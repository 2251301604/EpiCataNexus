# Model weights

Model files are hosted outside GitHub. Public URLs and immutable revisions will be
added after the external release is created; no checkpoint is stored in this source
repository.

## Initial release scope

Only the following pooled-feature neural checkpoints are planned for the initial
external release:

| Task | Historical filename | Parameters | SHA-256 |
|---|---|---:|---|
| `kcat` | `best_pocket_mamba_1792_clean_sggn.pkl` | 15,507,458 unique parameters | `25a9e5d2f8097c16f4c56b2850a0c5ec8e5f8ae5ae27f482ca92fef93a897306` |
| `Km` | `best_pocket_mamba_1792_km_sggn.pkl` | 15,507,458 unique parameters | `4501bbeda51db54bd9dcded21934f79fc130329a79248c2f908e4dbb1da88aae` |

The hashes above identify the current local `.pkl` state dictionaries. If files are
converted to `safetensors`, new hashes must be recorded for the converted artifacts.

Task-specific XGBoost, ExtraTrees, ridge, optimizer, and mixed-precision scaler states
are outside the release scope.

## Compatibility path

Use `epicatanexus.legacy_pooled.LegacyPooledEpiCataNexus` for these checkpoints. The
compatibility model expects:

| Input | Shape |
|---|---|
| `node_features` | `[N, 51]` |
| `coordinates` | `[N, 3]` |
| `edge_index` | `[2, E]` |
| `edge_features` | `[E, 92]` |
| `node_batch` | `[N]` |
| `smiles_tokens` | `[B, S]` |
| `t5_features` | `[B, 1024]` pooled |
| `trfm_features` | `[B, 1024]` pooled |
| `esm_features` | `[B, 1280]` pooled |
| `pst_features` | `[B, 1280]` pooled |

Strictly validate a trusted file before inference:

```bash
python scripts/verify_legacy_checkpoint.py /path/to/checkpoint.pkl
```

For safer distribution, convert each trusted tensor-only state dictionary and validate
the converted file:

```bash
python scripts/convert_legacy_checkpoint.py checkpoint.pkl model.safetensors
```

Never load an untrusted pickle-compatible checkpoint.

## Canonical residue-level path

`epicatanexus.models.EpiCataNexus` is the manuscript-aligned public architecture and
expects `[B, L, 1024]` ProtT5 states and `[B, L, 1280]` ESM-2 states plus an aligned
mask. Pooled checkpoints cannot be loaded into this class. Canonical residue-level
checkpoints are not part of the initial external weight release.
