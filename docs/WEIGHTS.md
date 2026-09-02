# Model weights

Model files are hosted outside GitHub. The initial pooled-feature neural checkpoints
are hosted on Hugging Face at
[`nnnnnnnnnnnn1111/EpiCataNexus`](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus).
The immutable release revision is
`7c78581fd5150a3bd60b91d158daafa8a7590133`. No checkpoint is stored in this source
repository.

## Initial release scope

Only the following pooled-feature neural checkpoints are included in the initial
external release:

| Task | Released file | Historical source filename | Parameters | SHA-256 | URL |
|---|---|---|---:|---|---|
| `kcat` | `epicatanexus_kcat_pooled.safetensors` | `best_pocket_mamba_1792_clean_sggn.pkl` | 15,507,458 unique parameters | `80e26098a3b8cbdd4a254bb8cb73357db710c9494fad9140f0607d812c925324` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_kcat_pooled.safetensors) |
| `Km` | `epicatanexus_km_pooled.safetensors` | `best_pocket_mamba_1792_km_sggn.pkl` | 15,507,458 unique parameters | `c9bafd221484ab56bad1a603132351b5f0582d2e1c6fb4b502d9f8fa860d50ef` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_km_pooled.safetensors) |

The hashes above identify the released `safetensors` artifacts. The historical `.pkl`
filenames are listed only to preserve the mapping to the original local training
outputs.

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
