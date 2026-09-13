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

## Auxiliary assets for PDB + SMILES inference

The single-query PDB + SMILES helper also needs preprocessing assets that are too large
or too binary-specific for GitHub. These files are hosted in the same Hugging Face
model repository at immutable revision
`3b83df8700f5ab2d17c704f78e3e06cd2fcd0921`:

| File | Role | SHA-256 | URL |
|---|---|---|---|
| `Model/model.pt` | PST `pst_t33_so` structure-only checkpoint | `5a54f878cfb4429bdbbb6a1ea6c9f12570015089fdb1f821e93bbd40ef1c921c` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/3b83df8700f5ab2d17c704f78e3e06cd2fcd0921/Model/model.pt) |
| `Model/trfm_12_23000.pkl` | pretrained TRFM/SMILES Transformer state dict | `6b56c8c05d048e7c7d143c4e3ba2bc6f76e5eda2358798cf636210406a700eb2` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/3b83df8700f5ab2d17c704f78e3e06cd2fcd0921/Model/trfm_12_23000.pkl) |
| `Model/vocab.pkl` | TRFM token vocabulary | `21a66c850a3222547ec0fbd30c05fe587d66d22d3de2ee2195c58250fe486fb7` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/3b83df8700f5ab2d17c704f78e3e06cd2fcd0921/Model/vocab.pkl) |
| `Model/bert_vocab.txt` | SMILES-Mamba tokenizer vocabulary | `2d03157ab523544f4c6216c0491da22e46c27f41be8df73b758fbc19f5767c70` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/3b83df8700f5ab2d17c704f78e3e06cd2fcd0921/Model/bert_vocab.txt) |

The command-line interface takes explicit local paths to these files, so users can
store them under any local directory after downloading them from Hugging Face.

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
python scripts/verify_legacy_checkpoint.py epicatanexus_kcat_pooled.safetensors
```

Run predictions with the legacy pooled compatibility script:

```bash
python scripts/predict_legacy_pooled.py \
  --checkpoint epicatanexus_kcat_pooled.safetensors \
  --batches data/features/legacy_pooled_new_pairs.pt \
  --output outputs/legacy_pooled_predictions.csv \
  --device cuda
```

For single PDB + SMILES queries, use the raw-feature helper:

```bash
python scripts/infer_pdb_smiles_kcat_km.py \
  --pdb examples/query_protein.pdb \
  --pocket-pdb examples/query_fpocket_top_pocket.pdb \
  --smiles "CC(=O)O" \
  --kcat-checkpoint epicatanexus_kcat_pooled.safetensors \
  --km-checkpoint epicatanexus_km_pooled.safetensors \
  --bert-vocab Model/bert_vocab.txt \
  --trfm-vocab Model/vocab.pkl \
  --trfm-model Model/trfm_12_23000.pkl \
  --pst-root /path/to/PST-main \
  --pst-checkpoint Model/model.pt \
  --output outputs/query_kcat_km.csv \
  --device cuda
```

This helper is PDB-first. If only a protein sequence is available, generate or provide
a PDB structure first; the released pooled checkpoints require a structure-derived
graph and PST features. Supplying `--pocket-pdb` is recommended for manuscript-aligned
inference. Without it, the script builds the graph from the full PDB and prints a
warning.

Do not pass these `.safetensors` files to `scripts/predict.py`; that script expects a
canonical residue-level `.pt` checkpoint containing `model_config` and `model_state`.

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
