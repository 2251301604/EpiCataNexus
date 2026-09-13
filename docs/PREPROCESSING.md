# Raw-to-PT preprocessing

This document describes how raw enzyme-substrate records are converted into the
prepared `.pt` tensor batches consumed by EpiCataNexus.

## Current release status

The repository currently provides the public tensor contract, split generation, a
synthetic batch generator, and residue-level ProtT5/ESM-2 extraction. It does not yet
provide a complete one-command raw-to-`.pt` preprocessing pipeline.

| Component | Current status |
|---|---|
| Input manifest validation and train/valid/test splitting | Provided by `scripts/prepare_data.py` |
| Residue-level ProtT5 and ESM-2 extraction | Provided by `scripts/extract_features.py` |
| Prepared-batch schema example | Provided by `scripts/create_example_batch.py` |
| Structure retrieval and cleaning wrapper | Not included in the current release |
| fpocket execution wrapper and pocket-residue table export | Not included in the current release |
| Pocket graph featurizer for the exact `[N, 51]` node and `[E, 92]` edge matrices | Not included in the current release |
| PST feature extraction | Not included in the current release |
| TRFM/SMILES Transformer feature extraction | Not included in the current release |
| Full feature merge into model-ready `.pt` batches | Not included in the current release |

Accordingly, users who want to run the current training, evaluation, or prediction
scripts must provide prepared tensor batches that follow the schema in
[DATA.md](DATA.md). Raw sequence/SMILES/structure-to-prediction inference should not be
described as a one-command workflow for this release.

## Five-step preprocessing workflow

### 1. Build the normalized manifest

Start from a TSV table containing one enzyme-substrate record per row:

| Column | Meaning |
|---|---|
| `protein_id` | stable protein identifier |
| `sequence` | amino-acid sequence |
| `smiles` | canonical substrate SMILES |
| `label` | `log10(kcat)` or `log10(Km)` target; optional for prediction-only batches |
| `structure_path` | local PDB/mmCIF path, or a resolvable structure identifier |

For paper-style splitting, run:

```bash
python scripts/prepare_data.py \
  --input data/processed/kcat_manifest.tsv \
  --output-dir data/splits/kcat \
  --seed 3407
```

`scripts/prepare_data.py` validates required columns and writes split TSV files. It
does not create model-ready `.pt` tensors.

### 2. Prepare structures and candidate pockets

For each protein, obtain a structure from a licensed local PDB/mmCIF file, PDB, or
AlphaFold source. The manuscript preprocessing uses a protein-only structure before
pocket prediction:

1. remove non-protein atoms;
2. run fpocket 4.2.3 on the cleaned structure;
3. retain the highest-scoring predicted cavity;
4. select residues with at least one heavy atom within 6 Å of a retained alpha sphere;
5. use residue C-alpha coordinates as node positions;
6. connect pocket residues whose C-alpha distance is at most 10 Å.

This step must produce a per-protein pocket graph. The current public repository
documents the expected graph tensors but does not yet include the exact structure
cleaning, fpocket wrapper, or graph featurizer used to create the manuscript feature
store.

### 3. Generate sequence and structure features

The canonical manuscript implementation expects residue-level ProtT5 and ESM-2 states:

```bash
python scripts/extract_features.py \
  --input data/processed/kcat_manifest.tsv \
  --output-dir data/features/protein_residue_states \
  --device cuda
```

This writes one tensor file per unique sequence containing:

| Key | Shape |
|---|---|
| `t5_states` | `[L, 1024]` |
| `esm_states` | `[L, 1280]` |
| `sequence_mask` | `[L]` |

The model also expects `pst_features` with shape `[B, 1280]`. In the current release,
PST is treated as an externally precomputed structural feature. The exact PST model
checkpoint, package version, pooling rule, and extraction script used for the
manuscript experiments are not included yet and should be recorded before claiming
full raw-data reproducibility.

### 4. Generate substrate features

The prepared batch requires both tokenized SMILES and pretrained TRFM features:

| Feature | Shape | Notes |
|---|---|---|
| `smiles_tokens` | `[B, S]` | integer token IDs for the SMILES-Mamba branch |
| `smiles_mask` | `[B, S]` | valid-token mask |
| `trfm_features` | `[B, 1024]` | pooled pretrained SMILES Transformer feature |

The current release documents the required dimensions but does not include the exact
TRFM/SMILES Transformer checkpoint, tokenizer/vocabulary export, pooling rule, or
feature extraction script used for the manuscript experiments.

### 5. Merge features into prepared `.pt` batches

Training, evaluation, and prediction scripts consume a `torch.save` payload containing
either a list of batch dictionaries or `{"batches": [...]}`. Each batch must contain:

| Key | Canonical shape |
|---|---|
| `node_features` | `[N, 51]` |
| `coordinates` | `[N, 3]` |
| `edge_index` | `[2, E]` |
| `edge_features` | `[E, 92]` |
| `node_batch` | `[N]` |
| `t5_states` | `[B, L, 1024]` |
| `esm_states` | `[B, L, 1280]` |
| `sequence_mask` | `[B, L]` |
| `smiles_tokens` | `[B, S]` |
| `smiles_mask` | `[B, S]` |
| `trfm_features` | `[B, 1024]` |
| `pst_features` | `[B, 1280]` |
| `target` | `[B]`, required for training/evaluation |
| `pair_id` | length `B`, optional but recommended |

Use the synthetic generator to inspect the schema without using manuscript data:

```bash
python scripts/create_example_batch.py --output outputs/example_batches.pt
```

The future raw-to-`.pt` merger should be deterministic and should record the manifest
revision, structure source, fpocket version, pretrained encoder versions, tokenizer
versions, and hashes of all generated feature stores.

## Legacy pooled-feature checkpoints

The Hugging Face checkpoints released with this repository use the legacy pooled-feature
interface. They are not compatible with the canonical residue-level `.pt` schema above.
For those checkpoints, the prepared batches must instead contain pooled protein
features:

| Key | Legacy pooled shape |
|---|---|
| `t5_features` | `[B, 1024]` |
| `esm_features` | `[B, 1280]` |
| `trfm_features` | `[B, 1024]` |
| `pst_features` | `[B, 1280]` |

The pocket graph and SMILES-token inputs are still required. See [WEIGHTS.md](WEIGHTS.md)
for the full legacy checkpoint input contract.

## What must be added for full raw-data reproducibility

Before the repository can claim complete raw-to-`.pt` preprocessing, it should include
or precisely reference:

1. structure retrieval and cleaning commands;
2. fpocket 4.2.3 command lines and binary checksum;
3. the pocket graph featurizer that creates the exact 51-dimensional node features and
   92-dimensional edge features;
4. PST model name, checkpoint, package version, pooling rule, and extraction script;
5. TRFM/SMILES Transformer model name, checkpoint, tokenizer/vocabulary, pooling rule,
   and extraction script;
6. a deterministic batch builder that merges the manifest, graph tensors, sequence
   features, substrate features, PST features, and targets into `.pt` files.
