# Data and preprocessing

## Required record fields

The normalized TSV manifest uses the following public schema:

| Column | Meaning |
|---|---|
| `protein_id` | stable protein identifier |
| `sequence` | amino-acid sequence |
| `smiles` | canonical substrate SMILES |
| `label` | `log10(kcat)` or `log10(Km)` target |
| `structure_path` | optional local PDB/mmCIF path |
| `ec_number` | optional EC annotation used only by explicitly configured analyses |

The source benchmark integrates measurements curated from BRENDA and SABIO-RK and
structures from PDB or AlphaFold. Their licenses and database terms govern
redistribution. This repository therefore does not include the original full table until
the authors confirm a redistributable artifact.

## Candidate-pocket construction

1. Remove non-protein atoms from the structure.
2. Run fpocket 4.2.3 on protein-only geometry.
3. Retain the highest-scoring cavity.
4. Select residues with at least one heavy atom within 6 Å of a retained alpha sphere.
5. Use residue Cα coordinates as node positions.
6. Connect pocket residues whose Cα distance is at most 10 Å.

No substrate identity, ligand coordinate, catalytic annotation, or kinetic label is used
to select the candidate pocket. Records sharing a protein use the same pocket.

## Residue-level sequence features

ProtT5 and ESM-2 must be extracted at residue resolution; pooled protein vectors are
not valid inputs to the manuscript implementation. Generate aligned states with:

    python scripts/extract_features.py --input data/processed/kcat_manifest.tsv \
      --output-dir data/features/protein_residue_states --device cuda

For each sequence the script stores ProtT5 states with shape `[L, 1024]`, ESM-2 states
with shape `[L, 1280]`, and a shared residue mask. It removes model-specific special
tokens and aborts if the two representations cannot be aligned to the same residue
indices.

## Legacy pooled-feature inputs

The separately released `kcat` and `Km` checkpoints predate the canonical residue-level
interface. They expect one pooled ProtT5 vector `[B, 1024]` and one pooled ESM-2 vector
`[B, 1280]` per protein, together with pooled TRFM `[B, 1024]`, pooled PST `[B, 1280]`,
SMILES tokens, and the pocket graph. Load these files only with
`epicatanexus.legacy_pooled`; they cannot be substituted for the residue-level tensors
in the prepared-batch contract below. See [WEIGHTS.md](WEIGHTS.md).

Generate a synthetic contract example without research data:

```bash
python scripts/create_example_batch.py --output outputs/example_batches.pt
```

## Prepared-batch contract

Training and inference scripts consume a `torch.save` payload containing either a list
of batch dictionaries or `{\"batches\": [...]}`. Each batch contains:

| Key | Shape | Description |
|---|---|---|
| `node_features` | `[N, 51]` | pocket residue features |
| `coordinates` | `[N, 3]` | Cα coordinates |
| `edge_index` | `[2, E]` | directed graph edges |
| `edge_features` | `[E, 92]` | distance, sequence, and direction features |
| `node_batch` | `[N]` | graph membership |
| `t5_states` | `[B, L, 1024]` | residue-level ProtT5 states |
| `esm_states` | `[B, L, 1280]` | residue-aligned ESM-2 states |
| `sequence_mask` | `[B, L]` | valid residues |
| `smiles_tokens` | `[B, S]` | tokenized SMILES |
| `smiles_mask` | `[B, S]` | valid SMILES tokens |
| `trfm_features` | `[B, 1024]` | pretrained SMILES Transformer features |
| `pst_features` | `[B, 1280]` | pretrained structural features |
| `target` | `[B]` | required for training/evaluation |
| `pair_id` | length `B` | optional stable output identifier |

Never load prepared PyTorch files from an untrusted source; pickle-compatible formats
can execute code during deserialization.

## Nested split

The paper uses seed 3407. For kcat, 11,869 records are split into 10,682 development
and 1,187 outer-test rows. The development set is then split into 9,613 inner-training
and 1,069 inner-validation rows. The outer test set must remain untouched until model
and checkpoint selection are complete.

Sequence-identity, pair-level, and substrate-cluster holdouts are distinct evaluations;
they are not derived from the nested random split.

