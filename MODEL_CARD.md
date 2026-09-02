# EpiCataNexus Model Card

## Model summary

EpiCataNexus is a substrate-guided, pocket-aware neural framework for predicting
`log10(kcat)` and `log10(Km)` and supporting enzyme-engineering ranking analyses.
The repository contains two explicitly separated model paths:

1. the canonical manuscript implementation in `epicatanexus.models`, which uses
   residue-aligned ProtT5 and ESM-2 states; and
2. the compatibility implementation in `epicatanexus.legacy_pooled`, which loads the
   separately released pooled-feature `kcat` and `Km` neural checkpoints.

Task-specific XGBoost weights are not distributed.

## Intended use

The software and checkpoints are intended for research in enzyme kinetic prediction,
comparative enzyme–substrate scoring, mutation prioritization, and computational
hypothesis generation followed by experimental validation.

## Out-of-scope use

EpiCataNexus is not a clinical, diagnostic, environmental-safety, or autonomous
biomanufacturing decision system. Predictions should not replace kinetic assays,
structural validation, or expert review.

## Inputs and outputs

The canonical path accepts candidate-pocket graphs, residue-level ProtT5/ESM-2 states,
SMILES tokens, TRFM features, and PST features. The legacy path accepts the same graph,
SMILES, TRFM, and PST modalities but uses one pooled ProtT5 vector and one pooled ESM-2
vector per protein. Both paths output a scalar kinetic prediction in the configured
`log10` target space.

Exact tensor shapes are documented in [docs/DATA.md](docs/DATA.md) and
[docs/WEIGHTS.md](docs/WEIGHTS.md).

## Released checkpoints

The initial external release is limited to the two pooled-feature neural checkpoints:

| Task | Released file | SHA-256 | Public location |
|---|---|---|---|
| `kcat` | `epicatanexus_kcat_pooled.safetensors` | `80e26098a3b8cbdd4a254bb8cb73357db710c9494fad9140f0607d812c925324` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_kcat_pooled.safetensors) |
| `Km` | `epicatanexus_km_pooled.safetensors` | `c9bafd221484ab56bad1a603132351b5f0582d2e1c6fb4b502d9f8fa860d50ef` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_km_pooled.safetensors) |

Both files contain the neural state dict for the pocket EGNN, pooled sequence fusion,
SMILES-Mamba, TRFM/PST projections, SGGN, and the regression head. Optimizer states,
feature caches, source datasets, and task-specific XGBoost models are not released.

## Evaluation

Numerical values displayed in the repository are results reported in the accompanying
manuscript. They are not represented as a newly regenerated benchmark run from the
canonical residue-level public implementation. Result provenance is documented in
[docs/RESULT_PROVENANCE.md](docs/RESULT_PROVENANCE.md).

## Limitations

- Kinetic databases contain assay heterogeneity, missing metadata, and organism and
  enzyme-family biases.
- Predicted pockets and static structures do not capture all conformational states.
- The legacy checkpoints depend on pretrained pooled features generated with the
  original feature pipeline and are not interchangeable with residue-level tensors.
- Rankings are hypotheses for experimental follow-up, not guarantees of improved
  catalytic performance.

## Recommended reporting

Report the model path (`canonical-residue` or `legacy-pooled`), task, checkpoint hash,
dataset/split revision, feature-extractor versions, fpocket version, random seed, target
transformation, and all evaluation metrics used.

## License and citation

The manuscript is not yet formally published and the final software and checkpoint
licenses have not yet been selected. The current pre-release notice in [LICENSE](LICENSE)
applies until it is replaced. Citation metadata is maintained in
[CITATION.cff](CITATION.cff).
