# Public-release checklist

## Manuscript-implementation alignment

- [x] ProtT5 and ESM-2 are retained at residue resolution, aligned by residue index,
  fused by asymmetric multi-head cross-attention, and pooled only after attention.
- [x] The pocket branch uses six distance-based EGNN layers followed by
  pocket-restricted mean readout and feature-wise self-gating.
- [x] SMILES-Mamba and TRFM form the substrate state; PST complements the protein
  state; SGGN gates the 1,280-dimensional protein state with the substrate state.
- [x] The canonical predictor is the neural MLP regression head optimized with MSE.
  No XGBoost estimator is part of the public manuscript implementation.
- [x] Keep historical pooled-feature checkpoints in the separate compatibility module;
  both current `kcat` and `Km` state dicts pass strict loading.
- [ ] Before claiming canonical-code metric reproduction, retrain `kcat` and `Km` from
  the fixed seed-3407 splits and regenerate every reported metric and ranking.
- [ ] Confirm or revise manuscript values when canonical checkpoints are available.
- [ ] Freeze exact configurations and checksums for every Fig. 5 ablation.
- [ ] Verify that all substrate-ranker train/validation/test candidate pools are leakage-safe.

## Artifacts

- [ ] Upload the pooled-feature `kcat` and `Km` neural checkpoints to Hugging Face.
- [x] Record the current local checkpoint SHA-256 values in `docs/WEIGHTS.md`.
- [ ] Add immutable Hugging Face revisions and converted `safetensors` hashes.
- [x] Exclude task-specific XGBoost and training-state files from the release.
- [ ] Publish redistributable split indices and minimal prepared example batches.
- [ ] Record fpocket 4.2.3 commands and binary checksum.
- [ ] Add task manifests for P25910, PETase, TEM-1, and substrate recovery.

## Later legal and publication metadata

- [ ] Select and approve the final code license; replace the pre-release notice.
- [ ] Confirm redistribution terms for BRENDA, SABIO-RK, PDB, AlphaFold, pretrained
  encoders, and derived feature stores.
- [ ] Confirm author order, corresponding authors, repository owner, and contact email.
- [ ] Add journal/preprint URL and DOI to `CITATION.cff` and `README.md`.
- [ ] Add funding acknowledgements if required by the target journal.

## Draft Data and Code Availability statement

> The source code, fixed evaluation splits, configuration files, and trained model
> weights for EpiCataNexus are available at [GitHub URL] and archived at [DOI]. Data
> derived from BRENDA, SABIO-RK, PDB, and AlphaFold are subject to the terms of their
> respective providers; scripts and identifiers required to reconstruct the processed
> datasets are provided in the repository. The exact software and data release version
> used in this study is [release tag].

Replace all bracketed fields before the final manuscript-linked release or archival DOI.
They may remain pending in the clearly labelled research-preview GitHub release.
