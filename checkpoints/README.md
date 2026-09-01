# Checkpoints

Model weights are intentionally not stored in Git. The planned initial external release
will contain only the pooled-feature `kcat` and `Km` neural state dictionaries.
Download URLs and immutable external revisions will be added after hosting is
finalized.

Expected files:

| Historical file | Task | SHA-256 |
|---|---|---|
| `best_pocket_mamba_1792_clean_sggn.pkl` | turnover-number prediction | `25a9e5d2f8097c16f4c56b2850a0c5ec8e5f8ae5ae27f482ca92fef93a897306` |
| `best_pocket_mamba_1792_km_sggn.pkl` | Michaelis-constant prediction | `4501bbeda51db54bd9dcded21934f79fc130329a79248c2f908e4dbb1da88aae` |

Task-specific XGBoost weights are not distributed. Use
`epicatanexus.legacy_pooled` rather than the residue-level canonical class to load these
files. Full details are in [`docs/WEIGHTS.md`](../docs/WEIGHTS.md).

