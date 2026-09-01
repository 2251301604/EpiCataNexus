# Result provenance

The compact public summaries transcribe the current manuscript tables, figures, and
associated final analysis summaries. They are labelled **manuscript-reported results**;
they are not yet claimed as reproduced by the unified public implementation.

| Result | Canonical manuscript source |
|---|---|
| kcat and Km benchmarks | Table 1 |
| sequence-identity generalization | Table 2 |
| pLDDT-stratified robustness | Table 3 |
| P25910, PETase, and TEM-1 | Fig. 2 and its underlying analysis tables |
| substrate recovery | Fig. 3 and its underlying analysis tables |
| activity/generalization panels | Fig. 4 |
| ablations | Fig. 5 |

The canonical public estimator is `epicatanexus.models.EpiCataNexus`: residue-aligned
cross-attention, pocket EGNN, SMILES-Mamba/TRFM, PST, SGGN, and an end-to-end MLP
regression head. The initial external weight release instead preserves the historical
pooled-feature `kcat` and `Km` neural checkpoints through the explicitly separate
`epicatanexus.legacy_pooled` compatibility path. Task-specific XGBoost weights are not
distributed.

The pooled checkpoints can be validated and used with their documented pooled input
contract, but they are not interchangeable with the canonical residue-level class.
The numerical tables remain labelled manuscript-reported results unless a benchmark is
regenerated from a named checkpoint, immutable inputs, and a recorded software version.
