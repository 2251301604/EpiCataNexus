# Checkpoints

Model weights are intentionally not stored in Git. The initial external release
contains only the pooled-feature `kcat` and `Km` neural state dictionaries hosted on
Hugging Face at revision `7c78581fd5150a3bd60b91d158daafa8a7590133`.

Expected files:

| Released file | Historical source file | Task | SHA-256 | URL |
|---|---|---|---|---|
| `epicatanexus_kcat_pooled.safetensors` | `best_pocket_mamba_1792_clean_sggn.pkl` | turnover-number prediction | `80e26098a3b8cbdd4a254bb8cb73357db710c9494fad9140f0607d812c925324` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_kcat_pooled.safetensors) |
| `epicatanexus_km_pooled.safetensors` | `best_pocket_mamba_1792_km_sggn.pkl` | Michaelis-constant prediction | `c9bafd221484ab56bad1a603132351b5f0582d2e1c6fb4b502d9f8fa860d50ef` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_km_pooled.safetensors) |

Task-specific XGBoost weights are not distributed. Use
`epicatanexus.legacy_pooled` rather than the residue-level canonical class to load these
files. Full details are in [`docs/WEIGHTS.md`](../docs/WEIGHTS.md).

