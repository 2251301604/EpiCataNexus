# Engineering-oriented workflows

## Variant ranking

After predicting wild type and candidate variants on the same substrate:

```bash
python scripts/engineering/rank_variants.py \
  --input outputs/variant_predictions.csv \
  --output outputs/variant_ranking.csv \
  --wild-type WT
```

The script reports `delta_log10 = prediction_variant - prediction_WT` and the relative
fold change `10 ** delta_log10`.

## Substrate ranking

Score every candidate substrate for each fixed enzyme and run:

```bash
python scripts/engineering/rank_substrates.py \
  --input outputs/substrate_scores.csv \
  --output outputs/substrate_ranking.csv
```

The input must contain `query_id` and `score`. Ranking is performed separately within
each enzyme query to prevent cross-query score mixing.

## Double-mutant epistasis

Prepare a CSV with `double_score`, `single_a_score`, `single_b_score`, and
`wild_type_score`, all on the same log10 scale:

```bash
python scripts/engineering/score_epistasis.py \
  --input outputs/double_mutant_components.csv \
  --output outputs/double_mutant_epistasis.csv
```

The reported quantity is

```text
epsilon = score_AB - score_A - score_B + score_WT
```

Ranking utilities prioritize experiments; they do not establish biochemical activity or
causality without experimental validation.

