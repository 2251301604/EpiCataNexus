"""Regression and ranking metrics used by the release scripts."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    finite = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[finite], pred[finite]
    if true.size == 0:
        raise ValueError("No finite observations were supplied.")

    residual = true - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    pcc = float("nan")
    if true.size > 1 and np.std(true) > 0 and np.std(pred) > 0:
        pcc = float(np.corrcoef(true, pred)[0, 1])
    return {
        "n": int(true.size),
        "r2": r2,
        "rmse": math.sqrt(float(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "pcc": pcc,
    }


def hit_at_k(relevance: Iterable[int | bool], k: int) -> float:
    values = np.asarray(list(relevance), dtype=bool)
    if k <= 0:
        raise ValueError("k must be positive")
    return float(values[:k].any())


def ndcg_at_k(relevance: Iterable[float], k: int) -> float:
    values = np.asarray(list(relevance), dtype=float)[:k]
    if values.size == 0:
        return float("nan")
    discounts = np.log2(np.arange(2, values.size + 2))
    dcg = float(np.sum((2.0**values - 1.0) / discounts))
    ideal = np.sort(values)[::-1]
    idcg = float(np.sum((2.0**ideal - 1.0) / discounts))
    return 0.0 if idcg == 0 else dcg / idcg

