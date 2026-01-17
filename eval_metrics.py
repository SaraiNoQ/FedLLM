from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math
import re

def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0

def _lcs_len(a: List[str], b: List[str]) -> int:
    # DP LCS length (O(nm)) — ok for short texts.
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()
    if len(pred_toks) == 0 and len(ref_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    prec = lcs / max(len(pred_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def aggregate_metric(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)

def negative_transfer_score(metric_fed: float, metric_local: float, higher_is_better: bool) -> float:
    """Return signed difference (fed - local) in the 'higher is better' space."""
    if higher_is_better:
        return metric_fed - metric_local
    # for loss/ppl: smaller is better => convert to gain
    # gain = local - fed (positive means fed improved)
    return metric_local - metric_fed
