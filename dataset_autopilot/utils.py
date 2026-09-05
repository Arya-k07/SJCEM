"""
Utility functions for safe JSON serialization, statistical metrics, and data helpers.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def safe_float(val: Any) -> Optional[float]:
    """Convert any numerical value to a JSON-safe float, converting NaN/Inf to None or 0."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def safe_int(val: Any) -> Optional[int]:
    """Convert any numerical value to a JSON-safe integer."""
    if val is None:
        return None
    try:
        if isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val)):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def clean_json_dict(obj: Any) -> Any:
    """Recursively convert NumPy/Pandas objects, NaNs, and timestamps to standard Python types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return safe_float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): clean_json_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
        return [clean_json_dict(item) for item in obj]
    return str(obj)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Population Stability Index (PSI) between baseline (expected) and current (actual).
    Returns (psi_value, bin_details).
    """
    # Filter out NaNs
    exp_clean = expected[~pd.isna(expected)]
    act_clean = actual[~pd.isna(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0, []

    # Check if numeric
    try:
        exp_clean = np.asarray(exp_clean, dtype=float)
        act_clean = np.asarray(act_clean, dtype=float)
    except (ValueError, TypeError):
        # Categorical PSI
        return calculate_categorical_psi(exp_clean, act_clean, epsilon=epsilon)

    # Determine quantiles on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_clean, percentiles)
    bin_edges = np.unique(bin_edges)  # handle duplicate edges

    if len(bin_edges) < 2:
        return 0.0, []

    min_val = float(min(exp_clean.min(), act_clean.min()))
    max_val = float(max(exp_clean.max(), act_clean.max()))
    bin_edges[0] = min_val - 1e-5
    bin_edges[-1] = max_val + 1e-5

    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Avoid zero division
    exp_pct = np.clip(exp_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    # Renormalize
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_per_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    total_psi = float(np.sum(psi_per_bin))

    bin_details = []
    for i in range(len(exp_counts)):
        low = "-inf" if np.isneginf(bin_edges[i]) else round(float(bin_edges[i]), 2)
        high = "+inf" if np.isposinf(bin_edges[i + 1]) else round(float(bin_edges[i + 1]), 2)
        bin_details.append({
            "bin_label": f"[{low}, {high})",
            "expected_pct": round(float(exp_pct[i]), 4),
            "actual_pct": round(float(act_pct[i]), 4),
            "psi_contribution": round(float(psi_per_bin[i]), 4)
        })

    return round(total_psi, 4), bin_details


def calculate_categorical_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    epsilon: float = 1e-4
) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculate PSI for categorical features."""
    exp_series = pd.Series(expected).astype(str)
    act_series = pd.Series(actual).astype(str)

    all_cats = list(set(exp_series.unique()).union(set(act_series.unique())))

    exp_counts = exp_series.value_counts(normalize=True)
    act_counts = act_series.value_counts(normalize=True)

    total_psi = 0.0
    bin_details = []

    for cat in all_cats:
        e = exp_counts.get(cat, epsilon)
        a = act_counts.get(cat, epsilon)
        e = max(e, epsilon)
        a = max(a, epsilon)
        psi_i = (a - e) * np.log(a / e)
        total_psi += psi_i
        bin_details.append({
            "bin_label": str(cat),
            "expected_pct": round(float(e), 4),
            "actual_pct": round(float(a), 4),
            "psi_contribution": round(float(psi_i), 4)
        })

    return round(float(total_psi), 4), bin_details


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V statistic for categorical-categorical association."""
    try:
        contingency_table = pd.crosstab(x, y)
        if contingency_table.size == 0:
            return 0.0
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        if min(k - 1, r - 1) == 0:
            return 0.0
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0
        return round(float(np.sqrt(phi2corr / denom)), 4)
    except Exception:
        return 0.0
