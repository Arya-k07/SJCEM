"""
Distribution Drift Engine: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    FindingCategory,
    FindingStatus,
    Severity,
)
from dataset_autopilot.schemas import DatasetProfile, DriftMetric, Finding
from dataset_autopilot.utils import calculate_psi, safe_float


def partition_for_drift(
    df: pd.DataFrame,
    profile: DatasetProfile,
    split_col: Optional[str] = None,
    time_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Partition dataset into baseline (expected) and current (actual) slices."""
    if split_col and split_col in df.columns:
        splits = df[split_col].dropna().unique()
        if len(splits) >= 2:
            return df[df[split_col] == splits[0]], df[df[split_col] == splits[1]], f"Split by column '{split_col}' ({splits[0]} vs {splits[1]})"

    if time_col and time_col in df.columns:
        sorted_df = df.sort_values(by=time_col)
        split_idx = int(len(sorted_df) * 0.70)
        return sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:], f"Time-ordered partition by '{time_col}' (70% Baseline / 30% Current)"

    # Look for any datetime feature
    for col, prof in profile.columns_profile.items():
        if prof.dtype.startswith("datetime") or "date" in col.lower() or "time" in col.lower():
            try:
                sorted_df = df.sort_values(by=col)
                split_idx = int(len(sorted_df) * 0.70)
                return sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:], f"Chronological split on detected time column '{col}' (70/30)"
            except Exception:
                pass

    # Default: Sequential chronological partition
    split_idx = int(len(df) * 0.70)
    return df.iloc[:split_idx], df.iloc[split_idx:], "Sequential Index Partition (70% Baseline / 30% Evaluation)"


def analyze_drift(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: Optional[str] = None,
    split_col: Optional[str] = None,
    time_col: Optional[str] = None,
    config: AutopilotConfig = DEFAULT_CONFIG
) -> Tuple[List[DriftMetric], List[Finding]]:
    """Analyze feature distribution shift between baseline and current data."""
    drift_metrics: List[DriftMetric] = []
    findings: List[Finding] = []

    if len(df) < 50:
        return drift_metrics, findings

    base_df, curr_df, partition_desc = partition_for_drift(df, profile, split_col, time_col)
    if len(base_df) < 20 or len(curr_df) < 20:
        return drift_metrics, findings

    finding_counter = 300

    for col in df.columns:
        if col == target_col or col == split_col or col == time_col:
            continue

        col_prof = profile.columns_profile.get(col)
        if not col_prof or col_prof.is_id or col_prof.is_constant:
            continue

        exp_vals = base_df[col].dropna().values
        act_vals = curr_df[col].dropna().values

        if len(exp_vals) < 10 or len(act_vals) < 10:
            continue

        psi_val, bin_details = calculate_psi(exp_vals, act_vals, num_bins=config.drift_bins)

        ks_stat = None
        ks_pval = None
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                res = stats.ks_2samp(exp_vals.astype(float), act_vals.astype(float))
                ks_stat = safe_float(res.statistic)
                ks_pval = safe_float(res.pvalue)
            except Exception:
                pass

        is_critical = psi_val >= config.drift_psi_critical
        is_warning = psi_val >= config.drift_psi_warning or (ks_pval is not None and ks_pval < 0.001 and psi_val >= 0.08)

        sev = Severity.CRITICAL if is_critical else (Severity.WARNING if is_warning else Severity.INFO)
        is_drift = is_critical or is_warning

        drift_metrics.append(
            DriftMetric(
                feature=col,
                psi=safe_float(psi_val) or 0.0,
                ks_statistic=ks_stat,
                ks_p_value=ks_pval,
                is_drift_detected=is_drift,
                severity=sev,
                bin_details=bin_details
            )
        )

        if is_drift:
            status_enum = FindingStatus.SUSPECTED if is_critical else FindingStatus.OBSERVED
            findings.append(
                Finding(
                    id=f"F-D{len(findings) + 1}",
                    type=FindingCategory.DISTRIBUTION_DRIFT,
                    status=status_enum,
                    severity=Severity.WARNING if is_critical else Severity.INFO,
                    column=col,
                    title=f"Distribution Shift Observed in '{col}' (PSI={psi_val:.3f})",
                    description=f"Feature '{col}' exhibits measurable population shift (PSI = {psi_val:.3f}).",
                    evidence={
                        "psi": safe_float(psi_val),
                        "ks_statistic": safe_float(ks_stat),
                        "ks_p_value": safe_float(ks_pval),
                        "baseline_rows": len(base_df),
                        "evaluation_rows": len(curr_df)
                    },
                    decision_question=f"Does temporal or group-aware model weighting mitigate performance degradation caused by shift in '{col}'?",
                    recommendation="Investigate if distribution shift represents seasonal variation, macroeconomic change, or data pipeline drift.",
                    confidence=0.90
                )
            )

    return drift_metrics, findings
