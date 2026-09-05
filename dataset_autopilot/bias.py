"""
Bias and Slice Analysis Engine: evaluates subgroup fairness gaps, disparate impact,
and slice performance disparities.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    FindingCategory,
    FindingStatus,
    Severity,
)
from dataset_autopilot.schemas import DatasetProfile, Finding, SliceMetric
from dataset_autopilot.utils import safe_float


SENSITIVE_KEYWORDS = [
    "gender", "sex", "age", "race", "ethnicity", "region", "country",
    "nationality", "marital", "religion", "tier", "segment", "income_bracket"
]


def identify_candidate_slices(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: Optional[str] = None
) -> Dict[str, pd.Series]:
    """Extract candidate slice series from categorical features and binned numeric features."""
    slices_dict: Dict[str, pd.Series] = {}

    for col in df.columns:
        if col == target_col:
            continue

        col_prof = profile.columns_profile.get(col)
        if not col_prof or col_prof.is_id or col_prof.is_constant:
            continue

        series = df[col]

        # Categorical columns with 2 to 8 categories
        if 2 <= col_prof.unique_count <= 8:
            slices_dict[col] = series.astype(str)
        # Check for numeric column with sensitive name (e.g. age) -> bin into 3-4 quantiles
        elif pd.api.types.is_numeric_dtype(series) and any(k in col.lower() for k in ["age", "income", "tenure", "score"]):
            try:
                binned = pd.qcut(series.dropna(), q=3, labels=["Low", "Medium", "High"], duplicates="drop")
                slices_dict[f"{col}_quantile"] = binned.astype(str)
            except Exception:
                pass

    return slices_dict


def analyze_slice_bias(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_classification: bool,
    config: AutopilotConfig = DEFAULT_CONFIG
) -> Tuple[List[SliceMetric], List[Finding]]:
    """Compute slice-level evaluation metrics and identify significant fairness or performance gaps."""
    slice_metrics: List[SliceMetric] = []
    findings: List[Finding] = []

    if target_col not in df.columns or len(y_true) == 0:
        return slice_metrics, findings

    candidate_slices = identify_candidate_slices(df, profile, target_col)
    if not candidate_slices:
        return slice_metrics, findings

    # Overall benchmark
    if is_classification:
        overall_metric_name = "F1-Score"
        try:
            overall_score = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception:
            overall_score = float(accuracy_score(y_true, y_pred))
            overall_metric_name = "Accuracy"
    else:
        overall_metric_name = "MAE"
        overall_score = float(mean_absolute_error(y_true, y_pred))

    finding_counter = 400

    for feature_name, slice_series in candidate_slices.items():
        # Align slice series with predictions
        common_len = min(len(slice_series), len(y_true), len(y_pred))
        aligned_slices = slice_series.iloc[:common_len]
        aligned_true = y_true[:common_len]
        aligned_pred = y_pred[:common_len]

        unique_groups = aligned_slices.dropna().unique()
        group_scores = {}

        for group_val in unique_groups:
            group_mask = aligned_slices == group_val
            group_count = int(group_mask.sum())

            if group_count < config.slice_min_sample_size:
                continue

            g_true = aligned_true[group_mask]
            g_pred = aligned_pred[group_mask]

            if is_classification:
                try:
                    g_score = float(f1_score(g_true, g_pred, average="weighted", zero_division=0))
                except Exception:
                    g_score = float(accuracy_score(g_true, g_pred))
                disparity = overall_score - g_score
                is_disparate = disparity >= config.slice_disparity_threshold
                ratio = round(g_score / overall_score, 3) if overall_score > 0 else 1.0
            else:
                g_score = float(mean_absolute_error(g_true, g_pred))
                # For MAE, higher is worse
                disparity = g_score - overall_score
                is_disparate = (g_score / overall_score) > 1.30 if overall_score > 0 else False
                ratio = round(g_score / overall_score, 3) if overall_score > 0 else 1.0

            group_scores[group_val] = g_score

            metric_item = SliceMetric(
                feature=feature_name,
                slice_value=str(group_val),
                sample_count=group_count,
                sample_pct=round(group_count / common_len, 4),
                metric_name=overall_metric_name,
                slice_score=safe_float(g_score) or 0.0,
                overall_score=safe_float(overall_score) or 0.0,
                disparity_ratio=safe_float(ratio) or 1.0,
                is_disparate=is_disparate
            )
            slice_metrics.append(metric_item)

            if is_disparate:
                finding_counter += 1
                findings.append(
                    Finding(
                        id=f"F-B{finding_counter}",
                        type=FindingCategory.SLICE_BIAS,
                        status=FindingStatus.OBSERVED,
                        severity=Severity.WARNING,
                        column=feature_name,
                        title=f"Performance Gap in Slice '{feature_name} = {group_val}'",
                        description=(
                            f"Model {overall_metric_name} on subgroup '{feature_name} = {group_val}' is {g_score:.3f}, "
                            f"showing a {abs(disparity):.3f} deviation from overall baseline ({overall_score:.3f})."
                        ),
                        evidence={
                            "subgroup": str(group_val),
                            "subgroup_sample_size": group_count,
                            "subgroup_metric": safe_float(g_score),
                            "overall_metric": safe_float(overall_score),
                            "disparity_delta": safe_float(disparity),
                            "disparity_ratio": safe_float(ratio)
                        },
                        decision_question=f"Does group-aware reweighting or separate slice modeling improve fairness for '{group_val}' without degrading overall performance?",
                        recommendation=f"Evaluate whether subgroup representation should be balanced or if separate subgroup decision thresholds are needed.",
                        confidence=0.88
                    )
                )

    return slice_metrics, findings
