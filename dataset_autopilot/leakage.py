"""
Leakage Detection Engine: identifies target leakage, single-feature predictive dominance,
post-event temporal leakage, and train-test contamination.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    FindingCategory,
    FindingStatus,
    Severity,
)
from dataset_autopilot.schemas import DatasetProfile, Finding
from dataset_autopilot.utils import compute_cramers_v, safe_float


POST_EVENT_KEYWORDS = re.compile(
    r"(refund|cancel|discharge|churn_date|exit_date|closed_date|payout|payout_date|post_|after_|resolution|approved_date|end_date)",
    re.IGNORECASE
)


def compute_single_feature_metric(
    feature_series: pd.Series,
    target_series: pd.Series,
    is_classification: bool
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute single-feature ROC-AUC / R2 and correlation with target.
    Returns (predictive_score, correlation).
    """
    # Align and drop NaNs
    valid_mask = feature_series.notna() & target_series.notna()
    if valid_mask.sum() < 20:
        return None, None

    x = feature_series[valid_mask]
    y = target_series[valid_mask]

    # Convert x to numeric or category codes
    if not pd.api.types.is_numeric_dtype(x):
        x = pd.Categorical(x).codes
        x = pd.Series(x, index=y.index)

    # Convert y if classification
    if is_classification:
        if not pd.api.types.is_numeric_dtype(y):
            y_cats = pd.Categorical(y).codes
        else:
            y_cats = y.values
    else:
        y_cats = y.astype(float).values

    corr_val = None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_val, _ = stats.spearmanr(x, y_cats)
            if np.isnan(corr_val):
                corr_val = 0.0
    except Exception:
        corr_val = 0.0

    score = None
    try:
        x_mat = np.asarray(x).reshape(-1, 1)
        if is_classification and len(np.unique(y_cats)) == 2:
            # Fit a shallow single-feature decision tree
            dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt.fit(x_mat, y_cats)
            probs = dt.predict_proba(x_mat)[:, 1]
            score = roc_auc_score(y_cats, probs)
            # Ensure AUC >= 0.5
            if score < 0.5:
                score = 1.0 - score
        elif not is_classification:
            dt = DecisionTreeRegressor(max_depth=3, random_state=42)
            dt.fit(x_mat, y_cats)
            score = dt.score(x_mat, y_cats)  # R2
    except Exception:
        score = None

    return safe_float(score), safe_float(corr_val)


def analyze_target_leakage(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: str,
    config: AutopilotConfig = DEFAULT_CONFIG
) -> List[Finding]:
    """Detect features that leak the target variable or occur post-target event."""
    findings: List[Finding] = []
    if target_col not in df.columns:
        return findings

    target_series = df[target_col]
    is_classification = profile.target_type in ["binary_classification", "multiclass_classification"]
    finding_counter = 200

    for col in df.columns:
        if col == target_col:
            continue

        col_prof = profile.columns_profile.get(col)
        if not col_prof or col_prof.is_id or col_prof.is_constant:
            continue

        series = df[col]
        score, corr = compute_single_feature_metric(series, target_series, is_classification)
        is_post_event_named = bool(POST_EVENT_KEYWORDS.search(col))

        is_critical = False
        is_warning = False
        reasons = []

        if is_classification and score is not None:
            if score >= config.leakage_auc_critical:
                is_critical = True
                reasons.append(f"Single-feature AUC is exceptionally high ({score:.3f} >= {config.leakage_auc_critical})")
            elif score >= config.leakage_auc_warning:
                is_warning = True
                reasons.append(f"Single-feature AUC is suspicious ({score:.3f} >= {config.leakage_auc_warning})")

        if corr is not None:
            if abs(corr) >= config.leakage_corr_critical:
                is_critical = True
                reasons.append(f"Extreme correlation with target (|r| = {abs(corr):.3f})")
            elif abs(corr) >= config.leakage_corr_warning:
                is_warning = True
                reasons.append(f"High correlation with target (|r| = {abs(corr):.3f})")

        if is_post_event_named and (is_warning or (score and score > 0.80) or (corr and abs(corr) > 0.50)):
            is_critical = True
            reasons.append(f"Feature name matches post-event semantic pattern ('{col}')")

        if is_critical or is_warning:
            finding_counter += 1
            findings.append(
                Finding(
                    id=f"F-L{finding_counter}",
                    type=FindingCategory.TARGET_LEAKAGE,
                    status=FindingStatus.SUSPECTED,
                    severity=Severity.WARNING,
                    column=col,
                    title=f"Suspected Target Leakage in '{col}'",
                    description=f"Feature '{col}' exhibits high predictive association with target ({'; '.join(reasons)}). In production, this information may be unavailable at inference time.",
                    evidence={
                        "single_feature_metric": "AUC" if is_classification else "R2",
                        "single_feature_score": score,
                        "correlation_with_target": corr,
                        "post_event_name_match": is_post_event_named,
                        "reasons": reasons
                    },
                    decision_question=f"How much of the model's apparent performance depends on suspicious feature '{col}'?",
                    recommendation=f"Execute feature ablation to measure true out-of-fold generalization without '{col}'.",
                    confidence=0.85
                )
            )

    return findings


def analyze_train_test_contamination(
    df: pd.DataFrame,
    profile: DatasetProfile,
    split_col: Optional[str] = None,
    id_col: Optional[str] = None
) -> List[Finding]:
    """Check for shared identifiers or duplicate records across train and test partitions."""
    findings: List[Finding] = []
    
    # Check if split column exists
    candidate_split_cols = [c for c in df.columns if c.lower() in ["split", "dataset_split", "is_train", "partition", "fold"]]
    active_split_col = split_col if split_col and split_col in df.columns else (candidate_split_cols[0] if candidate_split_cols else None)

    if not active_split_col:
        return findings

    split_series = df[active_split_col].astype(str)
    unique_splits = split_series.unique()
    if len(unique_splits) < 2:
        return findings

    # Look for candidate ID column
    candidate_id_cols = [col for col, p in profile.columns_profile.items() if p.is_id]
    active_id_col = id_col if id_col and id_col in df.columns else (candidate_id_cols[0] if candidate_id_cols else None)

    train_mask = split_series == unique_splits[0]
    test_mask = split_series == unique_splits[1]

    train_df = df[train_mask]
    test_df = df[test_mask]

    # Check ID overlap
    if active_id_col:
        train_ids = set(train_df[active_id_col].dropna().unique())
        test_ids = set(test_df[active_id_col].dropna().unique())
        shared_ids = train_ids.intersection(test_ids)
        if len(shared_ids) > 0:
            contamination_pct = len(shared_ids) / len(test_ids) if len(test_ids) > 0 else 0.0
            findings.append(
                Finding(
                    id="F-C101",
                    type=FindingCategory.TRAIN_TEST_CONTAMINATION,
                    status=FindingStatus.SUSPECTED,
                    severity=Severity.WARNING,
                    column=active_id_col,
                    title=f"Train-Test Identifier Overlap in '{active_id_col}'",
                    description=f"{len(shared_ids)} identifiers appear in both '{unique_splits[0]}' and '{unique_splits[1]}' partitions ({contamination_pct * 100:.2f}% test partition overlap).",
                    evidence={
                        "shared_id_count": len(shared_ids),
                        "contamination_pct": safe_float(contamination_pct),
                        "split_column": active_split_col,
                        "sample_shared_ids": list(shared_ids)[:5]
                    },
                    decision_question=f"Does group-aware partitioning on '{active_id_col}' provide a materially more realistic evaluation of generalization?",
                    recommendation=f"Use GroupKFold or entity-level partitioning on '{active_id_col}' to prevent entity leakage between splits.",
                    confidence=0.95
                )
            )

    return findings
