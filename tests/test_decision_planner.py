import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.config import FindingCategory, FindingStatus, Severity
from dataset_autopilot.decision_planner import plan_decision_candidates
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.quality import analyze_data_quality
from dataset_autopilot.schemas import AnalysisContext, Finding


def test_unsupervised_no_target_suppresses_supervised_candidates():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 70000, 80000, 90000]
    })
    prof = profile_dataset(df, specified_target=None)
    findings = [
        Finding(
            id="F-1",
            type=FindingCategory.TARGET_LEAKAGE,
            status=FindingStatus.SUSPECTED,
            severity=Severity.WARNING,
            column="income",
            title="Suspected Leakage",
            description="High correlation"
        )
    ]
    candidates = plan_decision_candidates(prof, findings, context=AnalysisContext(task_type="unsupervised"))
    # Supervised leakage candidates must be suppressed
    assert not any(c.category == "suspected_leakage" for c in candidates)


def test_missingness_decision_candidate_generation():
    np.random.seed(42)
    n = 100
    series_missing = np.random.normal(50, 10, size=n)
    series_missing[:15] = np.nan  # 15% missing

    df = pd.DataFrame({
        "feature_a": np.random.normal(10, 2, size=n),
        "series_missing": series_missing,
        "target": np.random.choice([0, 1], size=n)
    })
    prof = profile_dataset(df, specified_target="target")
    findings = analyze_data_quality(df, prof, target_col="target")

    candidates = plan_decision_candidates(prof, findings, df=df)
    missing_candidates = [c for c in candidates if c.category == "missingness_strategy"]
    assert len(missing_candidates) >= 1
    cand = missing_candidates[0]
    assert cand.affected_columns == ["series_missing"]
    assert "DROP_FEATURE" in cand.available_strategies
    assert "MEDIAN_IMPUTATION" in cand.available_strategies
    assert "MISSING_INDICATOR" in cand.available_strategies
    assert "Does retaining 'series_missing'" in cand.question


def test_leakage_decision_candidate_generation():
    findings = [
        Finding(
            id="F-L201",
            type=FindingCategory.TARGET_LEAKAGE,
            status=FindingStatus.SUSPECTED,
            severity=Severity.WARNING,
            column="post_refund",
            title="Suspected Target Leakage in 'post_refund'",
            description="Near perfect correlation",
            evidence={"single_feature_metric": "AUC", "single_feature_score": 0.99}
        )
    ]
    df = pd.DataFrame({"post_refund": [1, 0, 1], "target": [1, 0, 1]})
    prof = profile_dataset(df, specified_target="target")
    candidates = plan_decision_candidates(prof, findings, df=df)

    leakage_candidates = [c for c in candidates if c.category == "suspected_leakage"]
    assert len(leakage_candidates) == 1
    cand = leakage_candidates[0]
    assert cand.affected_columns == ["post_refund"]
    assert "REMOVE_SUSPECT_FEATURE" in cand.available_strategies
    assert cand.expected_impact == "HIGH"
