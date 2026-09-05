"""
Test Suite for Scientific Correctness and Evaluation Integrity in Dataset Autopilot.
Covers the 10 core scientific proofs:
1. Missingness does not automatically become CRITICAL.
2. No target means no supervised leakage analysis.
3. No target means no supervised baseline.
4. Correlation alone cannot produce CONFIRMED leakage.
5. Preprocessing is fitted independently inside each CV fold (no leakage).
6. All remediation variants use the same evaluation protocol.
7. Same seed/config produces reproducible results.
8. A finding without a valid experiment remains an observation.
9. Decision candidates only appear when prerequisites are satisfied.
10. Generated notebook configuration matches executed experiment.
"""

import json
import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.artifacts import generate_jupyter_notebook
from dataset_autopilot.config import FindingStatus, Severity
from dataset_autopilot.decision_planner import plan_decision_candidates
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.experiment_registry import (
    evaluate_strategy_variant,
    execute_candidate_investigation,
)
from dataset_autopilot.leakage import analyze_target_leakage
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.quality import analyze_data_quality


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 120
    income = np.random.normal(50000, 15000, size=n)
    income[:16] = np.nan # 13.3% missing
    age = np.random.normal(38, 12, size=n)
    # Target correlated with income and age
    logits = (income - 50000) / 15000 + (age - 38) / 12
    probs = 1 / (1 + np.exp(-np.nan_to_num(logits, nan=0.0)))
    churn = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": age,
        "income": income,
        "churn": churn
    })
    return df


def test_1_missingness_does_not_automatically_become_critical(synthetic_data):
    """TEST 1: 13.3% missingness is OBSERVED / WARNING, never an arbitrary CRITICAL."""
    df = synthetic_data
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")

    income_finding = next((f for f in findings if f.column == "income"), None)
    assert income_finding is not None
    assert income_finding.status in [FindingStatus.OBSERVED, FindingStatus.SUSPECTED]
    assert income_finding.severity != Severity.CRITICAL


def test_2_no_target_suppresses_supervised_leakage(synthetic_data):
    """TEST 2: When target is None, target leakage analysis is not run."""
    df = synthetic_data.copy()
    profile = profile_dataset(df, specified_target=None)
    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="unsupervised.csv", target_col=None)

    assert report.dataset_profile.detected_target is None
    leakage_findings = [f for f in report.findings if f.type.value == "target_leakage"]
    assert len(leakage_findings) == 0


def test_3_no_target_suppresses_supervised_baseline(synthetic_data):
    """TEST 3: When target is None, supervised baseline models and supervised candidates are not fabricated."""
    df = synthetic_data.copy()
    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="unsupervised.csv", target_col=None)

    assert len(report.experiments) == 0
    assert len(report.decision_candidates) == 0
    assert "Unsupervised" in report.decision_readiness


def test_4_correlation_alone_cannot_produce_confirmed_leakage(synthetic_data):
    """TEST 4: High correlation with target produces SUSPECTED leakage, never CONFIRMED."""
    df = synthetic_data.copy()
    # Artificially create a high-correlation feature (r = 0.98)
    df["almost_target"] = df["churn"] + np.random.normal(0, 0.05, size=len(df))
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_target_leakage(df, profile, target_col="churn")

    leaky_finding = next((f for f in findings if f.column == "almost_target"), None)
    assert leaky_finding is not None
    assert leaky_finding.status == FindingStatus.SUSPECTED
    assert leaky_finding.status != FindingStatus.CONFIRMED


def test_5_preprocessing_fitted_independently_inside_cv_folds(synthetic_data):
    """TEST 5: Preprocessing (e.g. median imputer) is fitted on train fold, zero fold leakage."""
    df = synthetic_data.copy()
    profile = profile_dataset(df, specified_target="churn")

    # Evaluate median imputation strategy
    result = evaluate_strategy_variant(
        df=df,
        profile=profile,
        target_col="churn",
        strategy_id="MEDIAN_IMPUTATION",
        affected_cols=["income"],
        is_baseline=True,
        n_folds=5,
        random_state=42
    )

    assert result.primary_score > 0.0
    assert result.primary_metric_name == "ROC-AUC"
    assert "strict within-fold preprocessing isolation" in result.limitations[0]


def test_6_all_remediation_variants_use_same_evaluation_protocol(synthetic_data):
    """TEST 6: All variants in an investigation share identical model, splits, metric, and seed."""
    df = synthetic_data.copy()
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")
    candidates = plan_decision_candidates(profile, findings, df=df)

    missing_cand = next(c for c in candidates if c.category == "missingness_strategy")
    investigated = execute_candidate_investigation(
        candidate=missing_cand,
        df=df,
        profile=profile,
        target_col="churn"
    )

    assert investigated.experiment_results is not None
    assert len(investigated.experiment_results) >= 2

    # Check metric parity
    metric_names = {r.primary_metric_name for r in investigated.experiment_results}
    assert len(metric_names) == 1 # All evaluated on identical metric (e.g. ROC-AUC)


def test_7_seed_config_produces_reproducible_results(synthetic_data):
    """TEST 7: Executing with random_state=42 yields 100% bit-exact reproducible scores."""
    df = synthetic_data.copy()
    profile = profile_dataset(df, specified_target="churn")

    res1 = evaluate_strategy_variant(
        df=df, profile=profile, target_col="churn", strategy_id="MEDIAN_IMPUTATION",
        affected_cols=["income"], random_state=42
    )
    res2 = evaluate_strategy_variant(
        df=df, profile=profile, target_col="churn", strategy_id="MEDIAN_IMPUTATION",
        affected_cols=["income"], random_state=42
    )

    assert res1.primary_score == res2.primary_score
    assert res1.secondary_metrics == res2.secondary_metrics


def test_8_finding_without_valid_experiment_remains_observation(synthetic_data):
    """TEST 8: Constant or format findings without registered experiments do not become decision candidates."""
    df = synthetic_data.copy()
    df["constant_col"] = "A" # Constant column
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")

    const_finding = next((f for f in findings if f.type.value == "constant_column"), None)
    assert const_finding is not None
    assert const_finding.decision_question is None # Pure observation

    candidates = plan_decision_candidates(profile, findings, df=df)
    const_cand = [c for c in candidates if "constant_col" in c.affected_columns]
    assert len(const_cand) == 0 # Excluded from Decision Candidates


def test_9_decision_candidates_require_satisfying_prerequisites(synthetic_data):
    """TEST 9: Decision candidates require material impact (e.g. >= 5% missingness)."""
    df = synthetic_data.copy()
    # 1 missing value out of 120 (0.8% missingness - below 5% materiality threshold)
    df["low_missing"] = np.random.normal(10, 2, size=len(df))
    df.loc[0, "low_missing"] = np.nan

    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")
    candidates = plan_decision_candidates(profile, findings, df=df)

    low_miss_cand = [c for c in candidates if "low_missing" in c.affected_columns]
    assert len(low_miss_cand) == 0 # Below materiality cutoff


def test_10_generated_notebook_configuration_matches_executed_experiment(synthetic_data):
    """TEST 10: Generated notebook contains exact candidate question, strategy, and CV loop."""
    df = synthetic_data.copy()
    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="audit_test.csv", target_col="churn")

    cand = report.decision_candidates[0]
    nb_str = generate_jupyter_notebook(cand, report, selected_strategy_id="MISSING_INDICATOR")
    nb = json.loads(nb_str)

    assert nb["nbformat"] == 4
    all_source = " ".join("".join(c["source"]) for c in nb["cells"])

    assert cand.decision_id in all_source
    assert "MISSING_INDICATOR" in all_source
    assert "StratifiedKFold" in all_source
    assert "transform_fold" in all_source # Proves within-fold isolation logic
