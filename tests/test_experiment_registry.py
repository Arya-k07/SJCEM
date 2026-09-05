import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.config import DecisionStatus
from dataset_autopilot.experiment_registry import (
    evaluate_strategy_variant,
    execute_candidate_investigation,
)
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.schemas import DecisionCandidate


def test_controlled_missingness_experiment_evaluation():
    np.random.seed(42)
    n = 120
    age = np.random.normal(35, 10, size=n)
    salary = np.random.normal(60000, 15000, size=n)
    # Inject missingness in salary
    salary[:20] = np.nan
    target = np.random.choice([0, 1], size=n)

    df = pd.DataFrame({"age": age, "salary": salary, "target": target})
    prof = profile_dataset(df, specified_target="target")

    candidate = DecisionCandidate(
        decision_id="DEC-001",
        category="missingness_strategy",
        question="Does retaining 'salary' with missing indicator outperform dropping it?",
        affected_columns=["salary"],
        rationale="16.7% missing values",
        available_strategies=["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"],
        status=DecisionStatus.EXPERIMENT_READY
    )

    updated = execute_candidate_investigation(candidate, df, prof, target_col="target")
    assert updated.status == DecisionStatus.EVIDENCE_AVAILABLE
    assert updated.experiment_results is not None
    assert len(updated.experiment_results) == 3

    strat_ids = [r.strategy_id for r in updated.experiment_results]
    assert strat_ids == ["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"]

    # Baseline is first
    assert updated.experiment_results[0].is_baseline is True
    assert updated.experiment_results[1].is_baseline is False
    assert updated.system_interpretation is not None


def test_controlled_leakage_ablation():
    np.random.seed(42)
    n = 100
    churn = np.random.choice([0, 1], size=n)
    leaky = churn.copy() # Perfect leakage
    legit = np.random.normal(50, 10, size=n)

    df = pd.DataFrame({"legit": legit, "leaky": leaky, "churn": churn})
    prof = profile_dataset(df, specified_target="churn")

    candidate = DecisionCandidate(
        decision_id="DEC-002",
        category="suspected_leakage",
        question="How much does 'leaky' contribute to model performance?",
        affected_columns=["leaky"],
        rationale="Suspected leakage",
        available_strategies=["BASELINE_WITH_FEATURE", "REMOVE_SUSPECT_FEATURE"],
        status=DecisionStatus.EXPERIMENT_READY
    )

    updated = execute_candidate_investigation(candidate, df, prof, target_col="churn")
    assert updated.status == DecisionStatus.EVIDENCE_AVAILABLE
    assert len(updated.experiment_results) == 2

    base_score = updated.experiment_results[0].primary_score
    ablation_score = updated.experiment_results[1].primary_score
    # Removing perfect leakage must drop score
    assert ablation_score < base_score
    assert updated.experiment_results[1].delta < 0


def test_reproducible_seed_consistency():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(10, 2, size=n),
        "x2": np.random.normal(20, 5, size=n),
        "y": np.random.choice([0, 1], size=n)
    })
    prof = profile_dataset(df, specified_target="y")

    res1 = evaluate_strategy_variant(df, prof, "y", "MEDIAN_IMPUTATION", ["x1"], random_state=42)
    res2 = evaluate_strategy_variant(df, prof, "y", "MEDIAN_IMPUTATION", ["x1"], random_state=42)

    assert res1.primary_score == res2.primary_score
    assert res1.secondary_metrics == res2.secondary_metrics
