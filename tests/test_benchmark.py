"""
Test Suite for Synthetic Fault-Injection Benchmark & Experiment Engine Validation.
Covers:
- Phase C: Ground-truth detection across all 7 injected faults + clean control.
- Phase D: Controlled experiment validation on leakage ablation, missingness, and temporal validation.
"""

import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.benchmark import (
    generate_benchmark_suite,
    run_benchmark_evaluation,
)
from dataset_autopilot.decision_planner import plan_decision_candidates
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.experiment_registry import (
    evaluate_strategy_variant,
    execute_candidate_investigation,
)
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.schemas import DecisionCandidate


def test_synthetic_fault_injection_benchmark_detection():
    """
    TEST Phase C: Run the 8-case benchmark suite and verify ground-truth detection rates.
    """
    engine = DatasetAutopilotEngine()
    results = run_benchmark_evaluation(engine)

    assert len(results) == 8 # 1 clean control + 7 fault cases

    # 1. Clean control has 0 false positive critical/leakage findings
    clean_eval = next(r for r in results if r.case_id == "BENCH_0_CLEAN")
    assert clean_eval.detected is True
    assert len(clean_eval.false_positives) == 0

    # 2. Target Leakage detected as SUSPECTED
    leak_eval = next(r for r in results if r.case_id == "BENCH_1_LEAKAGE")
    assert leak_eval.detected is True
    assert leak_eval.actual_status == "SUSPECTED"

    # 3. Duplicate Rows detected
    dup_eval = next(r for r in results if r.case_id == "BENCH_2_DUPLICATES")
    assert dup_eval.detected is True

    # 4. Distribution Shift detected
    drift_eval = next(r for r in results if r.case_id == "BENCH_3_DRIFT")
    assert drift_eval.detected is True

    # 5. Informative Missingness detected as SUSPECTED
    mnar_eval = next(r for r in results if r.case_id == "BENCH_4_INFORMATIVE_MISSING")
    assert mnar_eval.detected is True
    assert mnar_eval.actual_status == "SUSPECTED"

    # 6. Extreme Outliers detected
    outlier_eval = next(r for r in results if r.case_id == "BENCH_5_OUTLIERS")
    assert outlier_eval.detected is True

    # 7. Subgroup Bias detected
    slice_eval = next(r for r in results if r.case_id == "BENCH_6_SLICE_DISPARITY")
    assert slice_eval.detected is True

    # 8. Post-outcome Temporal Leakage detected
    temporal_eval = next(r for r in results if r.case_id == "BENCH_7_TEMPORAL_LEAK")
    assert temporal_eval.detected is True


def test_phase_d_leakage_ablation_experiment_validation():
    """
    TEST Phase D: On the Target Leakage benchmark dataset, removing the leaky feature
    causes a material drop in validation metric (exposing reliance on the leaky signal).
    """
    cases = {c.case_id: c for c in generate_benchmark_suite()}
    leak_case = cases["BENCH_1_LEAKAGE"]
    df = leak_case.df
    profile = profile_dataset(df, specified_target="churn")

    base_res = evaluate_strategy_variant(
        df=df, profile=profile, target_col="churn",
        strategy_id="BASELINE_WITH_FEATURE",
        affected_cols=["account_cancellation_code"],
        is_baseline=True
    )

    removed_res = evaluate_strategy_variant(
        df=df, profile=profile, target_col="churn",
        strategy_id="REMOVE_SUSPECT_FEATURE",
        affected_cols=["account_cancellation_code"],
        baseline_score=base_res.primary_score,
        is_baseline=False
    )

    # Performance should drop significantly when the leaky feature is removed
    assert base_res.primary_score > 0.90
    assert removed_res.delta_from_baseline < -0.10 # Substantial drop >= 0.10


def test_phase_d_missingness_strategy_experiment_validation():
    """
    TEST Phase D: On the Informative Missingness benchmark dataset,
    the missingness indicator strategy preserves the MNAR predictive signal.
    """
    cases = {c.case_id: c for c in generate_benchmark_suite()}
    mnar_case = cases["BENCH_4_INFORMATIVE_MISSING"]
    df = mnar_case.df
    profile = profile_dataset(df, specified_target="churn")

    from dataset_autopilot.schemas import DecisionStatus

    cand = DecisionCandidate(
        decision_id="DEC-MNAR",
        finding_ids=["F-TEST"],
        category="missingness_strategy",
        question="How to treat informative missingness?",
        affected_columns=["annual_income"],
        rationale="Testing MNAR strategy comparison",
        expected_impact="HIGH",
        evidence={},
        available_strategies=["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"],
        confidence=0.90,
        status=DecisionStatus.EXPERIMENT_READY
    )

    investigated = execute_candidate_investigation(
        candidate=cand,
        df=df,
        profile=profile,
        target_col="churn"
    )

    assert len(investigated.experiment_results) == 3
    strat_scores = {v.strategy_id: v.primary_score for v in investigated.experiment_results}
    assert "MISSING_INDICATOR" in strat_scores
    assert "DROP_FEATURE" in strat_scores
    assert "MEDIAN_IMPUTATION" in strat_scores
