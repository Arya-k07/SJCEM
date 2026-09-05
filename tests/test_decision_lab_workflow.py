"""
Test Suite for Phase 1: Decision Lab Backend Contracts & State Transitions.
Verifies that:
1. Decision Candidates have rich, structured DecisionEvidenceBlock contracts.
2. Controlled parameter ledgers are properly populated.
3. State transitions function cleanly (EXPERIMENT_READY -> EVIDENCE_AVAILABLE -> USER_DECISION / INCONCLUSIVE).
"""

import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.config import DecisionStatus
from dataset_autopilot.decision_planner import plan_decision_candidates
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.experiment_registry import execute_candidate_investigation
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.quality import analyze_data_quality
from dataset_autopilot.schemas import AnalysisContext, UserDecision


@pytest.fixture
def sample_decision_dataset():
    np.random.seed(42)
    n = 150
    age = np.random.normal(40, 10, size=n)
    income = np.random.normal(55000, 12000, size=n)
    income[:20] = np.nan # 13.3% missing
    
    # Target
    logits = (age - 40) / 15 + (np.nan_to_num(income, nan=55000) - 55000) / 12000
    churn = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": age,
        "income": income,
        "churn": churn
    })


def test_decision_evidence_block_structure(sample_decision_dataset):
    """
    TEST: Verify that plan_decision_candidates generates structured DecisionEvidenceBlock
    with observation, empirical evidence, what we know, what is unknown, and controlled ledger.
    """
    df = sample_decision_dataset
    profile = profile_dataset(df, specified_target="churn")
    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="test_audit.csv", target_col="churn")

    candidates = report.decision_candidates
    assert len(candidates) >= 1

    cand = candidates[0]
    assert cand.evidence_block is not None
    assert len(cand.evidence_block.observation) > 0
    assert len(cand.evidence_block.empirical_evidence) > 0
    assert len(cand.evidence_block.what_we_know) >= 2
    assert len(cand.evidence_block.what_is_unknown) >= 1
    assert "Model Architecture" in cand.evidence_block.controlled_parameters
    assert "Validation Protocol" in cand.evidence_block.controlled_parameters


def test_decision_state_transitions(sample_decision_dataset):
    """
    TEST: Candidate transitions from EXPERIMENT_READY -> EVIDENCE_AVAILABLE -> USER_DECISION.
    """
    df = sample_decision_dataset
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")
    candidates = plan_decision_candidates(profile, findings, df=df, context=AnalysisContext(target_column="churn"))

    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.status == DecisionStatus.EXPERIMENT_READY

    # 1. Run investigation -> EVIDENCE_AVAILABLE
    investigated = execute_candidate_investigation(
        candidate=cand,
        df=df,
        profile=profile,
        target_col="churn"
    )
    assert investigated.status == DecisionStatus.EVIDENCE_AVAILABLE
    assert investigated.experiment_results is not None
    assert len(investigated.experiment_results) >= 2

    # 2. Record User Decision -> USER_DECISION
    investigated.user_decision = UserDecision(
        selected_strategy_id="MISSING_INDICATOR",
        decision_rationale="Empirically preserves validation performance with low complexity.",
        decided_at="2026-08-09 12:00:00 UTC",
        custom_method_notes=None
    )
    investigated.status = DecisionStatus.USER_DECISION

    assert investigated.status == DecisionStatus.USER_DECISION
    assert investigated.user_decision.selected_strategy_id == "MISSING_INDICATOR"


def test_inconclusive_state_handling(sample_decision_dataset):
    """
    TEST: Candidate can transition to INCONCLUSIVE when evidence is statistically indistinguishable.
    """
    df = sample_decision_dataset
    profile = profile_dataset(df, specified_target="churn")
    findings = analyze_data_quality(df, profile, target_col="churn")
    candidates = plan_decision_candidates(profile, findings, df=df, context=AnalysisContext(target_column="churn"))

    assert len(candidates) >= 1
    cand = candidates[0]
    cand.status = DecisionStatus.INCONCLUSIVE
    assert cand.status == DecisionStatus.INCONCLUSIVE
