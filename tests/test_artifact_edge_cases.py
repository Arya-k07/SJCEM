"""
Tests for Phase H: Artifact generation with difficult / adversarial column names.
Tests columns with:
- spaces and parentheses: 'Annual Income (USD)'
- quotes: 'user"quote"name'
- apostrophes: "buyer's_address"
- unicode characters: 'revenue_€_total'
- reserved Python keywords: 'class', 'def', 'import'
"""

import json
import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.artifacts import (
    generate_decision_html,
    generate_jupyter_notebook,
    generate_pipeline_script,
)
from dataset_autopilot.config import DecisionStatus
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.schemas import (
    AnalysisContext,
    AuditReport,
    DecisionCandidate,
    HealthScoreBreakdown,
)


@pytest.fixture
def edge_case_dataset():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "Annual Income (USD)": np.random.normal(50000, 10000, size=n),
        "buyer's_address": np.random.choice(["Street A", "Street B"], size=n),
        "revenue_€_total": np.random.uniform(100, 1000, size=n),
        "class": np.random.normal(10, 2, size=n),
        "def": np.random.choice([0, 1], size=n),
        "target_col": np.random.choice([0, 1], size=n)
    })
    # Inject missingness in 'Annual Income (USD)'
    df.loc[:10, "Annual Income (USD)"] = np.nan
    return df


def test_artifact_generation_with_adversarial_column_names(edge_case_dataset):
    """
    TEST Phase H: Verify that generated notebook JSON and pipeline scripts are valid
    even with spaces, apostrophes, unicode, and reserved Python keywords in column names.
    """
    df = edge_case_dataset
    profile = profile_dataset(df, specified_target="target_col")

    candidate = DecisionCandidate(
        decision_id="DEC-EDGE",
        finding_ids=["F-EDGE-01"],
        category="missingness_strategy",
        question="How to handle 'Annual Income (USD)'?",
        affected_columns=["Annual Income (USD)", "buyer's_address", "class"],
        rationale="Testing edge cases with spaces, apostrophes, and reserved keywords",
        expected_impact="HIGH",
        evidence={},
        available_strategies=["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"],
        confidence=0.90,
        status=DecisionStatus.EXPERIMENT_READY
    )

    report = AuditReport(
        report_id="RPT-EDGE",
        dataset_name='test_adversarial_"dataset".csv',
        created_at="2026-08-09 12:00:00 UTC",
        health_score=85.0,
        health_grade="A",
        decision_readiness="1 decision ready",
        score_breakdown=HealthScoreBreakdown(
            overall_score=85.0, quality_score=85.0, leakage_score=85.0, drift_score=85.0, fairness_score=85.0, grade="A"
        ),
        dataset_profile=profile,
        findings=[],
        experiments=[],
        decision_candidates=[candidate],
        drift_metrics=[],
        slice_summary=[],
        executive_summary="Edge case test summary",
        analysis_context=AnalysisContext(target_column="target_col", time_column=None)
    )

    # 1. Jupyter Notebook .ipynb generation (must be valid parseable JSON)
    nb_str = generate_jupyter_notebook(candidate, report, selected_strategy_id="MISSING_INDICATOR")
    nb_json = json.loads(nb_str)
    assert nb_json["nbformat"] == 4
    assert len(nb_json["cells"]) >= 6

    # 2. Pipeline script generation
    pipeline_script = generate_pipeline_script(candidate, report, selected_strategy_id="MISSING_INDICATOR")
    assert "Annual Income (USD)" in pipeline_script
    assert "DecisionPreprocessor" in pipeline_script

    # 3. HTML Decision generation
    html = generate_decision_html(candidate, report)
    assert "Annual Income (USD)" in html
    assert "DEC-EDGE" in html
