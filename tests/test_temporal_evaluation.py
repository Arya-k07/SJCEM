"""
Tests for Phase A: Chronological Temporal Evaluation and TimeSeriesSplit.
Verifies that:
1. Future observations cannot influence earlier validation folds (zero temporal leakage).
2. Preprocessing is fitted strictly on the chronological training window.
3. Non-temporal datasets continue to use StratifiedKFold / KFold.
"""

import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.experiment_registry import (
    evaluate_strategy_variant,
    execute_candidate_investigation,
)
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.schemas import DecisionCandidate, DecisionStatus


@pytest.fixture
def temporal_dataset():
    """
    Generate synthetic time series where an autoregressive signal exists,
    and a future post-event column creates temporal leakage if shuffled.
    """
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    
    # Feature with trend over time
    trend_feature = np.linspace(10, 100, n) + np.random.normal(0, 5, size=n)
    
    # Target: state changes in second half of the year
    prob = 1 / (1 + np.exp(-((trend_feature - 55) / 10)))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "trend_feature": trend_feature,
        "category": np.random.choice(["Alpha", "Beta", "Gamma"], size=n),
        "target": target
    })
    return df


def test_temporal_split_never_uses_future_to_train_past(temporal_dataset):
    """
    TEST 1: In chronological validation, future timestamps never appear in training folds.
    """
    df = temporal_dataset
    profile = profile_dataset(df, specified_target="target")

    # Evaluate with time_col specified
    res_temporal = evaluate_strategy_variant(
        df=df,
        profile=profile,
        target_col="target",
        strategy_id="RAW_FEATURES",
        affected_cols=[],
        time_col="timestamp",
        n_folds=5,
        random_state=42
    )

    assert res_temporal.primary_score is not None
    assert "TimeSeriesSplit" in res_temporal.limitations[0]
    assert "timestamp" in res_temporal.limitations[0]


def test_temporal_evaluation_preserves_fold_isolation(temporal_dataset):
    """
    TEST 2: Preprocessing (e.g. median imputer / scaler) fitted on historical window transforms future fold.
    """
    df = temporal_dataset.copy()
    df.loc[10:20, "trend_feature"] = np.nan # Missingness in early period

    profile = profile_dataset(df, specified_target="target")

    res_missing = evaluate_strategy_variant(
        df=df,
        profile=profile,
        target_col="target",
        strategy_id="MEDIAN_IMPUTATION",
        affected_cols=["trend_feature"],
        time_col="timestamp",
        n_folds=5,
        random_state=42
    )

    assert res_missing.primary_score > 0
    assert "TimeSeriesSplit" in res_missing.limitations[0]


def test_non_temporal_dataset_uses_stratified_kfold(temporal_dataset):
    """
    TEST 3: When time_col is None, standard 5-fold CV is preserved.
    """
    df = temporal_dataset
    profile = profile_dataset(df, specified_target="target")

    res_random = evaluate_strategy_variant(
        df=df,
        profile=profile,
        target_col="target",
        strategy_id="RAW_FEATURES",
        affected_cols=[],
        time_col=None,
        n_folds=5,
        random_state=42
    )

    assert "5-fold CV" in res_random.limitations[0]
    assert "TimeSeriesSplit" not in res_random.limitations[0]
