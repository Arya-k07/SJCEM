import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.bias import analyze_slice_bias
from dataset_autopilot.drift import analyze_drift
from dataset_autopilot.leakage import analyze_target_leakage
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.quality import analyze_data_quality


def test_quality_missingness_and_outliers():
    np.random.seed(42)
    n = 100
    age = np.random.normal(30, 5, size=n)
    # Inject 5 extreme outliers (5% > 3% threshold)
    for idx in range(5):
        age[idx] = 500 + idx * 50

    salary = np.random.normal(50000, 10000, size=n)
    # Inject missingness
    salary[:25] = np.nan

    df = pd.DataFrame({"age": age, "salary": salary, "label": np.random.choice([0, 1], size=n)})
    prof = profile_dataset(df, specified_target="label")
    findings = analyze_data_quality(df, prof, target_col="label")

    types = [f.type.value for f in findings]
    assert "missingness" in types
    assert "outliers" in types


def test_target_leakage_detection():
    n = 100
    churn = np.random.choice([0, 1], size=n)
    # Perfectly leaking feature
    refund_post_churn = churn.copy()
    feature_clean = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({
        "feature_clean": feature_clean,
        "refund_post_churn": refund_post_churn,
        "churn": churn
    })
    prof = profile_dataset(df, specified_target="churn")
    findings = analyze_target_leakage(df, prof, target_col="churn")

    leaks = [f for f in findings if f.type.value == "target_leakage"]
    assert len(leaks) >= 1
    assert leaks[0].column == "refund_post_churn"


def test_drift_psi_analysis():
    np.random.seed(42)
    # Baseline normal, test shifted normal
    exp_vals = np.random.normal(10, 2, size=200)
    act_vals = np.random.normal(25, 4, size=100) # Heavy drift

    all_vals = np.concatenate([exp_vals, act_vals])
    split = ["train"] * 200 + ["test"] * 100

    df = pd.DataFrame({"feature_x": all_vals, "split": split, "y": np.random.choice([0, 1], size=300)})
    prof = profile_dataset(df, specified_target="y")
    drift_metrics, findings = analyze_drift(df, prof, target_col="y", split_col="split")

    assert len(drift_metrics) >= 1
    feat_drift = next(d for d in drift_metrics if d.feature == "feature_x")
    assert feat_drift.psi > 0.25
    assert feat_drift.is_drift_detected is True


def test_slice_bias_analysis():
    n = 200
    gender = np.array(["Male"] * 100 + ["Female"] * 100)
    y_true = np.random.choice([0, 1], size=n)
    # Predictions perform well for Male, poorly for Female
    y_pred = y_true.copy()
    y_pred[100:] = 1 - y_true[100:]  # flip female predictions to induce disparity

    df = pd.DataFrame({"gender": gender, "y": y_true})
    prof = profile_dataset(df, specified_target="y")
    slice_metrics, findings = analyze_slice_bias(
        df=df,
        profile=prof,
        target_col="y",
        y_true=y_true,
        y_pred=y_pred,
        is_classification=True
    )

    assert len(slice_metrics) > 0
    disparate = [s for s in slice_metrics if s.is_disparate]
    assert len(disparate) > 0
