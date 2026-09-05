"""
Experiment Planner & Runner: executes controlled A/B experiments to quantify
the exact impact of data issues and remediation strategies on model performance.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from dataset_autopilot.config import FindingCategory
from dataset_autopilot.models import train_and_evaluate_baseline
from dataset_autopilot.schemas import DatasetProfile, ExperimentResult, Finding
from dataset_autopilot.utils import safe_float


def plan_and_run_experiments(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: str,
    findings: List[Finding],
    baseline_result: Dict[str, Any]
) -> List[ExperimentResult]:
    """Plan and execute remediation experiments based on identified findings."""
    experiments: List[ExperimentResult] = []

    primary_metric = baseline_result["primary_metric_name"]
    baseline_score = float(baseline_result["primary_score"] or 0.0)

    # 1. Baseline Experiment Entry
    experiments.append(
        ExperimentResult(
            id="EXP-001",
            name="Baseline Model Benchmark",
            category="baseline",
            description="Controlled baseline Random Forest model evaluated with stratified K-Fold cross-validation.",
            model_name=baseline_result["model_name"],
            baseline_metric_name=primary_metric,
            baseline_score=baseline_score,
            modified_score=baseline_score,
            delta=0.0,
            delta_pct=0.0,
            status="baseline",
            notes="Reference benchmark score prior to applying remediations.",
            detailed_metrics=baseline_result["metrics"]
        )
    )

    # Extract finding categories and affected columns
    leakage_cols = [f.column for f in findings if f.type == FindingCategory.TARGET_LEAKAGE and f.column]
    missing_cols = [f.column for f in findings if f.type == FindingCategory.MISSINGNESS and f.column]
    outlier_cols = [f.column for f in findings if f.type == FindingCategory.OUTLIERS and f.column]

    exp_counter = 1

    # 2. Leakage Removal Experiment
    if leakage_cols:
        exp_counter += 1
        try:
            mod_res = train_and_evaluate_baseline(
                df=df,
                profile=profile,
                target_col=target_col,
                exclude_cols=leakage_cols
            )
            mod_score = float(mod_res["primary_score"] or 0.0)
            delta = round(mod_score - baseline_score, 4)
            delta_pct = round((delta / baseline_score) * 100, 2) if baseline_score != 0 else 0.0

            status = "leakage_confirmed" if delta < -0.03 else ("neutral" if abs(delta) <= 0.03 else "improved")
            notes = (
                f"Model score dropped by {abs(delta):.3f} when removing suspect feature(s) {leakage_cols}, "
                f"confirming artificial metric inflation caused by target leakage."
                if status == "leakage_confirmed"
                else f"Exclusion of {leakage_cols} maintained model performance at realistic baseline."
            )

            experiments.append(
                ExperimentResult(
                    id=f"EXP-00{exp_counter}",
                    name="Remove Leaky Features",
                    category="target_leakage",
                    description=f"Retrain baseline excluding detected leaky feature(s): {', '.join(leakage_cols)}.",
                    model_name=mod_res["model_name"],
                    baseline_metric_name=primary_metric,
                    baseline_score=baseline_score,
                    modified_score=mod_score,
                    delta=delta,
                    delta_pct=delta_pct,
                    status=status,
                    notes=notes,
                    detailed_metrics=mod_res["metrics"]
                )
            )
        except Exception as e:
            pass

    # 3. Outlier Winsorization / Robust Clipping Experiment
    if outlier_cols:
        exp_counter += 1
        try:
            mod_res = train_and_evaluate_baseline(
                df=df,
                profile=profile,
                target_col=target_col,
                exclude_cols=leakage_cols if leakage_cols else None,
                winsorize=True
            )
            mod_score = float(mod_res["primary_score"] or 0.0)
            ref_score = experiments[-1].modified_score if leakage_cols and len(experiments) > 1 else baseline_score
            delta = round(mod_score - ref_score, 4)
            delta_pct = round((delta / ref_score) * 100, 2) if ref_score != 0 else 0.0

            status = "improved" if delta > 0.005 else ("degraded" if delta < -0.01 else "neutral")
            notes = (
                f"Winsorizing extreme outliers (1st & 99th percentiles) yielded {delta:+.3f} change in {primary_metric}."
            )

            experiments.append(
                ExperimentResult(
                    id=f"EXP-00{exp_counter}",
                    name="Outlier Winsorization (1st/99th Percentile Clip)",
                    category="outliers",
                    description=f"Apply robust percentiles clipping to numerical features with high outlier rates ({', '.join(outlier_cols[:3])}).",
                    model_name=mod_res["model_name"],
                    baseline_metric_name=primary_metric,
                    baseline_score=ref_score,
                    modified_score=mod_score,
                    delta=delta,
                    delta_pct=delta_pct,
                    status=status,
                    notes=notes,
                    detailed_metrics=mod_res["metrics"]
                )
            )
        except Exception as e:
            pass

    # 4. Imputation Strategy Benchmark (Mean vs Median)
    if missing_cols:
        exp_counter += 1
        try:
            mod_res = train_and_evaluate_baseline(
                df=df,
                profile=profile,
                target_col=target_col,
                exclude_cols=leakage_cols if leakage_cols else None,
                impute_strategy="mean"
            )
            mod_score = float(mod_res["primary_score"] or 0.0)
            ref_score = experiments[1].modified_score if leakage_cols and len(experiments) > 1 else baseline_score
            delta = round(mod_score - ref_score, 4)
            delta_pct = round((delta / ref_score) * 100, 2) if ref_score != 0 else 0.0

            status = "improved" if delta > 0.005 else ("degraded" if delta < -0.005 else "neutral")
            notes = (
                f"Comparing Mean vs Median imputation across missing columns {missing_cols[:3]}."
            )

            experiments.append(
                ExperimentResult(
                    id=f"EXP-00{exp_counter}",
                    name="Imputation Strategy: Mean vs Median",
                    category="missingness",
                    description="Evaluate model sensitivity to mean vs robust median imputation.",
                    model_name=mod_res["model_name"],
                    baseline_metric_name=primary_metric,
                    baseline_score=ref_score,
                    modified_score=mod_score,
                    delta=delta,
                    delta_pct=delta_pct,
                    status=status,
                    notes=notes,
                    detailed_metrics=mod_res["metrics"]
                )
            )
        except Exception as e:
            pass

    return experiments
