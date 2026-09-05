"""
Synthetic Fault-Injection Benchmark Framework:
Generates controlled synthetic datasets with mathematically known ground-truth faults
and benchmarks the Dataset Autopilot detector suite and Decision Lab experiments against them.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataset_autopilot.config import FindingCategory, FindingStatus, Severity
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.schemas import AuditReport


@dataclass
class GroundTruthFault:
    fault_type: str
    affected_columns: List[str]
    injection_parameters: Dict[str, Any]
    expected_detector: str
    expected_status: FindingStatus
    description: str


@dataclass
class BenchmarkDatasetCase:
    case_id: str
    dataset_name: str
    df: pd.DataFrame
    target_col: Optional[str]
    time_col: Optional[str]
    ground_truth: Optional[GroundTruthFault]


@dataclass
class DetectionEvaluation:
    case_id: str
    fault_type: str
    detected: bool
    expected_detector: str
    actual_detector: Optional[str]
    expected_status: Optional[str]
    actual_status: Optional[str]
    false_positives: List[str]
    false_negatives: List[str]
    matched_finding: Optional[Dict[str, Any]]
    notes: str


def generate_benchmark_suite(seed: int = 42) -> List[BenchmarkDatasetCase]:
    """
    Generate clean control and 7 fault-injected synthetic datasets with explicit ground truth.
    """
    np.random.seed(seed)
    n = 200

    # Base features: moderate signal with realistic overlap (base model ROC-AUC ~ 0.72)
    age = np.random.normal(40, 10, size=n).clip(18, 80)
    tenure = np.random.exponential(12, size=n).clip(1, 72)
    monthly = np.random.normal(65, 15, size=n).clip(20, 150)
    contract = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n)

    # Moderate logistic signal with noise
    logits = (age - 40) / 25 - (tenure - 12) / 20 + np.random.normal(0, 1.2, size=n)
    probs = 1 / (1 + np.exp(-logits))
    churn = (probs > 0.5).astype(int)

    base_df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": age,
        "tenure": tenure,
        "monthly_charges": monthly,
        "contract_type": contract,
        "churn": churn
    })

    cases: List[BenchmarkDatasetCase] = []

    # -------------------------------------------------------------
    # 0. Clean Control Dataset (No Faults)
    # -------------------------------------------------------------
    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_0_CLEAN",
            dataset_name="clean_control.csv",
            df=base_df.copy(),
            target_col="churn",
            time_col=None,
            ground_truth=None
        )
    )

    # -------------------------------------------------------------
    # 1. Target Leakage (Direct Leaky Signal)
    # -------------------------------------------------------------
    df_leak = base_df.copy()
    # Injected cancellation code is 98% correlated with churn
    noise = np.random.normal(0, 0.05, size=n)
    df_leak["account_cancellation_code"] = (df_leak["churn"] + noise).clip(0, 1)

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_1_LEAKAGE",
            dataset_name="target_leakage.csv",
            df=df_leak,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="target_leakage",
                affected_columns=["account_cancellation_code"],
                injection_parameters={"correlation_with_target": 0.98},
                expected_detector="target_leakage",
                expected_status=FindingStatus.SUSPECTED,
                description="Feature highly correlated with target causing severe target leakage."
            )
        )
    )

    # -------------------------------------------------------------
    # 2. Train/Test Duplicate Contamination (Exact Duplicate Rows)
    # -------------------------------------------------------------
    df_dup = base_df.copy()
    dup_rows = df_dup.iloc[:30].copy() # 15% duplicate rows
    df_dup = pd.concat([df_dup, dup_rows], ignore_index=True)

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_2_DUPLICATES",
            dataset_name="duplicate_contamination.csv",
            df=df_dup,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="duplicates",
                affected_columns=[],
                injection_parameters={"duplicate_rows_injected": 30, "duplicate_pct": 0.13},
                expected_detector="format_error",
                expected_status=FindingStatus.OBSERVED,
                description="Dataset contains 30 exact duplicate rows."
            )
        )
    )

    # -------------------------------------------------------------
    # 3. Distribution Shift (Covariate Drift)
    # -------------------------------------------------------------
    df_drift = base_df.copy()
    half = n // 2
    # Injected massive shift in monthly_charges in second half (mean 65 -> 140)
    df_drift.loc[half:, "monthly_charges"] = np.random.normal(140, 15, size=half)

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_3_DRIFT",
            dataset_name="distribution_shift.csv",
            df=df_drift,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="distribution_drift",
                affected_columns=["monthly_charges"],
                injection_parameters={"mean_shift": "+75.0", "expected_psi": "> 0.30"},
                expected_detector="distribution_drift",
                expected_status=FindingStatus.OBSERVED,
                description="Severe covariate shift in monthly_charges between partitions."
            )
        )
    )

    # -------------------------------------------------------------
    # 4. Informative Missingness (MNAR - Missing Not At Random)
    # -------------------------------------------------------------
    df_mnar = base_df.copy()
    df_mnar["annual_income"] = np.random.normal(60000, 15000, size=n)
    # Missing only when churn == 1 (50% missing for churners, 0% for non-churners)
    churn_idx = df_mnar[df_mnar["churn"] == 1].index
    drop_idx = churn_idx[:len(churn_idx) // 2]
    df_mnar.loc[drop_idx, "annual_income"] = np.nan

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_4_INFORMATIVE_MISSING",
            dataset_name="informative_missingness.csv",
            df=df_mnar,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="informative_missingness",
                affected_columns=["annual_income"],
                injection_parameters={"mnar_target_link": "churn == 1", "missing_pct": 0.15},
                expected_detector="missingness",
                expected_status=FindingStatus.SUSPECTED,
                description="Missingness in annual_income statistically dependent on churn outcome."
            )
        )
    )

    # -------------------------------------------------------------
    # 5. Extreme Outliers (> 5.0x IQR)
    # -------------------------------------------------------------
    df_outliers = base_df.copy()
    df_outliers["account_balance"] = np.random.normal(5000, 1000, size=n)
    # Inject 12 extreme outliers with magnitude 100,000+
    outlier_idx = np.random.choice(n, size=12, replace=False)
    df_outliers.loc[outlier_idx, "account_balance"] = np.random.uniform(90000, 250000, size=12)

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_5_OUTLIERS",
            dataset_name="extreme_outliers.csv",
            df=df_outliers,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="extreme_outliers",
                affected_columns=["account_balance"],
                injection_parameters={"outlier_count": 12, "iqr_multiplier": "> 5.0"},
                expected_detector="outliers",
                expected_status=FindingStatus.OBSERVED,
                description="12 extreme outlier values far exceeding normal IQR range."
            )
        )
    )

    # -------------------------------------------------------------
    # 6. Subgroup Slice Disparity
    # -------------------------------------------------------------
    df_slice = base_df.copy()
    # 3 slices with adequate sample size (80, 80, 40)
    df_slice["service_region"] = np.random.choice(["East", "West", "North"], size=n, p=[0.40, 0.40, 0.20])
    # For North region, invert target to inject high error/disparity
    north_mask = df_slice["service_region"] == "North"
    df_slice.loc[north_mask, "churn"] = 1 - df_slice.loc[north_mask, "churn"]

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_6_SLICE_DISPARITY",
            dataset_name="slice_bias.csv",
            df=df_slice,
            target_col="churn",
            time_col=None,
            ground_truth=GroundTruthFault(
                fault_type="slice_bias",
                affected_columns=["service_region"],
                injection_parameters={"minority_slice": "North", "disparity": "Inverted error distribution"},
                expected_detector="slice_bias",
                expected_status=FindingStatus.OBSERVED,
                description="Significant performance degradation on minority North slice."
            )
        )
    )

    # -------------------------------------------------------------
    # 7. Post-Outcome Temporal Feature
    # -------------------------------------------------------------
    df_temporal = base_df.copy()
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    df_temporal["event_timestamp"] = dates.strftime("%Y-%m-%d")
    # Post-churn refund occurs only after churn occurs
    df_temporal["post_event_refund"] = np.where(df_temporal["churn"] == 1, np.random.uniform(50, 200, size=n), 0.0)

    cases.append(
        BenchmarkDatasetCase(
            case_id="BENCH_7_TEMPORAL_LEAK",
            dataset_name="post_outcome_temporal.csv",
            df=df_temporal,
            target_col="churn",
            time_col="event_timestamp",
            ground_truth=GroundTruthFault(
                fault_type="target_leakage",
                affected_columns=["post_event_refund"],
                injection_parameters={"temporal_post_event": True},
                expected_detector="target_leakage",
                expected_status=FindingStatus.SUSPECTED,
                description="Post-outcome event feature creating temporal data leakage."
            )
        )
    )

    return cases


def run_benchmark_evaluation(engine: Optional[DatasetAutopilotEngine] = None) -> List[DetectionEvaluation]:
    """
    Execute audit engine against all synthetic benchmark datasets and evaluate detection accuracy.
    """
    if engine is None:
        engine = DatasetAutopilotEngine()

    cases = generate_benchmark_suite()
    evaluations: List[DetectionEvaluation] = []

    for case in cases:
        report = engine.run_audit(
            df=case.df,
            dataset_name=case.dataset_name,
            target_col=case.target_col
        )

        if case.ground_truth is None:
            # Clean Control Evaluation
            critical_findings = [f for f in report.findings if f.severity == Severity.CRITICAL]
            leakage_findings = [f for f in report.findings if f.type == FindingCategory.TARGET_LEAKAGE]
            
            fps = [f"{f.type.value}:{f.column}" for f in (critical_findings + leakage_findings)]
            evaluations.append(
                DetectionEvaluation(
                    case_id=case.case_id,
                    fault_type="clean_control",
                    detected=len(fps) == 0,
                    expected_detector="none",
                    actual_detector=fps[0] if fps else "none",
                    expected_status="CLEAN",
                    actual_status="CLEAN" if len(fps) == 0 else "FALSE_POSITIVE",
                    false_positives=fps,
                    false_negatives=[],
                    matched_finding=None,
                    notes="Clean control baseline. Zero critical/leakage false positives expected."
                )
            )
            continue

        gt = case.ground_truth
        # Find matching finding in report
        matched = None
        for f in report.findings:
            if f.type.value == gt.expected_detector:
                if not gt.affected_columns or (f.column in gt.affected_columns):
                    matched = f
                    break

        detected = matched is not None
        status_match = (matched.status == gt.expected_status) if matched else False
        
        fns = gt.affected_columns if not detected else []
        fps = []

        notes = f"Ground truth: {gt.description} | Status: {matched.status.value if matched else 'NOT DETECTED'}"

        evaluations.append(
            DetectionEvaluation(
                case_id=case.case_id,
                fault_type=gt.fault_type,
                detected=detected,
                expected_detector=gt.expected_detector,
                actual_detector=matched.type.value if matched else None,
                expected_status=gt.expected_status.value,
                actual_status=matched.status.value if matched else None,
                false_positives=fps,
                false_negatives=fns,
                matched_finding=matched.__dict__ if matched else None,
                notes=notes
            )
        )

    return evaluations
