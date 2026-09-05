"""
Master Autopilot Orchestration Engine: coordinates profiling, schema inference,
quality analysis, leakage detection, drift testing, bias evaluation, baseline modeling,
controlled experiments, and report assembly.
"""

from datetime import datetime, timezone
import time
from typing import Callable, Dict, List, Optional
import uuid
import numpy as np
import pandas as pd
from dataset_autopilot.bias import analyze_slice_bias
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    Severity,
)
from dataset_autopilot.decision_planner import plan_decision_candidates
from dataset_autopilot.drift import analyze_drift
from dataset_autopilot.experiments import plan_and_run_experiments
from dataset_autopilot.leakage import (
    analyze_target_leakage,
    analyze_train_test_contamination,
)
from dataset_autopilot.models import train_and_evaluate_baseline
from dataset_autopilot.profiling import profile_dataset
from dataset_autopilot.quality import analyze_data_quality
from dataset_autopilot.reporting import (
    compute_health_scores,
    generate_executive_summary_deterministic,
    generate_executive_summary_gemini,
)
from dataset_autopilot.schemas import (
    AnalysisContext,
    AuditReport,
    DecisionCandidate,
    DatasetProfile,
    DriftMetric,
    ExperimentResult,
    Finding,
    SliceMetric,
)


class DatasetAutopilotEngine:
    """Orchestrates autonomous audit of tabular datasets."""

    def __init__(self, config: AutopilotConfig = DEFAULT_CONFIG):
        self.config = config

    def run_audit(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset.csv",
        target_col: Optional[str] = None,
        split_col: Optional[str] = None,
        time_col: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
        gemini_api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> AuditReport:
        """Execute full autonomous data audit on the given DataFrame."""
        report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        api_key = gemini_api_key or self.config.gemini_api_key

        def notify(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        # Stage 1: Ingestion & Validation (10%)
        notify(10, "Validating dataset structure and schema...")
        if len(df) == 0 or len(df.columns) == 0:
            raise ValueError("Dataset is empty. Cannot perform audit.")

        # Stage 2: Profiling (25%)
        notify(25, "Computing descriptive statistics, semantic types, and distributions...")
        effective_target = (context.target_column if context and context.target_column else target_col)
        profile: DatasetProfile = profile_dataset(df, specified_target=effective_target)
        detected_target = profile.detected_target

        all_findings: List[Finding] = []

        # Stage 3: Data Quality Checks (40%)
        notify(40, "Evaluating missingness patterns, informativeness, and outliers...")
        quality_findings = analyze_data_quality(df, profile, target_col=detected_target, config=self.config)
        all_findings.extend(quality_findings)

        # Stage 4: Leakage & Contamination (55%)
        notify(55, "Scanning for target leakage, proxies, and train-test contamination...")
        if detected_target:
            leakage_findings = analyze_target_leakage(df, profile, target_col=detected_target, config=self.config)
            all_findings.extend(leakage_findings)

        contamination_findings = analyze_train_test_contamination(df, profile, split_col=split_col)
        all_findings.extend(contamination_findings)

        # Stage 5: Distribution Drift (70%)
        notify(70, "Computing Population Stability Index (PSI) and Kolmogorov-Smirnov shift...")
        drift_metrics, drift_findings = analyze_drift(
            df=df,
            profile=profile,
            target_col=detected_target,
            split_col=split_col,
            time_col=time_col,
            config=self.config
        )
        all_findings.extend(drift_findings)

        # Stage 6: Baseline Oracle Modeling & Bias Slices (85%)
        notify(85, "Training baseline oracle models and evaluating demographic slices...")
        experiments: List[ExperimentResult] = []
        slice_metrics: List[SliceMetric] = []

        if detected_target and len(df) >= 30:
            try:
                base_result = train_and_evaluate_baseline(
                    df=df,
                    profile=profile,
                    target_col=detected_target
                )

                # Slice & Bias analysis using baseline out-of-fold predictions
                slice_metrics, bias_findings = analyze_slice_bias(
                    df=df,
                    profile=profile,
                    target_col=detected_target,
                    y_true=base_result["y_true"],
                    y_pred=base_result["y_pred"],
                    is_classification=base_result["is_classification"],
                    config=self.config
                )
                all_findings.extend(bias_findings)

                # Stage 7: Remediation Experiments
                notify(90, "Executing controlled remediation A/B experiments...")
                experiments = plan_and_run_experiments(
                    df=df,
                    profile=profile,
                    target_col=detected_target,
                    findings=all_findings,
                    baseline_result=base_result
                )
            except Exception:
                pass

        # Stage 8: Synthesize Decision Candidates
        notify(92, "Formulating prioritized Decision Candidates...")
        decision_candidates = plan_decision_candidates(
            profile=profile,
            findings=all_findings,
            context=context,
            df=df
        )

        num_decisions = len(decision_candidates)
        if num_decisions > 0:
            readiness_text = f"{num_decisions} decision{'s' if num_decisions > 1 else ''} may materially affect model evaluation."
        elif not detected_target:
            readiness_text = "Unsupervised dataset. Supervised decisions suppressed."
        else:
            readiness_text = "No high-impact data decisions detected."

        # Stage 9: Health Score & Report Synthesis (95%)
        notify(95, "Synthesizing composite score and executive summary...")
        score_breakdown = compute_health_scores(
            findings=all_findings,
            has_target=bool(detected_target),
            config=self.config
        )

        # Try Gemini API summary first if key present, fallback to deterministic
        exec_summary = None
        if api_key:
            exec_summary = generate_executive_summary_gemini(
                profile=profile,
                findings=all_findings,
                score_breakdown=score_breakdown,
                experiments=experiments,
                api_key=api_key,
                model_name=self.config.gemini_model_name
            )

        if not exec_summary:
            exec_summary = generate_executive_summary_deterministic(
                profile=profile,
                findings=all_findings,
                score_breakdown=score_breakdown,
                experiments=experiments
            )

        notify(100, "Audit completed successfully.")

        report = AuditReport(
            report_id=report_id,
            dataset_name=dataset_name,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            health_score=score_breakdown.overall_score,
            health_grade=score_breakdown.grade,
            decision_readiness=readiness_text,
            score_breakdown=score_breakdown,
            dataset_profile=profile,
            findings=all_findings,
            decision_candidates=decision_candidates,
            experiments=experiments,
            drift_summary=drift_metrics,
            slice_summary=slice_metrics,
            executive_summary=exec_summary,
            analysis_context=context
        )

        return report
