"""
Decision Planner: Converts raw dataset observations into prioritized Decision Candidates
using a strict 4-part filter: Actionability, Materiality, Available Alternatives, and Valid Experiment Design.
Structures candidates with explicit evidence blocks: Observation, Evidence, What We Know, What Is Unknown,
and Controlled Parameters.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from dataset_autopilot.config import (
    ColumnRole,
    DecisionStatus,
    FindingCategory,
    FindingStatus,
    PrimaryObjective,
)
from dataset_autopilot.schemas import (
    AnalysisContext,
    DatasetProfile,
    DecisionCandidate,
    DecisionEvidenceBlock,
    Finding,
)


def plan_decision_candidates(
    profile: DatasetProfile,
    findings: List[Finding],
    context: Optional[AnalysisContext] = None,
    df: Optional[pd.DataFrame] = None
) -> List[DecisionCandidate]:
    """
    Synthesize 2 to 4 prioritized DecisionCandidates from raw findings.
    Ensures non-actionable observations remain plain observations.
    """
    candidates: List[DecisionCandidate] = []
    
    # Resolve target and task
    target_col = context.target_column if context and context.target_column else profile.detected_target
    task_type = context.task_type if context and context.task_type else (
        "unsupervised" if not target_col else (
            "regression" if profile.target_type == "regression" else "classification"
        )
    )
    is_supervised = bool(target_col and task_type != "unsupervised")
    if not is_supervised:
        # Supervised decision candidates and model A/B comparisons are unavailable without a target
        return []

    objective = context.primary_objective if context else PrimaryObjective.PREDICTIVE_PERFORMANCE
    candidate_id_counter = 1
    total_rows = profile.rows
    target_metric = "ROC-AUC" if profile.target_type in ["binary_classification", "multiclass_classification"] else "R² Score"

    controlled_ledger = {
        "Model Architecture": "Random Forest (n_estimators=50, max_depth=6)",
        "Validation Protocol": "5-Fold Cross-Validation (Seed=42)",
        "Fold Isolation": "Preprocessing fitted strictly inside training folds",
        "Primary Metric": target_metric,
        "Controlled Change": "Treatment Strategy Variant Only"
    }

    # -------------------------------------------------------------
    # 1. Supervised Target Leakage Candidates
    # -------------------------------------------------------------
    leakage_findings = [
        f for f in findings
        if f.type == FindingCategory.TARGET_LEAKAGE and f.column and f.status == FindingStatus.SUSPECTED
    ]
    for f in leakage_findings:
        col = f.column
        if not col:
            continue

        metric_name = f.evidence.get("single_feature_metric", "AUC")
        metric_score = f.evidence.get("single_feature_score")
        score_str = f"{metric_score:.3f}" if metric_score is not None else "High"

        ev_block = DecisionEvidenceBlock(
            observation=f"Feature '{col}' exhibits near-complete predictive dominance ({metric_name} = {score_str}).",
            empirical_evidence=f"Single-feature model achieved {score_str} {metric_name} across {total_rows:,} records.",
            what_we_know=[
                f"Feature '{col}' alone achieves exceptional predictive association with '{target_col}'.",
                "High predictive dominance in tabular features frequently indicates post-event data logging or target proxy variables.",
                "No causal proof exists from observational correlation alone."
            ],
            what_is_unknown=[
                f"Whether '{col}' is genuinely available at production inference time.",
                f"How much true generalization performance drops when '{col}' is removed."
            ],
            controlled_parameters=controlled_ledger
        )

        candidates.append(
            DecisionCandidate(
                decision_id=f"DEC-{candidate_id_counter:03d}",
                finding_ids=[f.id],
                category="suspected_leakage",
                question=f"How much of the model's apparent performance depends on suspicious feature '{col}'?",
                affected_columns=[col],
                rationale=(
                    f"Feature '{col}' exhibits high predictive dominance ({metric_name} = {score_str}). "
                    f"Controlled ablation testing isolates genuine out-of-fold generalization from target leakage."
                ),
                expected_impact="HIGH",
                evidence=f.evidence,
                evidence_block=ev_block,
                available_strategies=["BASELINE_WITH_FEATURE", "REMOVE_SUSPECT_FEATURE"],
                confidence=0.90,
                status=DecisionStatus.EXPERIMENT_READY
            )
        )
        candidate_id_counter += 1
        if len(candidates) >= 2:
            break

    # -------------------------------------------------------------
    # 2. Missingness Strategy Candidates
    # -------------------------------------------------------------
    missing_findings = [
        f for f in findings
        if f.type == FindingCategory.MISSINGNESS and f.column
    ]
    missing_findings.sort(
        key=lambda x: (
            1 if x.status == FindingStatus.SUSPECTED else 0,
            x.evidence.get("missing_pct", 0)
        ),
        reverse=True
    )

    for f in missing_findings:
        col = f.column
        if not col:
            continue
        missing_pct = f.evidence.get("missing_pct", 0.0)
        missing_count = f.evidence.get("missing_count", int(missing_pct * total_rows))
        is_informative = f.evidence.get("is_informative", False)

        # Filter: Only actionable if missingness is between 2% and 85%
        if 0.02 <= missing_pct <= 0.85:
            strategies = ["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"]

            impact = "HIGH" if (is_informative or missing_pct > 0.20) else "MEDIUM"
            if objective == PrimaryObjective.SIMPLICITY:
                impact = "HIGH"

            ev_block = DecisionEvidenceBlock(
                observation=f"{missing_pct * 100:.1f}% of values in '{col}' are missing ({missing_count:,} / {total_rows:,} rows).",
                empirical_evidence=(
                    f"{missing_count:,} missing rows out of {total_rows:,} total rows. "
                    f"{'Statistically significant association detected between missingness indicator and target.' if is_informative else 'Central tendency imputation vs feature drop available.'}"
                ),
                what_we_know=[
                    f"Missingness affects {missing_pct * 100:.1f}% of observations in '{col}'.",
                    f"{'Missingness exhibits potential informative (MNAR) pattern linked to target/covariates.' if is_informative else 'Standard central imputation or ablation is required prior to training.'}",
                    "No data leakage has been confirmed."
                ],
                what_is_unknown=[
                    f"Whether retaining '{col}' provides net positive generalization value compared with dropping it.",
                    f"Whether appending a binary missingness indicator preserves valuable signal without overfitting."
                ],
                controlled_parameters=controlled_ledger
            )

            candidates.append(
                DecisionCandidate(
                    decision_id=f"DEC-{candidate_id_counter:03d}",
                    finding_ids=[f.id],
                    category="missingness_strategy",
                    question=f"Does retaining '{col}' with explicit missingness handling outperform dropping the feature?",
                    affected_columns=[col],
                    rationale=(
                        f"{missing_pct * 100:.1f}% of '{col}' is missing. "
                        f"{'Statistical association with target/covariates suggests potential informative missingness (MNAR). ' if is_informative else ''}"
                        f"Controlled comparison will determine if the feature provides net positive predictive value."
                    ),
                    expected_impact=impact,
                    evidence=f.evidence,
                    evidence_block=ev_block,
                    available_strategies=strategies,
                    confidence=0.85,
                    status=DecisionStatus.EXPERIMENT_READY
                )
            )
            candidate_id_counter += 1
            if len(candidates) >= 4:
                break

    # -------------------------------------------------------------
    # 3. Multicollinearity & Feature Redundancy Candidates
    # -------------------------------------------------------------
    if is_supervised and df is not None and len(df) >= 30:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        checked_pairs = set()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                if (c1, c2) in checked_pairs:
                    continue
                checked_pairs.add((c1, c2))

                s1 = df[c1].dropna()
                s2 = df[c2].dropna()
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) > 30:
                    try:
                        r = float(np.corrcoef(df.loc[common_idx, c1], df.loc[common_idx, c2])[0, 1])
                        if abs(r) >= 0.85:
                            ev_block = DecisionEvidenceBlock(
                                observation=f"Features '{c1}' and '{c2}' are strongly collinear (|r| = {abs(r):.3f}).",
                                empirical_evidence=f"Pairwise Pearson correlation coefficient r = {r:.3f} computed over {len(common_idx):,} common records.",
                                what_we_know=[
                                    f"Features '{c1}' and '{c2}' share {r**2 * 100:.1f}% mutual variance.",
                                    "High collinearity inflates model complexity and hampers feature importance interpretability."
                                ],
                                what_is_unknown=[
                                    f"Whether pruning redundant feature '{c2}' simplifies the pipeline without sacrificing validation {target_metric}."
                                ],
                                controlled_parameters=controlled_ledger
                            )

                            candidates.append(
                                DecisionCandidate(
                                    decision_id=f"DEC-{candidate_id_counter:03d}",
                                    finding_ids=[],
                                    category="feature_redundancy",
                                    question=f"Does pruning collinear feature '{c2}' preserve performance while simplifying the pipeline?",
                                    affected_columns=[c1, c2],
                                    rationale=(
                                        f"Features '{c1}' and '{c2}' are strongly collinear (r = {r:.3f}). "
                                        f"Removing redundant dimensions reduces pipeline complexity and improves interpretability."
                                    ),
                                    expected_impact="HIGH" if objective in [PrimaryObjective.SIMPLICITY, PrimaryObjective.INTERPRETABILITY] else "MEDIUM",
                                    evidence={"feature_1": c1, "feature_2": c2, "correlation": round(r, 3)},
                                    evidence_block=ev_block,
                                    available_strategies=["KEEP_ALL_CORRELATED", "PRUNE_CORRELATED_GROUP"],
                                    confidence=0.85,
                                    status=DecisionStatus.EXPERIMENT_READY
                                )
                            )
                            candidate_id_counter += 1
                            break
                    except Exception:
                        pass
            if len(candidates) >= 4:
                break

    # -------------------------------------------------------------
    # 4. Outlier Robustness Candidate
    # -------------------------------------------------------------
    if is_supervised and len(candidates) < 4:
        outlier_findings = [f for f in findings if f.type == FindingCategory.OUTLIERS and f.column]
        if outlier_findings:
            f = outlier_findings[0]
            col = f.column
            if col:
                extreme_count = f.evidence.get("extreme_outliers_count", 0)
                extreme_pct = f.evidence.get("extreme_pct", 0.0)

                ev_block = DecisionEvidenceBlock(
                    observation=f"Column '{col}' exhibits {extreme_count:,} values beyond 3.0x IQR bounds ({extreme_pct * 100:.2f}% of rows).",
                    empirical_evidence=f"{extreme_count:,} values outside IQR bounds: [q25={f.evidence.get('q25')}, q75={f.evidence.get('q75')}].",
                    what_we_know=[
                        f"Extreme tail distribution observed in '{col}'.",
                        "Unclipped extreme outliers can distort tree splits and linear gradients."
                    ],
                    what_is_unknown=[
                        f"Whether percentile winsorization (1st/99th clip) improves out-of-fold generalization stability."
                    ],
                    controlled_parameters=controlled_ledger
                )

                candidates.append(
                    DecisionCandidate(
                        decision_id=f"DEC-{candidate_id_counter:03d}",
                        finding_ids=[f.id],
                        category="outlier_treatment",
                        question=f"Does robust clipping (winsorization) of extreme values in '{col}' improve validation stability?",
                        affected_columns=[col],
                        rationale=f"Observed extreme outliers beyond 3.0x IQR in '{col}'. Testing if percentile winsorization stabilizes model variance.",
                        expected_impact="MEDIUM",
                        evidence=f.evidence,
                        evidence_block=ev_block,
                        available_strategies=["RAW_FEATURES", "WINSORIZE_OUTLIERS"],
                        confidence=0.80,
                        status=DecisionStatus.EXPERIMENT_READY
                    )
                )
                candidate_id_counter += 1

    return candidates[:4]
