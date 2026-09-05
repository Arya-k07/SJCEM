"""
Data Quality Engine: analyzes missingness patterns, MNAR/informative missingness,
outliers (IQR, z-score), type inconsistencies, constant columns, and duplicates.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    FindingCategory,
    FindingStatus,
    Severity,
)
from dataset_autopilot.schemas import DatasetProfile, Finding
from dataset_autopilot.utils import compute_cramers_v, safe_float


def analyze_data_quality(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: Optional[str] = None,
    config: AutopilotConfig = DEFAULT_CONFIG
) -> List[Finding]:
    """Analyze dataframe for data quality issues and return a list of observations."""
    findings: List[Finding] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    finding_counter = 100

    # 1. Check duplicate rows
    if profile.duplicate_rows > 0:
        finding_counter += 1
        findings.append(
            Finding(
                id=f"F-Q{finding_counter}",
                type=FindingCategory.FORMAT_ERROR,
                status=FindingStatus.OBSERVED,
                severity=Severity.WARNING if profile.duplicate_pct > 0.05 else Severity.INFO,
                column=None,
                title="Duplicate Rows Observed",
                description=f"Dataset contains {profile.duplicate_rows} duplicate rows ({profile.duplicate_pct * 100:.2f}% of total rows).",
                evidence={
                    "duplicate_count": profile.duplicate_rows,
                    "duplicate_pct": profile.duplicate_pct,
                    "total_rows": total_rows
                },
                decision_question="Should exact duplicate rows be deduplicated to prevent sample overweighting in evaluation?",
                recommendation="Review whether duplicate records represent distinct real-world events or duplicate data entries.",
                confidence=1.0
            )
        )

    # 2. Analyze each column
    for col_name, col_prof in profile.columns_profile.items():
        series = df[col_name]

        # 2a. Constant columns
        if col_prof.is_constant:
            finding_counter += 1
            findings.append(
                Finding(
                    id=f"F-Q{finding_counter}",
                    type=FindingCategory.CONSTANT_COLUMN,
                    status=FindingStatus.OBSERVED,
                    severity=Severity.INFO,
                    column=col_name,
                    title=f"Constant Value Observed: '{col_name}'",
                    description=f"Column '{col_name}' has only {col_prof.unique_count} distinct value across {total_rows} rows.",
                    evidence={
                        "unique_count": col_prof.unique_count,
                        "sample_value": col_prof.sample_values[0] if col_prof.sample_values else None
                    },
                    decision_question=None,
                    recommendation="Constant columns provide zero variance and can be excluded from modeling pipelines to reduce complexity.",
                    confidence=1.0
                )
            )
            continue

        # 2b. Missingness Analysis
        if col_prof.missing_count > 0:
            missing_pct = col_prof.missing_pct

            # Check for Informative Missingness / MNAR (Missing Not At Random)
            is_null_series = series.isna().astype(int)
            informative_target_link = None
            max_cramers = 0.0
            correlated_feature = None

            # Test correlation of missing indicator with Target
            if target_col and target_col in df.columns and target_col != col_name:
                target_series = df[target_col].dropna()
                common_idx = target_series.index.intersection(is_null_series.index)
                if len(common_idx) > 20:
                    try:
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            corr, pval = stats.pointbiserialr(is_null_series.loc[common_idx], df.loc[common_idx, target_col])
                            if pval < config.informative_missing_p_val and abs(corr) > 0.15:
                                informative_target_link = {
                                    "target": target_col,
                                    "correlation": safe_float(corr),
                                    "p_value": safe_float(pval)
                                }
                        else:
                            cv = compute_cramers_v(is_null_series.loc[common_idx], df.loc[common_idx, target_col])
                            if cv > 0.15:
                                informative_target_link = {
                                    "target": target_col,
                                    "cramers_v": safe_float(cv)
                                }
                    except Exception:
                        pass

            # Test correlation of missing indicator with other features
            for other_col in df.columns:
                if other_col == col_name or other_col == target_col:
                    continue
                other_s = df[other_col].dropna()
                if len(other_s) < 20:
                    continue
                common_idx = other_s.index.intersection(is_null_series.index)
                cv = compute_cramers_v(is_null_series.loc[common_idx], df.loc[common_idx, other_col].astype(str))
                if cv > max_cramers:
                    max_cramers = cv
                    correlated_feature = other_col

            is_informative = bool(informative_target_link or max_cramers > 0.20)
            evidence = {
                "missing_count": col_prof.missing_count,
                "missing_pct": missing_pct,
                "is_informative": is_informative,
            }
            if informative_target_link:
                evidence["target_association"] = informative_target_link
            if max_cramers > 0.20 and correlated_feature:
                evidence["feature_association"] = {
                    "feature": correlated_feature,
                    "cramers_v": safe_float(max_cramers)
                }

            desc = f"Column '{col_name}' has {col_prof.missing_count} missing values ({missing_pct * 100:.1f}% of total)."
            if is_informative:
                desc += " Statistical association detected between missingness indicator and other attributes, suggesting potential informative missingness (MNAR)."

            status = FindingStatus.SUSPECTED if is_informative else FindingStatus.OBSERVED
            sev = Severity.WARNING if (missing_pct >= 0.10 or is_informative) else Severity.INFO

            finding_counter += 1
            findings.append(
                Finding(
                    id=f"F-Q{finding_counter}",
                    type=FindingCategory.MISSINGNESS,
                    status=status,
                    severity=sev,
                    column=col_name,
                    title=f"Missing Values in '{col_name}' ({missing_pct * 100:.1f}%)",
                    description=desc,
                    evidence=evidence,
                    decision_question=f"Does retaining '{col_name}' with explicit missingness handling outperform dropping the feature?",
                    recommendation="Compare dropping the feature vs central imputation vs indicator encoding in a controlled experiment.",
                    confidence=0.90 if is_informative else 0.80
                )
            )

        # 2c. Outlier Analysis (for numeric columns)
        if pd.api.types.is_numeric_dtype(series) and not col_prof.is_id and col_prof.min is not None:
            clean_num = series.dropna()
            if len(clean_num) >= 30:
                q25 = clean_num.quantile(0.25)
                q75 = clean_num.quantile(0.75)
                iqr = q75 - q25

                if iqr > 0:
                    mild_lower = q25 - config.outlier_iqr_mild_multiplier * iqr
                    mild_upper = q75 + config.outlier_iqr_mild_multiplier * iqr
                    extreme_lower = q25 - config.outlier_iqr_extreme_multiplier * iqr
                    extreme_upper = q75 + config.outlier_iqr_extreme_multiplier * iqr

                    mild_outliers = int(((clean_num < mild_lower) | (clean_num > mild_upper)).sum())
                    extreme_outliers = int(((clean_num < extreme_lower) | (clean_num > extreme_upper)).sum())
                    extreme_pct = extreme_outliers / len(clean_num)

                    # Z-score check
                    mean = clean_num.mean()
                    std = clean_num.std()
                    z_outliers = 0
                    if std > 0:
                        z_scores = np.abs((clean_num - mean) / std)
                        z_outliers = int((z_scores > config.outlier_zscore_threshold).sum())

                    if extreme_pct >= config.outlier_pct_warning or (mild_outliers / len(clean_num)) > 0.08:
                        finding_counter += 1
                        findings.append(
                            Finding(
                                id=f"F-Q{finding_counter}",
                                type=FindingCategory.OUTLIERS,
                                status=FindingStatus.OBSERVED,
                                severity=Severity.WARNING if extreme_pct > 0.05 else Severity.INFO,
                                column=col_name,
                                title=f"Outlier Tail Distribution in '{col_name}'",
                                description=f"Column '{col_name}' exhibits {extreme_outliers} values beyond 3.0x IQR bounds ({extreme_pct * 100:.2f}% of rows).",
                                evidence={
                                    "extreme_outliers_count": extreme_outliers,
                                    "extreme_pct": safe_float(extreme_pct),
                                    "mild_outliers_count": mild_outliers,
                                    "zscore_outliers_count": z_outliers,
                                    "q25": safe_float(q25),
                                    "q75": safe_float(q75),
                                    "iqr": safe_float(iqr),
                                    "extreme_lower_bound": safe_float(extreme_lower),
                                    "extreme_upper_bound": safe_float(extreme_upper),
                                },
                                decision_question=f"Does robust treatment (e.g. clipping/winsorization) of '{col_name}' improve validation stability?",
                                recommendation="Evaluate whether extreme values reflect valid rare events or measurement distortion.",
                                confidence=0.85
                            )
                        )

        # 2d. High Cardinality Categorical
        if not col_prof.is_id and col_prof.dtype in ["object", "category", "string"]:
            if col_prof.unique_count > config.high_cardinality_unique_count and col_prof.unique_ratio < 0.85:
                finding_counter += 1
                findings.append(
                    Finding(
                        id=f"F-Q{finding_counter}",
                        type=FindingCategory.HIGH_CARDINALITY,
                        status=FindingStatus.OBSERVED,
                        severity=Severity.INFO,
                        column=col_name,
                        title=f"High Cardinality Categories in '{col_name}'",
                        description=f"Column '{col_name}' contains {col_prof.unique_count} distinct categories across {total_rows} rows.",
                        evidence={
                            "unique_count": col_prof.unique_count,
                            "unique_ratio": col_prof.unique_ratio
                        },
                        decision_question=f"Does category frequency grouping reduce pipeline complexity without sacrificing predictive performance?",
                        recommendation="Consider frequency grouping or target encoding if cardinality causes high-dimensional sparsity.",
                        confidence=0.85
                    )
                )

    return findings
