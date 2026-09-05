"""
Data profiling module: computes comprehensive column and dataset statistics.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataset_autopilot.config import ColumnRole
from dataset_autopilot.schema import detect_semantic_type, detect_target_column, infer_column_role
from dataset_autopilot.schemas import (
    ColumnProfile,
    DatasetProfile,
    HistogramBin,
    ValueFrequency,
)
from dataset_autopilot.utils import safe_float, safe_int


def profile_column(
    series: pd.Series,
    col_name: str,
    specified_target: Optional[str] = None
) -> ColumnProfile:
    """Compute detailed profile statistics for a single column."""
    total_len = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(missing_count / total_len, 4) if total_len > 0 else 0.0

    valid_series = series.dropna()
    unique_count = int(valid_series.nunique())
    unique_ratio = round(unique_count / total_len, 4) if total_len > 0 else 0.0

    semantic_type = detect_semantic_type(series)
    inferred_role = infer_column_role(col_name, series, semantic_type, specified_target)

    is_constant = unique_count <= 1
    is_id = inferred_role == ColumnRole.ID_CANDIDATE

    # Top frequent values
    top_val_counts = valid_series.astype(str).value_counts().head(10)
    top_values = [
        ValueFrequency(
            value=str(val)[:100],
            count=int(cnt),
            pct=round(cnt / total_len, 4) if total_len > 0 else 0.0
        )
        for val, cnt in top_val_counts.items()
    ]

    # Sample values
    sample_values = [str(x)[:100] for x in valid_series.unique()[:5]]

    # Numeric statistics & histogram
    min_val = max_val = mean_val = median_val = std_val = None
    p5_val = p25_val = p75_val = p95_val = skew_val = kurt_val = None
    histogram_bins: List[HistogramBin] = []

    if pd.api.types.is_numeric_dtype(series) and not is_constant and len(valid_series) > 0:
        clean_num = valid_series.astype(float)
        min_val = safe_float(clean_num.min())
        max_val = safe_float(clean_num.max())
        mean_val = safe_float(clean_num.mean())
        median_val = safe_float(clean_num.median())
        std_val = safe_float(clean_num.std())
        p5_val = safe_float(clean_num.quantile(0.05))
        p25_val = safe_float(clean_num.quantile(0.25))
        p75_val = safe_float(clean_num.quantile(0.75))
        p95_val = safe_float(clean_num.quantile(0.95))
        skew_val = safe_float(clean_num.skew())
        kurt_val = safe_float(clean_num.kurtosis())

        # Generate 10 histogram bins
        try:
            counts, bin_edges = np.histogram(clean_num, bins=10)
            for i in range(len(counts)):
                histogram_bins.append(
                    HistogramBin(
                        bin_start=round(float(bin_edges[i]), 3),
                        bin_end=round(float(bin_edges[i + 1]), 3),
                        count=int(counts[i]),
                        pct=round(float(counts[i]) / len(clean_num), 4)
                    )
                )
        except Exception:
            pass

    return ColumnProfile(
        name=col_name,
        dtype=str(series.dtype),
        inferred_role=inferred_role,
        semantic_type=semantic_type,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        min=min_val,
        max=max_val,
        mean=mean_val,
        median=median_val,
        std=std_val,
        p5=p5_val,
        p25=p25_val,
        p75=p75_val,
        p95=p95_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        top_values=top_values,
        histogram=histogram_bins,
        is_constant=is_constant,
        is_id=is_id,
        sample_values=sample_values,
    )


def profile_dataset(
    df: pd.DataFrame,
    specified_target: Optional[str] = None
) -> DatasetProfile:
    """Compute complete profile of the dataset and all its columns."""
    rows = len(df)
    columns = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_rows / rows, 4) if rows > 0 else 0.0
    memory_bytes = int(df.memory_usage(deep=True).sum())

    target_col, target_type = detect_target_column(df, specified_target)

    columns_profile: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        columns_profile[col] = profile_column(
            series=df[col],
            col_name=col,
            specified_target=target_col
        )

    return DatasetProfile(
        rows=rows,
        columns=columns,
        duplicate_rows=duplicate_rows,
        duplicate_pct=duplicate_pct,
        memory_bytes=memory_bytes,
        columns_profile=columns_profile,
        detected_target=target_col,
        target_type=target_type
    )
