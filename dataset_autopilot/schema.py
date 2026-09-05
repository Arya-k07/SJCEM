"""
Schema intelligence module: semantic type detection, role inference, and target detection.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataset_autopilot.config import ColumnRole, SemanticType


PATTERNS = {
    SemanticType.EMAIL: re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"),
    SemanticType.PHONE: re.compile(r"^(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}$"),
    SemanticType.SSN: re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    SemanticType.CREDIT_CARD: re.compile(r"^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$"),
    SemanticType.IP_ADDRESS: re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
    SemanticType.URL: re.compile(r"^https?://[^\s]+"),
    SemanticType.UUID: re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    SemanticType.POSTAL_CODE: re.compile(r"^\d{5}(-\d{4})?$"),
    SemanticType.CURRENCY: re.compile(r"^[\$\€\£\¥]\s?\d+([,\.]\d{2})?"),
}

COMMON_TARGET_NAMES = [
    "target", "label", "churn", "is_churn", "default", "is_default",
    "fraud", "is_fraud", "survived", "price", "salary", "outcome",
    "class", "status", "y", "converted", "attrition", "risk", "click"
]

ID_NAME_PATTERNS = re.compile(
    r"(^id$|_id$|^uuid$|^guid$|^key$|^index$|^pk$|customer_id|user_id|order_id|client_id|account_id)",
    re.IGNORECASE
)


def detect_semantic_type(series: pd.Series) -> SemanticType:
    """Detect rich semantic type for a pandas Series using pattern matching and parsing."""
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return SemanticType.GENERIC_TEXT

    # Check for boolean
    if series.dtype == bool:
        return SemanticType.BOOLEAN

    unique_vals_lower = {str(x).strip().lower() for x in valid_series.head(50)}
    if unique_vals_lower.issubset({"true", "false", "0", "1", "yes", "no", "t", "f"}):
        return SemanticType.BOOLEAN

    # Check for numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return SemanticType.GENERIC_NUMERIC

    # Check for datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME_STRING

    # String / Object samples
    sample_vals = [str(x).strip() for x in valid_series.head(100) if str(x).strip()]
    if not sample_vals:
        return SemanticType.GENERIC_TEXT

    # Try regex patterns
    for sem_type, pattern in PATTERNS.items():
        match_count = sum(1 for val in sample_vals if pattern.match(val))
        if match_count / len(sample_vals) >= 0.70:
            return sem_type

    # Try parsing as datetime
    try:
        sample_dates = pd.to_datetime(sample_vals[:20], errors="coerce", format="mixed")
        if sample_dates.notna().sum() / len(sample_vals[:20]) >= 0.80:
            return SemanticType.DATETIME_STRING
    except Exception:
        pass

    # Try parsing as numeric string
    try:
        sample_nums = pd.to_numeric(sample_vals[:20], errors="coerce")
        if sample_nums.notna().sum() / len(sample_vals[:20]) >= 0.80:
            return SemanticType.NUMERIC_STRING
    except Exception:
        pass

    # Check if categorical vs generic text
    if series.nunique() <= 50 or (series.nunique() / len(series)) < 0.05:
        return SemanticType.GENERIC_CATEGORICAL

    return SemanticType.GENERIC_TEXT


def infer_column_role(
    col_name: str,
    series: pd.Series,
    semantic_type: SemanticType,
    specified_target: Optional[str] = None
) -> ColumnRole:
    """Infer the structural machine learning role of a column."""
    if specified_target and col_name.lower() == specified_target.lower():
        return ColumnRole.TARGET

    num_unique = series.nunique()
    total_len = len(series)

    # Constant column
    if num_unique <= 1:
        return ColumnRole.CONSTANT

    # ID Candidate
    unique_ratio = num_unique / total_len if total_len > 0 else 0
    is_float = pd.api.types.is_float_dtype(series)

    if ID_NAME_PATTERNS.search(col_name) and unique_ratio >= 0.80:
        return ColumnRole.ID_CANDIDATE

    if unique_ratio >= 0.98 and not is_float and series.dtype in ["object", "string", "category", "int64", "int32"]:
        # Only object/string or integer with ID-like high cardinality
        if ID_NAME_PATTERNS.search(col_name) or not pd.api.types.is_numeric_dtype(series):
            return ColumnRole.ID_CANDIDATE

    if semantic_type in (SemanticType.UUID, SemanticType.SSN, SemanticType.CREDIT_CARD):
        return ColumnRole.ID_CANDIDATE

    if semantic_type == SemanticType.DATETIME_STRING or pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.DATETIME_FEATURE

    if semantic_type in (SemanticType.GENERIC_NUMERIC, SemanticType.NUMERIC_STRING, SemanticType.CURRENCY):
        return ColumnRole.NUMERIC_FEATURE

    if semantic_type in (SemanticType.GENERIC_CATEGORICAL, SemanticType.BOOLEAN):
        return ColumnRole.CATEGORICAL_FEATURE

    if semantic_type in (SemanticType.EMAIL, SemanticType.URL, SemanticType.IP_ADDRESS, SemanticType.GENERIC_TEXT):
        if num_unique < 50 or unique_ratio < 0.10:
            return ColumnRole.CATEGORICAL_FEATURE
        return ColumnRole.TEXT_FEATURE

    return ColumnRole.CATEGORICAL_FEATURE


PRIORITY_TARGET_NAMES = [
    "target", "label", "churn", "is_churn", "default", "is_default",
    "fraud", "is_fraud", "survived", "outcome", "class", "status",
    "y", "converted", "attrition", "risk", "click"
]

SECONDARY_TARGET_NAMES = ["price", "salary", "revenue", "sales"]


def detect_target_column(
    df: pd.DataFrame,
    specified_target: Optional[str] = None,
    auto_detect: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find target column and infer target type (binary_classification, multiclass, regression).
    Only searches heuristics if auto_detect=True or specified_target is provided.
    """
    target_col = None

    if specified_target:
        for col in df.columns:
            if col.lower() == specified_target.lower():
                target_col = col
                break
        if not target_col and specified_target in df.columns:
            target_col = specified_target

    if not target_col and auto_detect:
        # Search priority target names first
        for name in PRIORITY_TARGET_NAMES:
            for col in df.columns:
                if col.lower() == name:
                    target_col = col
                    break
            if target_col:
                break

        if not target_col:
            # Search secondary target names
            for name in SECONDARY_TARGET_NAMES:
                for col in df.columns:
                    if col.lower() == name:
                        target_col = col
                        break
                if target_col:
                    break

    if not target_col:
        return None, None

    # Infer target task type
    target_series = df[target_col].dropna()
    num_unique = target_series.nunique()

    if num_unique == 2:
        target_type = "binary_classification"
    elif 3 <= num_unique <= 20:
        target_type = "multiclass_classification"
    elif pd.api.types.is_numeric_dtype(target_series):
        target_type = "regression"
    else:
        target_type = "multiclass_classification"

    return target_col, target_type
