import numpy as np
import pandas as pd
import pytest
from dataset_autopilot.config import ColumnRole, SemanticType
from dataset_autopilot.profiling import profile_column, profile_dataset
from dataset_autopilot.schema import detect_semantic_type, detect_target_column, infer_column_role


def test_detect_semantic_types():
    email_series = pd.Series(["user1@test.com", "user2@domain.org", "info@company.co"])
    assert detect_semantic_type(email_series) == SemanticType.EMAIL

    numeric_series = pd.Series([10.5, 20.1, 30.8])
    assert detect_semantic_type(numeric_series) == SemanticType.GENERIC_NUMERIC

    bool_series = pd.Series(["true", "false", "true"])
    assert detect_semantic_type(bool_series) == SemanticType.BOOLEAN


def test_infer_column_role():
    id_series = pd.Series([f"ID_{i}" for i in range(100)])
    role = infer_column_role("customer_id", id_series, SemanticType.GENERIC_TEXT)
    assert role == ColumnRole.ID_CANDIDATE

    const_series = pd.Series([1, 1, 1, 1])
    role = infer_column_role("const_col", const_series, SemanticType.GENERIC_NUMERIC)
    assert role == ColumnRole.CONSTANT


def test_detect_target_column():
    df = pd.DataFrame({"age": [20, 30], "salary": [50, 60], "churn": [0, 1]})
    
    # Explicit target
    col, target_type = detect_target_column(df, specified_target="churn")
    assert col == "churn"
    assert target_type == "binary_classification"

    # Unsupervised: target is None without auto-detect
    col_none, type_none = detect_target_column(df, specified_target=None, auto_detect=False)
    assert col_none is None
    assert type_none is None

    # Explicit auto_detect
    col_auto, type_auto = detect_target_column(df, auto_detect=True)
    assert col_auto == "churn"
    assert type_auto == "binary_classification"


def test_profile_dataset():
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(50)],
        "age": np.random.randint(18, 70, size=50),
        "income": np.random.normal(50000, 10000, size=50),
        "status": np.random.choice(["A", "B"], size=50),
        "churn": np.random.choice([0, 1], size=50)
    })
    prof = profile_dataset(df, specified_target="churn")
    assert prof.rows == 50
    assert prof.columns == 5
    assert prof.detected_target == "churn"
    assert prof.columns_profile["user_id"].is_id is True
    assert prof.columns_profile["age"].mean is not None
    assert len(prof.columns_profile["income"].histogram) > 0
