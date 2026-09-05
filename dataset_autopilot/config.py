"""
Configuration, thresholds, and constants for Dataset Autopilot.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import os


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class FindingStatus(str, Enum):
    OBSERVED = "OBSERVED"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    NOT_ASSESSED = "NOT_ASSESSED"


class DecisionStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INVESTIGATING = "INVESTIGATING"
    EXPERIMENT_READY = "EXPERIMENT_READY"
    EXPERIMENT_RUNNING = "EXPERIMENT_RUNNING"
    EVIDENCE_AVAILABLE = "EVIDENCE_AVAILABLE"
    USER_DECISION = "USER_DECISION"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"


class PrimaryObjective(str, Enum):
    PREDICTIVE_PERFORMANCE = "predictive_performance"
    GENERALIZATION = "generalization"
    INTERPRETABILITY = "interpretability"
    SIMPLICITY = "simplicity"
    DATA_RETENTION = "data_retention"
    EFFICIENCY = "efficiency"


class FindingCategory(str, Enum):
    TARGET_LEAKAGE = "target_leakage"
    TRAIN_TEST_CONTAMINATION = "train_test_contamination"
    MISSINGNESS = "missingness"
    OUTLIERS = "outliers"
    FORMAT_ERROR = "format_error"
    DISTRIBUTION_DRIFT = "distribution_drift"
    SLICE_BIAS = "slice_bias"
    CONSTANT_COLUMN = "constant_column"
    HIGH_CARDINALITY = "high_cardinality"
    MULTICOLLINEARITY = "multicollinearity"


class ColumnRole(str, Enum):
    ID_CANDIDATE = "id_candidate"
    TARGET = "target"
    NUMERIC_FEATURE = "numeric_feature"
    CATEGORICAL_FEATURE = "categorical_feature"
    DATETIME_FEATURE = "datetime_feature"
    TEXT_FEATURE = "text_feature"
    CONSTANT = "constant"


class SemanticType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    POSTAL_CODE = "postal_code"
    CURRENCY = "currency"
    UUID = "uuid"
    DATETIME_STRING = "datetime_string"
    NUMERIC_STRING = "numeric_string"
    GENERIC_TEXT = "generic_text"
    GENERIC_CATEGORICAL = "generic_categorical"
    GENERIC_NUMERIC = "generic_numeric"
    BOOLEAN = "boolean"


@dataclass
class AutopilotConfig:
    # Target Leakage thresholds
    leakage_auc_critical: float = 0.94
    leakage_auc_warning: float = 0.88
    leakage_corr_critical: float = 0.85
    leakage_corr_warning: float = 0.70
    leakage_mi_threshold: float = 0.50

    # Missingness thresholds
    missing_pct_critical: float = 0.40
    missing_pct_warning: float = 0.05
    informative_missing_p_val: float = 0.01

    # Outlier thresholds
    outlier_pct_warning: float = 0.03
    outlier_zscore_threshold: float = 3.0
    outlier_iqr_extreme_multiplier: float = 3.0
    outlier_iqr_mild_multiplier: float = 1.5

    # Drift thresholds
    drift_psi_critical: float = 0.25
    drift_psi_warning: float = 0.10
    drift_ks_p_val: float = 0.05
    drift_bins: int = 10

    # Slice / Fairness thresholds
    slice_min_sample_size: int = 25
    slice_disparity_threshold: float = 0.15  # 15% drop from overall
    disparate_impact_lower: float = 0.80
    disparate_impact_upper: float = 1.25

    # Schema & Cardinality
    id_unique_ratio_threshold: float = 0.98
    high_cardinality_unique_count: int = 50

    # Modeling
    cv_folds: int = 5
    random_state: int = 42
    max_train_samples: int = 20000

    # LLM Settings (Gemini API)
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model_name: str = "gemini-2.5-flash"

    # Category Health Weights (Sum = 1.0)
    weight_quality: float = 0.30
    weight_leakage: float = 0.30
    weight_drift: float = 0.20
    weight_fairness: float = 0.20


DEFAULT_CONFIG = AutopilotConfig()
