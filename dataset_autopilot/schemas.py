"""
Pydantic schemas and data models for Dataset Autopilot reports and API contracts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dataset_autopilot.config import (
    ColumnRole,
    DecisionStatus,
    FindingCategory,
    FindingStatus,
    PrimaryObjective,
    SemanticType,
    Severity,
)


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    pct: float


class ValueFrequency(BaseModel):
    value: str
    count: int
    pct: float


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_role: ColumnRole
    semantic_type: SemanticType
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    top_values: List[ValueFrequency] = Field(default_factory=list)
    histogram: List[HistogramBin] = Field(default_factory=list)
    is_constant: bool = False
    is_id: bool = False
    sample_values: List[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_pct: float
    memory_bytes: int
    columns_profile: Dict[str, ColumnProfile]
    detected_target: Optional[str] = None
    target_type: Optional[str] = None  # binary_classification, multiclass, regression


class Finding(BaseModel):
    id: str
    type: FindingCategory
    status: FindingStatus = FindingStatus.OBSERVED
    severity: Severity = Severity.INFO
    column: Optional[str] = None
    title: str
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    decision_question: Optional[str] = None
    confidence: float = 1.0
    ai_summary: Optional[str] = None


class AnalysisContext(BaseModel):
    task_type: Optional[str] = "classification"  # classification, regression, unsupervised
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    entity_column: Optional[str] = None
    primary_objective: PrimaryObjective = PrimaryObjective.PREDICTIVE_PERFORMANCE
    constraints: List[str] = Field(default_factory=list)


class UserDecision(BaseModel):
    selected_strategy_id: str
    decision_rationale: Optional[str] = None
    decided_at: str
    custom_method_notes: Optional[str] = None


class ExperimentVariantResult(BaseModel):
    strategy_id: str
    strategy_name: str
    description: str
    is_baseline: bool = False
    primary_metric_name: str
    primary_score: float
    secondary_metrics: Dict[str, float] = Field(default_factory=dict)
    delta_from_baseline: float = 0.0
    delta_pct: float = 0.0
    rows_affected: int = 0
    rows_retained_pct: float = 100.0
    feature_count: int = 0
    pipeline_complexity: str = "Low"  # Low, Medium, High
    train_duration_ms: float = 0.0
    statistical_significance: str = "Indistinguishable"
    practical_importance: str = "Negligible"
    limitations: List[str] = Field(default_factory=list)
    detailed_metrics: Dict[str, Any] = Field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.delta_from_baseline


class DecisionEvidenceBlock(BaseModel):
    observation: str
    empirical_evidence: str
    what_we_know: List[str] = Field(default_factory=list)
    what_is_unknown: List[str] = Field(default_factory=list)
    controlled_parameters: Dict[str, str] = Field(default_factory=dict)


class DecisionCandidate(BaseModel):
    decision_id: str
    finding_ids: List[str] = Field(default_factory=list)
    category: str
    question: str
    affected_columns: List[str] = Field(default_factory=list)
    rationale: str
    expected_impact: str = "UNKNOWN"  # HIGH, MEDIUM, LOW, UNKNOWN
    evidence: Dict[str, Any] = Field(default_factory=dict)
    evidence_block: Optional[DecisionEvidenceBlock] = None
    available_strategies: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    status: DecisionStatus = DecisionStatus.EXPERIMENT_READY
    user_decision: Optional[UserDecision] = None
    experiment_results: Optional[List[ExperimentVariantResult]] = None
    system_interpretation: Optional[str] = None


class ExperimentResult(BaseModel):
    id: str
    name: str
    category: str
    description: str
    model_name: str
    baseline_metric_name: str
    baseline_score: float
    modified_score: float
    delta: float
    delta_pct: float
    status: str  # "leakage_confirmed", "improved", "degraded", "neutral"
    notes: str
    detailed_metrics: Dict[str, Any] = Field(default_factory=dict)


class SliceMetric(BaseModel):
    feature: str
    slice_value: str
    sample_count: int
    sample_pct: float
    metric_name: str
    slice_score: float
    overall_score: float
    disparity_ratio: float
    is_disparate: bool


class DriftMetric(BaseModel):
    feature: str
    psi: float
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    is_drift_detected: bool
    severity: Severity = Severity.INFO
    status: FindingStatus = FindingStatus.OBSERVED
    bin_details: List[Dict[str, Any]] = Field(default_factory=list)


class HealthScoreBreakdown(BaseModel):
    overall_score: float
    quality_score: float
    leakage_score: float
    drift_score: float
    fairness_score: float
    grade: str


class AuditReport(BaseModel):
    report_id: str
    dataset_name: str
    created_at: str
    health_score: float
    health_grade: str
    decision_readiness: str = "Ready for evaluation"
    score_breakdown: HealthScoreBreakdown
    dataset_profile: DatasetProfile
    findings: List[Finding]
    decision_candidates: List[DecisionCandidate] = Field(default_factory=list)
    experiments: List[ExperimentResult] = Field(default_factory=list)
    drift_summary: List[DriftMetric] = Field(default_factory=list)
    slice_summary: List[SliceMetric] = Field(default_factory=list)
    executive_summary: str
    analysis_context: Optional[AnalysisContext] = None


class AnalysisRequest(BaseModel):
    dataset_url: Optional[str] = None
    dataset_base64: Optional[str] = None
    file_name: Optional[str] = "dataset.csv"
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    split_column: Optional[str] = None
    gemini_api_key: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress_pct: int
    current_stage: str
    error_message: Optional[str] = None
    created_at: str
