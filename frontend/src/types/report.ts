export type Severity = "CRITICAL" | "WARNING" | "INFO";

export type FindingStatus = "OBSERVED" | "SUSPECTED" | "CONFIRMED" | "NOT_ASSESSED";

export type DecisionStatus =
  | "OBSERVED"
  | "INVESTIGATING"
  | "EXPERIMENT_READY"
  | "EXPERIMENT_RUNNING"
  | "EVIDENCE_AVAILABLE"
  | "USER_DECISION"
  | "INCONCLUSIVE"
  | "NOT_APPLICABLE"
  | "FAILED";

export type PrimaryObjective =
  | "predictive_performance"
  | "generalization"
  | "interpretability"
  | "simplicity"
  | "data_retention"
  | "efficiency";

export type FindingCategory =
  | "target_leakage"
  | "train_test_contamination"
  | "missingness"
  | "outliers"
  | "format_error"
  | "distribution_drift"
  | "slice_bias"
  | "constant_column"
  | "high_cardinality"
  | "multicollinearity";

export type ColumnRole =
  | "id_candidate"
  | "target"
  | "numeric_feature"
  | "categorical_feature"
  | "datetime_feature"
  | "text_feature"
  | "constant";

export type SemanticType =
  | "email"
  | "phone"
  | "ssn"
  | "credit_card"
  | "ip_address"
  | "url"
  | "postal_code"
  | "currency"
  | "uuid"
  | "datetime_string"
  | "numeric_string"
  | "generic_text"
  | "generic_categorical"
  | "generic_numeric"
  | "boolean";

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
  pct: number;
}

export interface ValueFrequency {
  value: string;
  count: number;
  pct: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  inferred_role: ColumnRole;
  semantic_type: SemanticType;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  unique_ratio: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  p5?: number | null;
  p25?: number | null;
  p75?: number | null;
  p95?: number | null;
  skewness?: number | null;
  kurtosis?: number | null;
  top_values: ValueFrequency[];
  histogram: HistogramBin[];
  is_constant: boolean;
  is_id: boolean;
  sample_values: string[];
}

export interface DatasetProfile {
  rows: number;
  columns: number;
  duplicate_rows: number;
  duplicate_pct: number;
  memory_bytes: number;
  columns_profile: Record<string, ColumnProfile>;
  detected_target?: string | null;
  target_type?: string | null;
}

export interface Finding {
  id: string;
  type: FindingCategory;
  status: FindingStatus;
  severity: Severity;
  column?: string | null;
  title: string;
  description: string;
  evidence: Record<string, any>;
  recommendation?: string | null;
  decision_question?: string | null;
  confidence: number;
  ai_summary?: string | null;
}

export interface AnalysisContext {
  task_type?: string | null;
  target_column?: string | null;
  time_column?: string | null;
  prediction_timestamp?: string | null;
  entity_column?: string | null;
  primary_objective?: PrimaryObjective;
  constraints?: string[];
}

export interface UserDecision {
  selected_strategy_id: string;
  decision_rationale?: string | null;
  decided_at: string;
  custom_method_notes?: string | null;
}

export interface ExperimentVariantResult {
  strategy_id: string;
  strategy_name: string;
  description: string;
  is_baseline: boolean;
  primary_metric_name: string;
  primary_score: number;
  secondary_metrics: Record<string, number>;
  delta_from_baseline: number;
  delta_pct: number;
  rows_affected: number;
  rows_retained_pct: number;
  feature_count: number;
  pipeline_complexity: "Low" | "Medium" | "High";
  train_duration_ms: number;
  statistical_significance: string;
  practical_importance: string;
  limitations: string[];
  detailed_metrics?: Record<string, any>;
}

export interface DecisionEvidenceBlock {
  observation: string;
  empirical_evidence: string;
  what_we_know: string[];
  what_is_unknown: string[];
  controlled_parameters: Record<string, string>;
}

export interface DecisionCandidate {
  decision_id: string;
  finding_ids: string[];
  category: string;
  question: string;
  affected_columns: string[];
  rationale: string;
  expected_impact: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";
  evidence: Record<string, any>;
  evidence_block?: DecisionEvidenceBlock | null;
  available_strategies: string[];
  confidence: number;
  status: DecisionStatus;
  user_decision?: UserDecision | null;
  experiment_results?: ExperimentVariantResult[] | null;
  system_interpretation?: string | null;
}

export interface ExperimentResult {
  id: string;
  name: string;
  category: string;
  description: string;
  model_name: string;
  baseline_metric_name: string;
  baseline_score: number;
  modified_score: number;
  delta: number;
  delta_pct: number;
  status: string;
  notes: string;
  detailed_metrics: Record<string, any>;
}

export interface SliceMetric {
  feature: string;
  slice_value: string;
  sample_count: number;
  sample_pct: number;
  metric_name: string;
  slice_score: number;
  overall_score: number;
  disparity_ratio: number;
  is_disparate: boolean;
}

export interface DriftMetric {
  feature: string;
  psi: number;
  ks_statistic?: number | null;
  ks_p_value?: number | null;
  is_drift_detected: boolean;
  severity: Severity;
  status?: FindingStatus;
  bin_details: Array<{
    bin_label: string;
    expected_pct: number;
    actual_pct: number;
    psi_contribution: number;
  }>;
}

export interface HealthScoreBreakdown {
  overall_score: number;
  quality_score: number;
  leakage_score: number;
  drift_score: number;
  fairness_score: number;
  grade: string;
}

export interface AuditReport {
  report_id: string;
  dataset_name: string;
  created_at: string;
  health_score: number;
  health_grade: string;
  decision_readiness: string;
  score_breakdown: HealthScoreBreakdown;
  dataset_profile: DatasetProfile;
  findings: Finding[];
  decision_candidates: DecisionCandidate[];
  experiments: ExperimentResult[];
  drift_summary: DriftMetric[];
  slice_summary: SliceMetric[];
  executive_summary: string;
  analysis_context?: AnalysisContext | null;
}

export interface SampleDataset {
  id: string;
  title: string;
  description: string;
  target?: string;
  time_col?: string;
  file_path: string;
  badge: string;
  color: string;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress_pct: number;
  current_stage: string;
  error_message?: string | null;
  created_at: string;
}

