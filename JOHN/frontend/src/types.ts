export type RecordValue = string | number | boolean | null | string[] | Record<string, unknown>;
export type FeedbackRecord = Record<string, RecordValue>;
export interface Insights {
  executive_summary?: string;
  key_insights?: Array<{ title?: string; description?: string; importance?: string; evidence?: string[] }>;
  hotspots?: Array<{ dimension?: string; value?: string; reason?: string; evidence?: string }>;
  trends?: Array<{ metric?: string; direction?: string; description?: string; evidence?: string }>;
  themes?: Array<{ theme?: string; description?: string; sentiment?: string; importance?: string }>;
  positive_signals?: Array<{ area?: string; description?: string; evidence?: string }>;
  anomalies?: Array<{ description?: string; evidence?: string; confidence?: string }>;
  recommendations?: Array<{ priority?: string; action?: string; reason?: string; target?: string }>;
  data_limitations?: string[];
  generation_status?: string;
}
