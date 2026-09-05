import type {
  AuditReport,
  DecisionCandidate,
  ExperimentResult,
  ExperimentVariantResult,
  JobStatus,
  SampleDataset,
} from "../types/report";

const API_BASE = "http://127.0.0.1:8000";

export const api = {
  async fetchSamples(): Promise<SampleDataset[]> {
    const res = await fetch(`${API_BASE}/api/samples`);
    if (!res.ok) throw new Error("Failed to fetch benchmark sample datasets");
    return res.json();
  },

  async triggerSampleAnalysis(sampleId: string, geminiApiKey?: string): Promise<{ job_id: string; status: string }> {
    const url = new URL(`${API_BASE}/api/samples/${sampleId}/analyze`);
    if (geminiApiKey) {
      url.searchParams.append("gemini_api_key", geminiApiKey);
    }
    const res = await fetch(url.toString(), { method: "POST" });
    if (!res.ok) throw new Error("Failed to trigger sample analysis");
    return res.json();
  },

  async uploadAndAnalyze(formData: FormData): Promise<{ job_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || "Failed to upload and analyze dataset");
    }
    return res.json();
  },

  async getJobStatus(jobId: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/api/status/${jobId}`);
    if (!res.ok) throw new Error("Failed to fetch job status");
    return res.json();
  },

  async getReport(jobId: string): Promise<AuditReport> {
    const res = await fetch(`${API_BASE}/api/results/${jobId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Results not ready" }));
      throw new Error(err.detail || "Failed to fetch audit report");
    }
    return res.json();
  },

  async getExperiments(jobId: string): Promise<ExperimentResult[]> {
    const res = await fetch(`${API_BASE}/api/experiments/${jobId}`);
    if (!res.ok) throw new Error("Failed to fetch experiments");
    return res.json();
  },

  // Decision Lab Methods
  async fetchDecisionCandidates(runId: string): Promise<DecisionCandidate[]> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/decision-candidates`);
    if (!res.ok) throw new Error("Failed to fetch decision candidates");
    return res.json();
  },

  async getDecisionDetails(runId: string, decisionId: string): Promise<DecisionCandidate> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/decisions/${decisionId}`);
    if (!res.ok) throw new Error("Failed to fetch decision details");
    return res.json();
  },

  async runCandidateExperiments(
    runId: string,
    decisionId: string,
    strategyIds?: string[]
  ): Promise<DecisionCandidate> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/decisions/${decisionId}/experiments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy_ids: strategyIds }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Experiment run failed" }));
      throw new Error(err.detail || "Failed to execute decision experiments");
    }
    return res.json();
  },

  async getDecisionResults(runId: string, decisionId: string): Promise<ExperimentVariantResult[]> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/decisions/${decisionId}/results`);
    if (!res.ok) throw new Error("Failed to fetch experiment results");
    return res.json();
  },

  async recordDecision(
    runId: string,
    decisionId: string,
    payload: { selected_strategy_id: string; decision_rationale?: string; custom_method_notes?: string }
  ): Promise<DecisionCandidate> {
    const res = await fetch(`${API_BASE}/api/runs/${runId}/decisions/${decisionId}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to record decision");
    return res.json();
  },

  getArtifactUrl(runId: string, decisionId: string, artifactType: "notebook" | "pipeline" | "html" | "json"): string {
    return `${API_BASE}/api/runs/${runId}/decisions/${decisionId}/artifacts/${artifactType}`;
  },

  getHtmlExportUrl(jobId: string): string {
    return `${API_BASE}/api/export/${jobId}/html`;
  },
};
