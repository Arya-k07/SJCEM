import type { FeedbackRecord, Insights } from "../types";

const baseUrl = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001").replace(/\/$/, "");
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `Request failed (${response.status})`);
  return response.json() as Promise<T>;
}
export async function health(): Promise<boolean> { try { await request("/health"); return true; } catch { return false; } }
export async function analyze(file: File): Promise<FeedbackRecord[]> {
  const body = new FormData(); body.append("file", file);
  const response = await fetch(`${baseUrl}/api/analyze`, { method: "POST", body });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || "Unable to analyze file");
  const data = await response.json() as { data?: FeedbackRecord[] };
  if (!Array.isArray(data.data)) throw new Error("Backend returned a malformed analysis response");
  return data.data;
}
export async function decide(record: FeedbackRecord): Promise<FeedbackRecord> {
  return request<FeedbackRecord>("/api/decision", { method: "POST", body: JSON.stringify(record) });
}
export async function insights(records: FeedbackRecord[]): Promise<Insights> {
  return request<Insights>("/api/insights", { method: "POST", body: JSON.stringify({ records }) });
}
