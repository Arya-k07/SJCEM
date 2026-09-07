import type { FeedbackRecord, Insights } from "../types";

const baseUrl = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001").replace(/\/$/, "");
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = Array.isArray(payload?.detail)
      ? payload.detail.map((item: { loc?: unknown[]; msg?: string }) => `${item.loc?.join(".") || "request"}: ${item.msg || "invalid value"}`).join("; ")
      : payload?.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
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
  const parseList = (value: unknown): string[] => {
    if (Array.isArray(value)) return value.map(String);
    if (typeof value === "string") {
      try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed.map(String) : value ? [value] : [];
      } catch {
        return value ? [value] : [];
      }
    }
    return [];
  };
  const parseAspects = (value: unknown): Array<Record<string, string>> => {
    if (Array.isArray(value)) return value.filter(item => item && typeof item === "object").map(item => Object.fromEntries(Object.entries(item as Record<string, unknown>).map(([key, itemValue]) => [key, String(itemValue)])));
    if (typeof value === "string") {
      try { return parseAspects(JSON.parse(value)); } catch { return []; }
    }
    return [];
  };
  const numberValue = (value: unknown) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : 0;
  };
  const normalized = {
    ...record,
    feedback: String(record.feedback ?? record.cleaned_text ?? ""),
    cleaned_text: String(record.cleaned_text ?? record.feedback ?? ""),
    sentiment: String(record.sentiment ?? "Neutral"),
    sentiment_score: numberValue(record.sentiment_score),
    sentiment_confidence: numberValue(record.sentiment_confidence),
    topics: parseList(record.topics),
    keywords: parseList(record.keywords),
    emotion: String(record.emotion ?? "Unknown"),
    aspect_sentiments: parseAspects(record.aspect_sentiments),
  };
  return request<FeedbackRecord>("/api/decision", { method: "POST", body: JSON.stringify(normalized) });
}
export async function insights(records: FeedbackRecord[]): Promise<Insights> {
  return request<Insights>("/api/insights", { method: "POST", body: JSON.stringify({ records }) });
}
