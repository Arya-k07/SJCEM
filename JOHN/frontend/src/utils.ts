import type { FeedbackRecord } from "./types";
export const text = (value: unknown) => Array.isArray(value) ? value.join(", ") : value == null ? "" : String(value);
export const countBy = (rows: FeedbackRecord[], key: string) => rows.reduce<Record<string, number>>((out, row) => { const value = text(row[key]) || "Unknown"; out[value] = (out[value] || 0) + 1; return out; }, {});
export const priorityScore = (value: unknown) => ({ P1: 4, P2: 3, P3: 2, P4: 1 }[text(value)] || 0);
export const categoryKey = (row: FeedbackRecord) => text(row.category || row.department) || "Unknown";
export const severityOrder = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
export const priorityOrder = ["P1", "P2", "P3", "P4"];
