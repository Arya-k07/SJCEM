import React, { useState, useEffect } from "react";
import { Loader2, AlertTriangle } from "lucide-react";
import type { JobStatus } from "../types/report";

interface ProgressTrackerProps {
  status: JobStatus;
}

const STAGES = [
  { label: "Ingestion", minPct: 10 },
  { label: "Profiling", minPct: 25 },
  { label: "Quality Audit", minPct: 40 },
  { label: "Leakage Check", minPct: 55 },
  { label: "Drift & Stability", minPct: 70 },
  { label: "Oracle Models", minPct: 85 },
  { label: "Decision Lab", minPct: 95 },
];

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({ status }) => {
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatElapsed = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div style={{ maxWidth: "680px", margin: "48px auto", padding: "0 20px" }}>
      <div
        style={{
          background: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          padding: "36px 32px",
          textAlign: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div
          style={{
            width: "56px",
            height: "56px",
            borderRadius: "50%",
            background: "#f0fdfa",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 16px",
          }}
        >
          {status.status === "failed" ? (
            <AlertTriangle size={28} color="#b91c1c" />
          ) : (
            <Loader2 size={28} color="#0f766e" style={{ animation: "spin 1.4s linear infinite" }} />
          )}
        </div>

        <h2 style={{ fontSize: "20px", fontWeight: 800, color: "#0f172a", marginBottom: "6px" }}>
          {status.status === "failed" ? "Audit Encountered an Issue" : "Autonomous Dataset Audit in Progress"}
        </h2>
        <p style={{ fontSize: "13.5px", color: "#475569", marginBottom: "20px" }}>
          {status.current_stage || "Analyzing tabular structure, distributions, and ML risks..."}
        </p>

        {/* Elapsed Timer Badge */}
        <div style={{ marginBottom: "20px" }}>
          <span
            style={{
              fontSize: "12px",
              fontWeight: 600,
              padding: "3px 10px",
              borderRadius: "4px",
              background: "#f8fafc",
              color: "#64748b",
              border: "1px solid #e2e8f0",
            }}
          >
            Elapsed {formatElapsed(elapsedSeconds)}
          </span>
        </div>

        {/* Progress Bar */}
        <div
          style={{
            height: "6px",
            background: "#f1f5f9",
            borderRadius: "9999px",
            overflow: "hidden",
            marginBottom: "28px",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${status.progress_pct}%`,
              background: "#0f766e",
              borderRadius: "9999px",
              transition: "width 0.3s ease",
            }}
          />
        </div>

        {/* Stages Stepper */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${STAGES.length}, 1fr)`,
            gap: "8px",
            textAlign: "center",
          }}
        >
          {STAGES.map((s, idx) => {
            const isDone = status.progress_pct >= s.minPct;
            const isCurrent = status.progress_pct >= (STAGES[idx - 1]?.minPct || 0) && status.progress_pct < s.minPct;
            return (
              <div key={s.label} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div
                  style={{
                    width: "22px",
                    height: "22px",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "11px",
                    fontWeight: 700,
                    marginBottom: "6px",
                    background: isDone ? "#0f766e" : isCurrent ? "#ccfbf1" : "#f1f5f9",
                    color: isDone ? "#ffffff" : isCurrent ? "#0f766e" : "#94a3b8",
                    border: isCurrent ? "1px solid #0f766e" : "none",
                  }}
                >
                  {isDone ? "✓" : idx + 1}
                </div>
                <span
                  style={{
                    fontSize: "10px",
                    color: isDone || isCurrent ? "#0f172a" : "#94a3b8",
                    fontWeight: isCurrent ? 700 : 500,
                    lineHeight: 1.2,
                  }}
                >
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
