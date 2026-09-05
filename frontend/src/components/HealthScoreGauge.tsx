import React from "react";
import { ShieldCheck, AlertOctagon, TrendingUp, Users, Sparkles } from "lucide-react";
import type { AuditReport } from "../types/report";

interface HealthScoreGaugeProps {
  report: AuditReport;
}

export const HealthScoreGauge: React.FC<HealthScoreGaugeProps> = ({ report }) => {
  const score = report.health_score;
  const grade = report.health_grade;
  const breakdown = report.score_breakdown;

  const getScoreColor = (val: number) => {
    if (val >= 90) return "var(--color-success)";
    if (val >= 80) return "var(--accent-cyan)";
    if (val >= 70) return "var(--accent-blue)";
    if (val >= 55) return "var(--color-warning)";
    return "var(--color-critical)";
  };

  const primaryColor = getScoreColor(score);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const critCount = report.findings.filter((f) => f.severity === "CRITICAL").length;
  const warnCount = report.findings.filter((f) => f.severity === "WARNING").length;

  return (
    <div className="glass-panel" style={{ padding: "28px", marginBottom: "28px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "32px", alignItems: "center" }}>
        {/* SVG Circular Gauge */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ position: "relative", width: "136px", height: "136px" }}>
            <svg width="136" height="136" viewBox="0 0 136 136" style={{ transform: "rotate(-90deg)" }}>
              {/* Background Track */}
              <circle
                cx="68"
                cy="68"
                r={radius}
                fill="transparent"
                stroke="rgba(255, 255, 255, 0.08)"
                strokeWidth="10"
              />
              {/* Progress Arc */}
              <circle
                cx="68"
                cy="68"
                r={radius}
                fill="transparent"
                stroke={primaryColor}
                strokeWidth="10"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                style={{ transition: "stroke-dashoffset 1s ease" }}
              />
            </svg>

            {/* Inner Center Label */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span style={{ fontSize: "32px", fontWeight: 900, color: primaryColor, lineHeight: 1 }}>
                {Math.round(score)}
              </span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", marginTop: "2px" }}>
                Health Score
              </span>
            </div>
          </div>

          <div
            style={{
              marginTop: "12px",
              padding: "4px 12px",
              borderRadius: "var(--radius-full)",
              background: `rgba(${score >= 80 ? "16, 185, 129" : score >= 55 ? "245, 158, 11" : "239, 68, 68"}, 0.15)`,
              border: `1px solid ${primaryColor}`,
              color: primaryColor,
              fontWeight: 800,
              fontSize: "13px",
              letterSpacing: "0.04em",
            }}
          >
            GRADE {grade}
          </div>
        </div>

        {/* Breakdown & Executive Summary */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
            <div>
              <h2 style={{ fontSize: "20px", fontWeight: 700 }}>{report.dataset_name}</h2>
              <div style={{ display: "flex", gap: "12px", fontSize: "12.5px", color: "var(--text-muted)", marginTop: "2px" }}>
                <span><strong>{report.dataset_profile.rows.toLocaleString()}</strong> Rows</span>
                <span>•</span>
                <span><strong>{report.dataset_profile.columns}</strong> Columns</span>
                <span>•</span>
                <span>Target: <code style={{ color: "var(--accent-cyan)" }}>{report.dataset_profile.detected_target || "None (Unsupervised)"}</code></span>
                <span>•</span>
                <span style={{ color: critCount > 0 ? "var(--color-critical)" : "var(--color-success)" }}>
                  <strong>{critCount}</strong> Critical / <strong>{warnCount}</strong> Warnings
                </span>
              </div>
            </div>
          </div>

          {/* Executive Summary Callout */}
          <div
            style={{
              background: "rgba(255, 255, 255, 0.03)",
              borderLeft: "3px solid var(--accent-cyan)",
              borderRadius: "4px",
              padding: "10px 14px",
              fontSize: "13.5px",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
              marginBottom: "18px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase", marginBottom: "4px" }}>
              <Sparkles size={12} /> Executive Audit Summary
            </div>
            {report.executive_summary}
          </div>

          {/* Category Score Tiles */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
            <div className="glass-panel" style={{ padding: "12px", background: "rgba(255,255,255,0.02)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                <ShieldCheck size={13} color="var(--accent-cyan)" /> Quality (30%)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 800, marginTop: "4px", color: getScoreColor(breakdown.quality_score) }}>
                {breakdown.quality_score}
              </div>
            </div>

            <div className="glass-panel" style={{ padding: "12px", background: "rgba(255,255,255,0.02)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                <AlertOctagon size={13} color="var(--color-critical)" /> Leakage (30%)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 800, marginTop: "4px", color: getScoreColor(breakdown.leakage_score) }}>
                {breakdown.leakage_score}
              </div>
            </div>

            <div className="glass-panel" style={{ padding: "12px", background: "rgba(255,255,255,0.02)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                <TrendingUp size={13} color="var(--color-warning)" /> Drift (20%)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 800, marginTop: "4px", color: getScoreColor(breakdown.drift_score) }}>
                {breakdown.drift_score}
              </div>
            </div>

            <div className="glass-panel" style={{ padding: "12px", background: "rgba(255,255,255,0.02)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                <Users size={13} color="var(--accent-purple)" /> Slices (20%)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 800, marginTop: "4px", color: getScoreColor(breakdown.fairness_score) }}>
                {breakdown.fairness_score}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
