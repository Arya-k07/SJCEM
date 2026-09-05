import React from "react";
import type { AuditReport } from "../types/report";
import { Scale } from "lucide-react";

interface FairnessTabProps {
  report: AuditReport;
}

export const FairnessTab: React.FC<FairnessTabProps> = ({ report }) => {
  const slices = report.slice_summary || [];
  const disparateCount = slices.filter((s) => s.is_disparate).length;

  return (
    <div>
      {/* Top Banner */}
      <div
        className="glass-panel"
        style={{
          padding: "20px 24px",
          marginBottom: "24px",
          borderLeft: `4px solid ${disparateCount > 0 ? "var(--color-warning)" : "var(--color-success)"}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Scale size={20} color={disparateCount > 0 ? "var(--color-warning)" : "var(--color-success)"} />
            <h3 style={{ fontSize: "17px", fontWeight: 700 }}>
              {disparateCount > 0
                ? `${disparateCount} Demographic / Categorical Subgroup(s) with Performance Disparity`
                : "Equitable Subgroup Performance Confirmed"}
            </h3>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Evaluated model predictions across categorical slices and quantile partitions to detect group fairness gaps (&gt; 15% disparity).
          </p>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Fairness Score</div>
          <div style={{ fontSize: "24px", fontWeight: 800, color: report.score_breakdown.fairness_score >= 80 ? "var(--color-success)" : "var(--color-warning)" }}>
            {report.score_breakdown.fairness_score}/100
          </div>
        </div>
      </div>

      {/* Slices Table */}
      <div className="glass-panel" style={{ padding: "24px" }}>
        <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "16px" }}>
          Subgroup Performance Breakdown ({slices.length} Slices Evaluated)
        </h4>

        {slices.length === 0 ? (
          <div style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
            No demographic or categorical slices met minimum sample size criteria for group evaluation.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13.5px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left" }}>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Feature / Slice</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Subgroup Value</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Sample Size</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Metric</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Slice Score</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Overall Score</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Disparity Ratio</th>
                  <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {slices.map((s, idx) => (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: "1px solid rgba(255,255,255,0.04)",
                      background: s.is_disparate ? "rgba(245, 158, 11, 0.04)" : "transparent",
                    }}
                  >
                    <td style={{ padding: "12px" }}>
                      <code>{s.feature}</code>
                    </td>
                    <td style={{ padding: "12px", fontWeight: 600 }}>
                      {s.slice_value}
                    </td>
                    <td style={{ padding: "12px", color: "var(--text-muted)" }}>
                      {s.sample_count} ({(s.sample_pct * 100).toFixed(1)}%)
                    </td>
                    <td style={{ padding: "12px", color: "var(--text-secondary)" }}>
                      {s.metric_name}
                    </td>
                    <td style={{ padding: "12px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      {s.slice_score.toFixed(3)}
                    </td>
                    <td style={{ padding: "12px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      {s.overall_score.toFixed(3)}
                    </td>
                    <td style={{ padding: "12px", fontFamily: "var(--font-mono)", fontWeight: 700, color: s.disparity_ratio < 0.85 ? "var(--color-warning)" : "var(--text-primary)" }}>
                      {s.disparity_ratio.toFixed(2)}x
                    </td>
                    <td style={{ padding: "12px" }}>
                      {s.is_disparate ? (
                        <span className="badge badge-warning">Disparity &gt; 15%</span>
                      ) : (
                        <span className="badge badge-success">Equitable</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
