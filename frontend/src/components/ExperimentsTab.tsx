import React from "react";
import type { AuditReport } from "../types/report";
import { FlaskConical } from "lucide-react";

interface ExperimentsTabProps {
  report: AuditReport;
}

export const ExperimentsTab: React.FC<ExperimentsTabProps> = ({ report }) => {
  const experiments = report.experiments || [];

  return (
    <div>
      {/* Top Banner */}
      <div
        className="glass-panel"
        style={{
          padding: "20px 24px",
          marginBottom: "24px",
          borderLeft: "4px solid var(--accent-cyan)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <FlaskConical size={20} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: "17px", fontWeight: 700 }}>
              Controlled Oracle Model Remediation Experiments ({experiments.length})
            </h3>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Empirical A/B tests executed on baseline Random Forest oracle to quantify metric inflation, leakage effect, and imputation efficacy.
          </p>
        </div>
      </div>

      {/* Experiments Cards Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "18px" }}>
        {experiments.map((e) => {
          const isBaseline = e.category === "baseline";
          const isLeakConfirmed = e.status === "leakage_confirmed";
          const isImproved = e.status === "improved";

          return (
            <div
              key={e.id}
              className="glass-panel"
              style={{
                padding: "22px",
                borderTop: `4px solid ${
                  isBaseline
                    ? "var(--accent-blue)"
                    : isLeakConfirmed
                    ? "var(--color-critical)"
                    : isImproved
                    ? "var(--color-success)"
                    : "var(--text-muted)"
                }`,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                <span className="badge badge-secondary" style={{ fontSize: "11px", textTransform: "uppercase" }}>
                  {e.category.replace(/_/g, " ")}
                </span>
                <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{e.id}</span>
              </div>

              <h4 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "6px" }}>{e.name}</h4>
              <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: 1.4, marginBottom: "16px" }}>
                {e.description}
              </p>

              {/* Metric Comparison Box */}
              <div
                style={{
                  background: "rgba(0,0,0,0.25)",
                  padding: "12px 14px",
                  borderRadius: "var(--radius-sm)",
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 1fr",
                  gap: "10px",
                  textAlign: "center",
                  marginBottom: "14px",
                }}
              >
                <div>
                  <div style={{ fontSize: "10.5px", color: "var(--text-muted)", textTransform: "uppercase" }}>Baseline</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)", marginTop: "2px" }}>
                    {e.baseline_score.toFixed(3)}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: "10.5px", color: "var(--text-muted)", textTransform: "uppercase" }}>Modified</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--accent-cyan)", marginTop: "2px" }}>
                    {e.modified_score.toFixed(3)}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: "10.5px", color: "var(--text-muted)", textTransform: "uppercase" }}>Impact (Δ)</div>
                  <div
                    style={{
                      fontSize: "15px",
                      fontWeight: 800,
                      fontFamily: "var(--font-mono)",
                      marginTop: "2px",
                      color: e.delta > 0 ? "var(--color-success)" : e.delta < 0 ? "var(--color-critical)" : "var(--text-muted)",
                    }}
                  >
                    {e.delta > 0 ? `+${e.delta.toFixed(3)}` : e.delta.toFixed(3)}
                  </div>
                </div>
              </div>

              {/* Empirical Notes */}
              <div style={{ fontSize: "12px", color: "var(--text-muted)", fontStyle: "italic", lineHeight: 1.4 }}>
                <strong>Outcome:</strong> {e.notes}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
