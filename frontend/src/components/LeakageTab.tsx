import React from "react";
import type { Finding, AuditReport } from "../types/report";
import { AlertOctagon, CheckCircle2, ArrowUpRight } from "lucide-react";

interface LeakageTabProps {
  report: AuditReport;
  onSelectFinding: (finding: Finding) => void;
}

export const LeakageTab: React.FC<LeakageTabProps> = ({ report, onSelectFinding }) => {
  const targetCol = report.dataset_profile.detected_target;
  const leakageFindings = report.findings.filter(
    (f) => f.type === "target_leakage" || f.type === "train_test_contamination"
  );

  return (
    <div>
      {/* Overview Banner */}
      <div
        className="glass-panel"
        style={{
          padding: "20px 24px",
          marginBottom: "24px",
          borderLeft: `4px solid ${leakageFindings.length > 0 ? "var(--color-critical)" : "var(--color-success)"}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            {leakageFindings.length > 0 ? (
              <AlertOctagon size={20} color="var(--color-critical)" />
            ) : (
              <CheckCircle2 size={20} color="var(--color-success)" />
            )}
            <h3 style={{ fontSize: "17px", fontWeight: 700 }}>
              {leakageFindings.length > 0
                ? `${leakageFindings.length} Target Leakage or Contamination Risk(s) Flagged`
                : "No Target Leakage Detected"}
            </h3>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Target: <code>{targetCol || "None"}</code> • Audited features for single-feature AUC &gt; 0.88, correlation &gt; 0.70, and post-event naming.
          </p>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Leakage Score</div>
          <div style={{ fontSize: "24px", fontWeight: 800, color: report.score_breakdown.leakage_score >= 80 ? "var(--color-success)" : "var(--color-critical)" }}>
            {report.score_breakdown.leakage_score}/100
          </div>
        </div>
      </div>

      {/* Flagged Leakage Findings List */}
      {leakageFindings.length > 0 && (
        <div style={{ marginBottom: "28px" }}>
          <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "12px" }}>
            Suspect Features with Empirical Evidence
          </h4>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
            {leakageFindings.map((f) => (
              <div
                key={f.id}
                className="glass-panel"
                style={{
                  padding: "20px",
                  cursor: "pointer",
                  background: "rgba(239, 68, 68, 0.04)",
                  border: "1px solid var(--border-critical)",
                }}
                onClick={() => onSelectFinding(f)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                  <span className={`badge badge-${f.severity.toLowerCase()}`}>{f.severity}</span>
                  <span style={{ fontSize: "12px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{f.id}</span>
                </div>

                <h4 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "6px" }}>{f.title}</h4>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.4, marginBottom: "14px" }}>
                  {f.description}
                </p>

                {/* Evidence Metrics */}
                <div style={{ background: "rgba(0,0,0,0.3)", padding: "10px", borderRadius: "var(--radius-sm)", fontSize: "12px" }}>
                  {f.evidence.single_feature_score !== undefined && (
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                      <span style={{ color: "var(--text-muted)" }}>Single-Feature {f.evidence.single_feature_metric || "AUC"}:</span>
                      <strong style={{ color: "var(--color-critical)", fontFamily: "var(--font-mono)" }}>
                        {f.evidence.single_feature_score}
                      </strong>
                    </div>
                  )}
                  {f.evidence.correlation_with_target !== undefined && (
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-muted)" }}>Target Correlation (|r|):</span>
                      <strong style={{ color: "var(--color-critical)", fontFamily: "var(--font-mono)" }}>
                        {Math.abs(f.evidence.correlation_with_target)}
                      </strong>
                    </div>
                  )}
                </div>

                <div style={{ marginTop: "12px", display: "flex", justifyContent: "flex-end" }}>
                  <span style={{ fontSize: "12px", color: "var(--accent-cyan)", display: "flex", alignItems: "center", gap: "4px" }}>
                    View Deep Evidence <ArrowUpRight size={13} />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Explanatory Callout */}
      <div className="glass-panel" style={{ padding: "20px" }}>
        <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "6px" }}>
          Why Target Leakage Destroys ML Reliability in Production
        </h4>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
          Target leakage occurs when training data incorporates features that won’t be available when the model makes predictions in real time
          (such as a post-churn refund timestamp or direct outcome status). Models trained on leaky features show artificially stellar cross-validation
          scores (e.g. 0.99 AUC) but catastrophically fail upon production deployment.
        </p>
      </div>
    </div>
  );
};
