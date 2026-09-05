import React, { useState } from "react";
import type { AuditReport, DecisionCandidate } from "../types/report";
import { ChevronDown, ChevronUp } from "lucide-react";

interface DecisionWorkbenchProps {
  report: AuditReport;
  onSelectDecision: (decision: DecisionCandidate) => void;
}

export const DecisionWorkbench: React.FC<DecisionWorkbenchProps> = ({
  report,
  onSelectDecision,
}) => {
  const candidates = report.decision_candidates || [];
  const targetCol = report.dataset_profile.detected_target;
  const isSupervised = Boolean(targetCol);

  // Track expanded state for detailed evidence blocks per candidate
  const [expandedDecisions, setExpandedDecisions] = useState<Record<string, boolean>>({});

  const toggleExpand = (decisionId: string) => {
    setExpandedDecisions((prev) => ({
      ...prev,
      [decisionId]: !prev[decisionId],
    }));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }} className="animate-fade-in">
      {/* Workstation Header & Decision Readiness */}
      <div
        style={{
          padding: "20px 24px",
          background: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          borderLeft: "4px solid #0f766e",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
            <span
              style={{
                fontSize: "11px",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                padding: "2px 8px",
                borderRadius: "4px",
                background: "#f0fdfa",
                color: "#0f766e",
                border: "1px solid #ccfbf1",
              }}
            >
              Decision Readiness
            </span>
            <span style={{ fontSize: "12.5px", color: "#64748b" }}>
              Audited {report.dataset_profile.rows.toLocaleString()} rows • {report.dataset_profile.columns} columns
            </span>
          </div>
          <h2 style={{ fontSize: "19px", fontWeight: 800, margin: 0, color: "#0f172a" }}>
            {report.decision_readiness}
          </h2>
          <p style={{ fontSize: "13px", color: "#475569", marginTop: "4px", marginBottom: 0, lineHeight: "1.5" }}>
            {isSupervised
              ? `Target: '${targetCol}' (${report.dataset_profile.target_type || "classification"}). Controlled empirical comparisons isolate the generalizability and complexity impact of alternative data strategies.`
              : "Unsupervised dataset mode. Supervised baseline models and decision candidates are suppressed until a target is specified."}
          </p>
        </div>

        <div style={{ display: "flex", gap: "20px", textAlign: "right" }}>
          <div>
            <div style={{ fontSize: "24px", fontWeight: 800, color: "#0f766e" }}>
              {candidates.length}
            </div>
            <div style={{ fontSize: "11px", fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Actionable Decisions
            </div>
          </div>
          <div style={{ borderLeft: "1px solid #e2e8f0", paddingLeft: "20px" }}>
            <div style={{ fontSize: "24px", fontWeight: 800, color: "#334155" }}>
              {report.findings.length}
            </div>
            <div style={{ fontSize: "11px", fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Raw Observations
            </div>
          </div>
        </div>
      </div>

      {/* Decision Candidates List */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
          <div>
            <h3 style={{ fontSize: "15.5px", fontWeight: 700, color: "#0f172a", margin: 0 }}>
              Prioritized Decision Candidates ({candidates.length})
            </h3>
            <p style={{ fontSize: "12.5px", color: "#64748b", margin: "2px 0 0 0" }}>
              Only findings satisfying materiality thresholds and testable alternative hypotheses become Decision Candidates.
            </p>
          </div>
          <span style={{ fontSize: "11.5px", color: "#64748b", background: "#f8fafc", padding: "3px 8px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
            Deterministic 4-Part Filter Active
          </span>
        </div>

        {candidates.length === 0 ? (
          <div
            style={{
              padding: "48px 32px",
              textAlign: "center",
              background: "#ffffff",
              borderRadius: "8px",
              border: "1px solid #e2e8f0",
              color: "#475569",
            }}
          >
            <div style={{ fontSize: "15.5px", fontWeight: 600, color: "#0f172a", marginBottom: "6px" }}>
              {isSupervised
                ? "No high-impact decision candidates detected."
                : "Unsupervised dataset. No supervised decision candidates."}
            </div>
            <p style={{ fontSize: "13px", color: "#64748b", maxWidth: "540px", margin: "0 auto" }}>
              {isSupervised
                ? "All dataset properties fall within standard operational tolerances or represent baseline features with no conflicting alternative hypotheses."
                : "To formulate and run supervised Decision Lab experiments, supply a target column in the analysis configuration."}
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {candidates.map((cand) => {
              const isDecided = cand.status === "USER_DECISION";
              const isEvidenceReady = cand.status === "EVIDENCE_AVAILABLE" || isDecided;
              const isExpanded = Boolean(expandedDecisions[cand.decision_id]);
              const ev = cand.evidence_block;

              const impactBg =
                cand.expected_impact === "HIGH"
                  ? "#fef2f2"
                  : cand.expected_impact === "MEDIUM"
                  ? "#fffbeb"
                  : "#f0f9ff";
              const impactText =
                cand.expected_impact === "HIGH"
                  ? "#b91c1c"
                  : cand.expected_impact === "MEDIUM"
                  ? "#b45309"
                  : "#0369a1";
              const impactBorder =
                cand.expected_impact === "HIGH"
                  ? "#fecaca"
                  : cand.expected_impact === "MEDIUM"
                  ? "#fde68a"
                  : "#bae6fd";

              return (
                <div
                  key={cand.decision_id}
                  style={{
                    background: "#ffffff",
                    borderRadius: "8px",
                    border: "1px solid #e2e8f0",
                    borderLeft: isDecided ? "4px solid #10b981" : `4px solid ${impactText}`,
                    padding: "20px 24px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.02)",
                  }}
                >
                  {/* Card Meta & Badges */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                      <span
                        style={{
                          fontSize: "11.5px",
                          fontWeight: 700,
                          padding: "2px 8px",
                          borderRadius: "4px",
                          background: "#f1f5f9",
                          color: "#334155",
                          border: "1px solid #cbd5e1",
                        }}
                      >
                        {cand.decision_id}
                      </span>
                      <span
                        style={{
                          fontSize: "11px",
                          fontWeight: 700,
                          padding: "2px 8px",
                          borderRadius: "4px",
                          background: impactBg,
                          color: impactText,
                          border: `1px solid ${impactBorder}`,
                          textTransform: "uppercase",
                        }}
                      >
                        {cand.expected_impact} Materiality
                      </span>
                      <span
                        style={{
                          fontSize: "11px",
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: "4px",
                          background: "#f8fafc",
                          color: "#475569",
                          border: "1px solid #e2e8f0",
                          textTransform: "capitalize",
                        }}
                      >
                        {cand.category.replace("_", " ")}
                      </span>
                      {isDecided && (
                        <span
                          style={{
                            fontSize: "11px",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: "4px",
                            background: "#ecfdf5",
                            color: "#047857",
                            border: "1px solid #a7f3d0",
                          }}
                        >
                          ✓ Decision Confirmed
                        </span>
                      )}
                      {cand.status === "INCONCLUSIVE" && (
                        <span
                          style={{
                            fontSize: "11px",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: "4px",
                            background: "#f8fafc",
                            color: "#475569",
                            border: "1px solid #cbd5e1",
                          }}
                        >
                          Evidence Inconclusive
                        </span>
                      )}
                    </div>

                    <div style={{ fontSize: "12px", color: "#64748b", fontWeight: 500 }}>
                      {cand.available_strategies.length} testable strategies
                    </div>
                  </div>

                  {/* Decision Question Headline (Always Visible) */}
                  <div>
                    <h4 style={{ fontSize: "16px", fontWeight: 700, color: "#0f172a", margin: "0 0 4px 0", lineHeight: "1.4" }}>
                      {cand.question}
                    </h4>
                    <p style={{ fontSize: "13px", color: "#475569", margin: 0, lineHeight: "1.5" }}>
                      {cand.rationale}
                    </p>
                  </div>

                  {/* Expand/Collapse Toggle for Secondary Evidence */}
                  {ev && (
                    <div>
                      <button
                        onClick={() => toggleExpand(cand.decision_id)}
                        style={{
                          background: "transparent",
                          border: "none",
                          padding: "0",
                          fontSize: "12px",
                          fontWeight: 600,
                          color: "#0f766e",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          gap: "4px",
                        }}
                      >
                        {isExpanded ? (
                          <>
                            <ChevronUp size={14} /> Hide Detailed Evidence Brief
                          </>
                        ) : (
                          <>
                            <ChevronDown size={14} /> View Empirical Evidence & Hypotheses ({ev.what_we_know.length + ev.what_is_unknown.length} facts)
                          </>
                        )}
                      </button>

                      {isExpanded && (
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 1fr",
                            gap: "12px",
                            background: "#f8fafc",
                            padding: "14px",
                            borderRadius: "6px",
                            border: "1px solid #e2e8f0",
                            fontSize: "12px",
                            marginTop: "10px",
                          }}
                        >
                          <div>
                            <div style={{ fontWeight: 700, color: "#0f766e", textTransform: "uppercase", fontSize: "10.5px", marginBottom: "4px" }}>
                              What We Know (Empirical Evidence)
                            </div>
                            <ul style={{ margin: 0, paddingLeft: "16px", color: "#334155", lineHeight: "1.5" }}>
                              {ev.what_we_know.map((item, i) => (
                                <li key={i}>{item}</li>
                              ))}
                            </ul>
                          </div>

                          <div>
                            <div style={{ fontWeight: 700, color: "#b45309", textTransform: "uppercase", fontSize: "10.5px", marginBottom: "4px" }}>
                              What Is Still Unknown (Empirical Hypothesis)
                            </div>
                            <ul style={{ margin: 0, paddingLeft: "16px", color: "#334155", lineHeight: "1.5" }}>
                              {ev.what_is_unknown.map((item, i) => (
                                <li key={i}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Footer: Affected Features, Available Strategies & Action Button */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      borderTop: "1px solid #f1f5f9",
                      paddingTop: "12px",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap" }}>
                      <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                        <span style={{ fontSize: "11.5px", color: "#64748b", fontWeight: 600 }}>Affected:</span>
                        {cand.affected_columns.map((c) => (
                          <code
                            key={c}
                            style={{
                              background: "#f1f5f9",
                              padding: "2px 6px",
                              borderRadius: "4px",
                              fontSize: "11.5px",
                              color: "#0f172a",
                            }}
                          >
                            {c}
                          </code>
                        ))}
                      </div>

                      <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                        <span style={{ fontSize: "11.5px", color: "#64748b", fontWeight: 600 }}>Strategies:</span>
                        <span style={{ fontSize: "11.5px", color: "#334155" }}>
                          {cand.available_strategies.join(" vs ")}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={() => onSelectDecision(cand)}
                      style={{
                        padding: "7px 14px",
                        fontSize: "12.5px",
                        fontWeight: 600,
                        borderRadius: "6px",
                        background: "#0f766e",
                        color: "#ffffff",
                        border: "none",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "6px",
                        boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                      }}
                    >
                      {isEvidenceReady ? "View Evidence Matrix →" : "Investigate in Decision Lab →"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
