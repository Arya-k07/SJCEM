import React from "react";
import { X, AlertOctagon, AlertTriangle, Info, CheckCircle } from "lucide-react";
import type { Finding } from "../types/report";

interface FindingDrawerProps {
  finding: Finding | null;
  onClose: () => void;
}

export const FindingDrawer: React.FC<FindingDrawerProps> = ({ finding, onClose }) => {
  if (!finding) return null;

  const getSeverityIcon = () => {
    if (finding.severity === "CRITICAL") return <AlertOctagon size={20} color="var(--color-critical)" />;
    if (finding.severity === "WARNING") return <AlertTriangle size={20} color="var(--color-warning)" />;
    return <Info size={20} color="var(--color-info)" />;
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        background: "rgba(0, 0, 0, 0.65)",
        backdropFilter: "blur(6px)",
        zIndex: 100,
        display: "flex",
        justifyContent: "flex-end",
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "580px",
          height: "100%",
          borderRadius: 0,
          background: "var(--bg-card-solid)",
          padding: "32px",
          overflowY: "auto",
          boxShadow: "-10px 0 30px rgba(0,0,0,0.5)",
          borderLeft: "1px solid var(--border-color)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {getSeverityIcon()}
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className={`badge badge-${finding.severity.toLowerCase()}`}>
                  {finding.severity}
                </span>
                <span style={{ fontSize: "12px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  {finding.id}
                </span>
              </div>
              <span style={{ fontSize: "12px", color: "var(--text-muted)", textTransform: "capitalize", marginTop: "2px" }}>
                {finding.type.replace(/_/g, " ")}
              </span>
            </div>
          </div>

          <button
            onClick={onClose}
            className="btn btn-secondary btn-sm"
            style={{ width: "32px", height: "32px", padding: 0 }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Title & Description */}
        <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "10px" }}>{finding.title}</h3>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: "24px" }}>
          {finding.description}
        </p>

        {/* Column Affected */}
        {finding.column && (
          <div style={{ marginBottom: "20px" }}>
            <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Affected Column
            </span>
            <div style={{ marginTop: "4px" }}>
              <code style={{ fontSize: "14px", padding: "4px 8px" }}>{finding.column}</code>
            </div>
          </div>
        )}

        {/* Actionable Remediation Box */}
        <div
          style={{
            background: "rgba(16, 185, 129, 0.08)",
            border: "1px solid rgba(16, 185, 129, 0.25)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
            marginBottom: "24px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-success)", fontWeight: 700, fontSize: "13px", marginBottom: "6px" }}>
            <CheckCircle size={16} /> Recommended Action & Remediation
          </div>
          <p style={{ fontSize: "13.5px", color: "var(--text-primary)", lineHeight: 1.5 }}>
            {finding.recommendation}
          </p>
        </div>

        {/* Evidence Dictionary */}
        <div>
          <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "10px" }}>
            Empirical Evidence & Metrics
          </span>

          <div
            style={{
              background: "rgba(0, 0, 0, 0.25)",
              border: "1px solid var(--border-color)",
              borderRadius: "var(--radius-md)",
              padding: "16px",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <tbody>
                {Object.entries(finding.evidence).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <td style={{ padding: "8px 4px", color: "var(--text-muted)", fontWeight: 600, width: "40%" }}>
                      {k.replace(/_/g, " ")}
                    </td>
                    <td style={{ padding: "8px 4px", color: "var(--text-primary)", fontFamily: typeof v === "number" ? "var(--font-mono)" : "inherit" }}>
                      {typeof v === "object" && v !== null ? (
                        <pre style={{ margin: 0, fontSize: "11.5px", background: "rgba(0,0,0,0.3)", padding: "6px", borderRadius: "4px" }}>
                          {JSON.stringify(v, null, 2)}
                        </pre>
                      ) : (
                        String(v)
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confidence rating */}
        <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "12px", color: "var(--text-muted)" }}>
          <span>Detection Confidence: <strong>{Math.round(finding.confidence * 100)}%</strong></span>
        </div>
      </div>
    </div>
  );
};
