import React from "react";
import { Moon, Sun, Database, FileText, RotateCcw } from "lucide-react";
import type { AuditReport } from "../types/report";

interface HeaderProps {
  theme: "dark" | "light";
  onToggleTheme: () => void;
  onNewAudit: () => void;
  hasReport: boolean;
  report?: AuditReport | null;
  onOpenExport?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  theme,
  onToggleTheme,
  onNewAudit,
  hasReport,
  report,
  onOpenExport,
}) => {
  return (
    <header
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "12px 24px",
        borderBottom: "1px solid #e2e8f0",
        background: "#ffffff",
        position: "sticky",
        top: 0,
        zIndex: 50,
        boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div
          style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }}
          onClick={onNewAudit}
          title="Return to Ingestion"
        >
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "6px",
              background: "#0f766e",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Database size={16} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ fontSize: "16px", fontWeight: 800, color: "#0f172a", letterSpacing: "-0.01em" }}>
                Dataset <span style={{ color: "#0f766e" }}>Autopilot</span>
              </span>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  padding: "1px 5px",
                  borderRadius: "4px",
                  background: "#f1f5f9",
                  color: "#475569",
                  border: "1px solid #cbd5e1",
                }}
              >
                Decision Lab
              </span>
            </div>
          </div>
        </div>

        {/* Active Dataset Context Badge */}
        {hasReport && report && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              paddingLeft: "16px",
              borderLeft: "1px solid #e2e8f0",
              fontSize: "12px",
              color: "#64748b",
            }}
          >
            <span style={{ fontWeight: 600, color: "#0f172a" }}>{report.dataset_name}</span>
            <span>•</span>
            <span>{report.dataset_profile.rows.toLocaleString()} rows</span>
            <span>•</span>
            <span>{report.dataset_profile.columns} columns</span>
            {report.dataset_profile.detected_target && (
              <>
                <span>•</span>
                <span>
                  Target: <strong>{report.dataset_profile.detected_target}</strong>
                </span>
              </>
            )}
          </div>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        {hasReport && (
          <>
            <button
              onClick={onOpenExport}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "6px 12px",
                fontSize: "12.5px",
                fontWeight: 600,
                color: "#334155",
                background: "#f8fafc",
                border: "1px solid #cbd5e1",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              <FileText size={14} /> Export Report
            </button>

            <button
              onClick={onNewAudit}
              title="Reset workspace and audit a new dataset"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "6px 12px",
                fontSize: "12.5px",
                fontWeight: 600,
                color: "#64748b",
                background: "transparent",
                border: "1px solid #e2e8f0",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              <RotateCcw size={13} /> New Audit
            </button>
          </>
        )}

        <button
          onClick={onToggleTheme}
          style={{
            width: "32px",
            height: "32px",
            padding: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "transparent",
            border: "1px solid #e2e8f0",
            borderRadius: "6px",
            cursor: "pointer",
            color: "#64748b",
          }}
          title="Toggle Theme"
        >
          {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  );
};
