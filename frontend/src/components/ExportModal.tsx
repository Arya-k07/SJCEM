import React, { useState } from "react";
import { X, Copy, Check, FileCode, FileText } from "lucide-react";
import type { AuditReport } from "../types/report";
import { api } from "../services/api";

interface ExportModalProps {
  report: AuditReport;
  onClose: () => void;
}

export const ExportModal: React.FC<ExportModalProps> = ({ report, onClose }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopyJSON = () => {
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `autopilot_report_${report.report_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleDownloadHTML = () => {
    window.open(api.getHtmlExportUrl(report.report_id), "_blank");
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        background: "rgba(0, 0, 0, 0.7)",
        backdropFilter: "blur(8px)",
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "760px",
          maxHeight: "85vh",
          background: "var(--bg-card-solid)",
          padding: "32px",
          display: "flex",
          flexDirection: "column",
          borderRadius: "var(--radius-xl)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
          <div>
            <h3 style={{ fontSize: "20px", fontWeight: 700 }}>Export & Download Audit Report</h3>
            <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "2px" }}>
              Report ID: <code>{report.report_id}</code>
            </p>
          </div>

          <button
            onClick={onClose}
            className="btn btn-secondary btn-sm"
            style={{ width: "32px", height: "32px", padding: 0 }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Action Buttons */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "20px" }}>
          <button className="btn btn-primary btn-sm" onClick={handleDownloadHTML}>
            <FileText size={15} /> Download HTML Report
          </button>

          <button className="btn btn-secondary btn-sm" onClick={handleDownloadJSON}>
            <FileCode size={15} /> Download JSON Data
          </button>

          <button className="btn btn-secondary btn-sm" onClick={handleCopyJSON}>
            {copied ? <Check size={15} color="var(--color-success)" /> : <Copy size={15} />}
            {copied ? "Copied!" : "Copy JSON to Clipboard"}
          </button>
        </div>

        {/* JSON Preview */}
        <div style={{ flex: 1, minHeight: "260px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "6px" }}>
            JSON Contract Preview
          </span>
          <pre
            style={{
              flex: 1,
              background: "rgba(0,0,0,0.4)",
              padding: "14px",
              borderRadius: "var(--radius-md)",
              fontFamily: "var(--font-mono)",
              fontSize: "12px",
              color: "var(--text-secondary)",
              overflowY: "auto",
              border: "1px solid var(--border-color)",
            }}
          >
            {JSON.stringify(report, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
};
