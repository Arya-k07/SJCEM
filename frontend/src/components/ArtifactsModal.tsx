import React, { useState, useEffect } from "react";
import type { AuditReport, DecisionCandidate } from "../types/report";
import { api } from "../services/api";

interface ArtifactsModalProps {
  report: AuditReport;
  candidate: DecisionCandidate | null;
  onClose: () => void;
}

export const ArtifactsModal: React.FC<ArtifactsModalProps> = ({
  report,
  candidate,
  onClose,
}) => {
  if (!candidate) return null;

  const decisionId = candidate.decision_id;
  const runId = report.report_id;

  const [activePreviewTab, setActivePreviewTab] = useState<"pipeline" | "json">("pipeline");
  const [pipelineCode, setPipelineCode] = useState<string>("Loading reproducible pipeline script...");
  const [jsonContent, setJsonContent] = useState<string>("Loading decision contract JSON...");
  const [copied, setCopied] = useState<boolean>(false);

  const notebookUrl = api.getArtifactUrl(runId, decisionId, "notebook");
  const pipelineUrl = api.getArtifactUrl(runId, decisionId, "pipeline");
  const htmlUrl = api.getArtifactUrl(runId, decisionId, "html");
  const jsonUrl = api.getArtifactUrl(runId, decisionId, "json");

  useEffect(() => {
    // Fetch pipeline code preview
    fetch(pipelineUrl)
      .then((res) => res.text())
      .then((text) => setPipelineCode(text))
      .catch(() => setPipelineCode("# Failed to load pipeline code preview"));

    // Fetch JSON preview
    fetch(jsonUrl)
      .then((res) => res.json())
      .then((data) => setJsonContent(JSON.stringify(data, null, 2)))
      .catch(() => setJsonContent('{"error": "Failed to load JSON contract"}'));
  }, [runId, decisionId, pipelineUrl, jsonUrl]);

  const handleCopyCode = () => {
    const textToCopy = activePreviewTab === "pipeline" ? pipelineCode : jsonContent;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(15, 23, 42, 0.6)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "20px",
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "820px",
          maxWidth: "95vw",
          maxHeight: "90vh",
          backgroundColor: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #cbd5e1",
          boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid #e2e8f0",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: 700,
                  padding: "2px 8px",
                  borderRadius: "4px",
                  background: "#f0fdfa",
                  color: "#0f766e",
                  border: "1px solid #ccfbf1",
                }}
              >
                {decisionId} Reproducibility Pack
              </span>
              <span style={{ fontSize: "12px", color: "#64748b" }}>
                Selected: {candidate.user_decision?.selected_strategy_id || candidate.available_strategies[0]}
              </span>
            </div>
            <h3 style={{ fontSize: "18px", fontWeight: 800, color: "#0f172a", margin: 0 }}>
              Deterministic Code & Audit Artifacts
            </h3>
          </div>

          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#94a3b8",
              fontSize: "20px",
              cursor: "pointer",
              padding: "4px 8px",
              borderRadius: "4px",
            }}
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: "20px 24px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "20px" }}>
          <p style={{ fontSize: "13px", color: "#475569", margin: 0, lineHeight: "1.5" }}>
            These artifacts compile the exact within-fold preprocessing transformations, cross-validation protocol, and model evaluation parameters executed in Decision Lab into deterministic, production-ready code.
          </p>

          {/* Download Buttons Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            {/* Jupyter Notebook */}
            <div
              style={{
                padding: "12px 16px",
                borderRadius: "6px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: "13px", color: "#0f172a" }}>Jupyter Notebook (.ipynb)</div>
                <div style={{ fontSize: "11.5px", color: "#64748b" }}>Runnable within-fold CV analysis</div>
              </div>
              <a
                href={notebookUrl}
                download={`${decisionId}_workflow.ipynb`}
                style={{
                  padding: "6px 12px",
                  fontSize: "12px",
                  fontWeight: 700,
                  background: "#0f766e",
                  color: "#ffffff",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download
              </a>
            </div>

            {/* Python Pipeline */}
            <div
              style={{
                padding: "12px 16px",
                borderRadius: "6px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: "13px", color: "#0f172a" }}>Python Script (pipeline.py)</div>
                <div style={{ fontSize: "11.5px", color: "#64748b" }}>Scikit-Learn preprocessor script</div>
              </div>
              <a
                href={pipelineUrl}
                download={`${decisionId}_pipeline.py`}
                style={{
                  padding: "6px 12px",
                  fontSize: "12px",
                  fontWeight: 700,
                  background: "#334155",
                  color: "#ffffff",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download
              </a>
            </div>

            {/* HTML Audit Report */}
            <div
              style={{
                padding: "12px 16px",
                borderRadius: "6px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: "13px", color: "#0f172a" }}>HTML Decision Report</div>
                <div style={{ fontSize: "11.5px", color: "#64748b" }}>Standalone executive summary</div>
              </div>
              <a
                href={htmlUrl}
                download={`${decisionId}_report.html`}
                style={{
                  padding: "6px 12px",
                  fontSize: "12px",
                  fontWeight: 700,
                  background: "#f1f5f9",
                  color: "#334155",
                  border: "1px solid #cbd5e1",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download
              </a>
            </div>

            {/* JSON Contract */}
            <div
              style={{
                padding: "12px 16px",
                borderRadius: "6px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: "13px", color: "#0f172a" }}>JSON Metadata Contract</div>
                <div style={{ fontSize: "11.5px", color: "#64748b" }}>Machine-readable decision log</div>
              </div>
              <a
                href={jsonUrl}
                download={`${decisionId}_contract.json`}
                style={{
                  padding: "6px 12px",
                  fontSize: "12px",
                  fontWeight: 700,
                  background: "#f1f5f9",
                  color: "#334155",
                  border: "1px solid #cbd5e1",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download
              </a>
            </div>
          </div>

          {/* Interactive Code Preview Box */}
          <div
            style={{
              borderRadius: "6px",
              border: "1px solid #e2e8f0",
              background: "#0f172a",
              color: "#e2e8f0",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "8px 16px",
                background: "#1e293b",
                borderBottom: "1px solid #334155",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  onClick={() => setActivePreviewTab("pipeline")}
                  style={{
                    background: activePreviewTab === "pipeline" ? "#0f766e" : "transparent",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "4px",
                    padding: "4px 10px",
                    fontSize: "12px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  pipeline.py
                </button>
                <button
                  onClick={() => setActivePreviewTab("json")}
                  style={{
                    background: activePreviewTab === "json" ? "#0f766e" : "transparent",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "4px",
                    padding: "4px 10px",
                    fontSize: "12px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  decision_contract.json
                </button>
              </div>

              <button
                onClick={handleCopyCode}
                style={{
                  background: "transparent",
                  color: "#94a3b8",
                  border: "1px solid #475569",
                  borderRadius: "4px",
                  padding: "4px 10px",
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                {copied ? "✓ Copied" : "Copy Code"}
              </button>
            </div>

            <pre
              style={{
                margin: 0,
                padding: "16px",
                fontSize: "12px",
                lineHeight: "1.5",
                maxHeight: "220px",
                overflowY: "auto",
                fontFamily: 'Consolas, Monaco, "Courier New", monospace',
              }}
            >
              {activePreviewTab === "pipeline" ? pipelineCode : jsonContent}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
