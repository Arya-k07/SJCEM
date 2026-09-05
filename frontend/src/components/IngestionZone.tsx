import React, { useState, useEffect } from "react";
import { UploadCloud, Play, Key, Sparkles, CheckCircle2 } from "lucide-react";
import type { SampleDataset } from "../types/report";
import { api } from "../services/api";

interface IngestionZoneProps {
  onStartAnalysis: (params: {
    file?: File;
    sampleId?: string;
    targetColumn?: string;
    timeColumn?: string;
    splitColumn?: string;
    geminiApiKey?: string;
  }) => void;
  isLoading: boolean;
}

export const IngestionZone: React.FC<IngestionZoneProps> = ({ onStartAnalysis, isLoading }) => {
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState<string>("");
  const [timeCol, setTimeCol] = useState<string>("");
  const [splitCol, setSplitCol] = useState<string>("");
  const [geminiApiKey, setGeminiApiKey] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [dragOver, setDragOver] = useState<boolean>(false);

  useEffect(() => {
    api.fetchSamples().then(setSamples).catch(() => {
      // Fallback default samples if server is warming up
      setSamples([
        {
          id: "telecom_churn_leakage",
          title: "Telecom Customer Churn",
          description: "Customer churn dataset containing post-event refund leakage and informative missingness.",
          target: "churn",
          file_path: "data/samples/telecom_churn_leakage.csv",
          badge: "Target Leakage Demo",
          color: "var(--color-critical)",
        },
        {
          id: "credit_risk_drift",
          title: "Credit Risk & Loan Default",
          description: "Loan application records with macroeconomic time-based distribution shift (PSI > 0.28).",
          target: "default_risk",
          time_col: "application_date",
          file_path: "data/samples/credit_risk_drift.csv",
          badge: "Distribution Drift Demo",
          color: "var(--color-warning)",
        },
        {
          id: "titanic_quality_audit",
          title: "Titanic Passenger Survival",
          description: "Classical survival dataset featuring structured missingness (MNAR) and extreme fare outliers.",
          target: "survived",
          file_path: "data/samples/titanic_quality_audit.csv",
          badge: "Data Quality & Outliers",
          color: "var(--color-info)",
        },
        {
          id: "recruitment_bias_slices",
          title: "Recruitment & Hiring Pipeline",
          description: "Job applicant screening data with candidate demographic slices and disparate impact.",
          target: "hired",
          file_path: "data/samples/recruitment_bias_slices.csv",
          badge: "Slice Fairness & Bias",
          color: "var(--accent-teal)",
        },
      ]);
    });
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmitUpload = () => {
    if (!selectedFile) return;
    onStartAnalysis({
      file: selectedFile,
      targetColumn: targetCol.trim() || undefined,
      timeColumn: timeCol.trim() || undefined,
      splitColumn: splitCol.trim() || undefined,
      geminiApiKey: geminiApiKey.trim() || undefined,
    });
  };

  const handleSelectSample = (sample: SampleDataset) => {
    onStartAnalysis({
      sampleId: sample.id,
      geminiApiKey: geminiApiKey.trim() || undefined,
    });
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "32px auto", padding: "0 20px" }}>
      {/* Hero Title */}
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h1
          style={{
            fontSize: "32px",
            fontWeight: 800,
            letterSpacing: "-0.02em",
            marginBottom: "10px",
            color: "var(--text-primary)",
          }}
        >
          Decision Lab for Tabular Datasets
        </h1>
        <p style={{ fontSize: "15px", color: "var(--text-secondary)", maxWidth: "680px", margin: "0 auto", lineHeight: "1.6" }}>
          Convert raw data quality observations into controlled empirical experiments. Compare remediation strategies with seed-locked cross-validation, make defensible decisions, and export reproducible pipelines.
        </p>
      </div>

      {/* 1-Click Benchmark Demo Datasets */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
          <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            1-Click Benchmark Datasets
          </span>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Instant demonstration of decision workflows</span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "12px",
          }}
        >
          {samples.map((s) => (
            <div
              key={s.id}
              className="glass-panel"
              onClick={() => !isLoading && handleSelectSample(s)}
              style={{
                padding: "16px",
                cursor: isLoading ? "not-allowed" : "pointer",
                borderLeft: `4px solid ${s.color}`,
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                <span
                  style={{
                    fontSize: "10.5px",
                    fontWeight: 700,
                    color: s.color,
                    background: "var(--bg-subtle)",
                    padding: "2px 7px",
                    borderRadius: "4px",
                    border: "1px solid var(--border-color)",
                  }}
                >
                  {s.badge}
                </span>
                <Play size={13} color={s.color} />
              </div>
              <div style={{ fontWeight: 700, fontSize: "14px", marginBottom: "4px", color: "var(--text-primary)" }}>
                {s.title}
              </div>
              <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: 1.4 }}>{s.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Upload Box */}
      <div className="glass-panel" style={{ padding: "28px", marginBottom: "24px" }}>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragOver ? "var(--accent-teal)" : "var(--border-color)"}`,
            borderRadius: "var(--radius-md)",
            padding: "36px 20px",
            textAlign: "center",
            background: dragOver ? "var(--accent-teal-subtle)" : "var(--bg-subtle)",
            transition: "all 0.15s ease",
            cursor: "pointer",
          }}
          onClick={() => document.getElementById("file-upload-input")?.click()}
        >
          <input
            id="file-upload-input"
            type="file"
            accept=".csv,.parquet,.pq,.json,.jsonl,.tsv,.xlsx"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />

          <div
            style={{
              width: "48px",
              height: "48px",
              borderRadius: "50%",
              background: "var(--accent-teal-subtle)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 12px",
            }}
          >
            <UploadCloud size={24} color="var(--accent-teal)" />
          </div>

          {selectedFile ? (
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", fontWeight: 700, fontSize: "15px", color: "var(--accent-teal)" }}>
                <CheckCircle2 size={16} /> {selectedFile.name}
              </div>
              <div style={{ fontSize: "12.5px", color: "var(--text-muted)", marginTop: "4px" }}>
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for audit
              </div>
            </div>
          ) : (
            <div>
              <div style={{ fontWeight: 700, fontSize: "15px", marginBottom: "4px", color: "var(--text-primary)" }}>
                Drop your dataset here or <span style={{ color: "var(--accent-teal)" }}>browse files</span>
              </div>
              <div style={{ fontSize: "12.5px", color: "var(--text-muted)" }}>
                Supports CSV, Parquet, TSV, JSON, and Excel up to 250MB
              </div>
            </div>
          )}
        </div>

        {/* Configuration inputs */}
        <div style={{ marginTop: "20px", display: "grid", gridTemplateColumns: "1fr", gap: "14px" }}>
          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "6px" }}>
              Target Column (Optional — Leave blank for Unsupervised Profiling)
            </label>
            <input
              type="text"
              placeholder="e.g. churn, default_risk, survived"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              style={{
                width: "100%",
                padding: "9px 12px",
                borderRadius: "var(--radius-sm)",
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                outline: "none",
              }}
            />
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div style={{ marginTop: "14px" }}>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            style={{
              background: "none",
              border: "none",
              color: "var(--accent-teal)",
              fontSize: "12.5px",
              fontWeight: 600,
              cursor: "pointer",
              padding: 0,
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            {showAdvanced ? "▾ Hide Advanced Options" : "▸ Show Advanced Options (Timestamp, Split, API Key)"}
          </button>

          {showAdvanced && (
            <div style={{ marginTop: "14px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              <div>
                <label style={{ display: "block", fontSize: "12.5px", color: "var(--text-muted)", marginBottom: "4px" }}>
                  Timestamp Column (for temporal drift)
                </label>
                <input
                  type="text"
                  placeholder="e.g. created_at, transaction_date"
                  value={timeCol}
                  onChange={(e) => setTimeCol(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    background: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "12.5px", color: "var(--text-muted)", marginBottom: "4px" }}>
                  Partition / Split Column
                </label>
                <input
                  type="text"
                  placeholder="e.g. split, dataset_fold"
                  value={splitCol}
                  onChange={(e) => setSplitCol(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    background: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div style={{ gridColumn: "1 / -1" }}>
                <label style={{ display: "block", fontSize: "12.5px", color: "var(--text-muted)", marginBottom: "4px" }}>
                  Developer Gemini API Key (Optional — Server env default is active)
                </label>
                <div style={{ position: "relative" }}>
                  <input
                    type="password"
                    placeholder="Optional: Enter custom Gemini key or leave blank for server/deterministic default"
                    value={geminiApiKey}
                    onChange={(e) => setGeminiApiKey(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "8px 12px 8px 34px",
                      borderRadius: "var(--radius-sm)",
                      background: "var(--bg-primary)",
                      border: "1px solid var(--border-color)",
                      color: "var(--text-primary)",
                    }}
                  />
                  <Key size={14} color="var(--text-muted)" style={{ position: "absolute", left: "10px", top: "11px" }} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Start Button */}
        <div style={{ marginTop: "24px", display: "flex", justifyContent: "flex-end" }}>
          <button
            className="btn btn-primary"
            onClick={handleSubmitUpload}
            disabled={!selectedFile || isLoading}
            style={{ opacity: !selectedFile || isLoading ? 0.5 : 1, padding: "10px 24px", fontSize: "14px" }}
          >
            <Sparkles size={15} /> Run Autonomous Audit
          </button>
        </div>
      </div>
    </div>
  );
};
