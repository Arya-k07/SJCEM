import { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { IngestionZone } from "./components/IngestionZone";
import { ProgressTracker } from "./components/ProgressTracker";
import { DecisionWorkbench } from "./components/DecisionWorkbench";
import { DecisionLabView } from "./components/DecisionLabView";
import { FindingsTable } from "./components/FindingsTable";
import { FindingDrawer } from "./components/FindingDrawer";
import { ProfilingTab } from "./components/ProfilingTab";
import { DriftTab } from "./components/DriftTab";
import { FairnessTab } from "./components/FairnessTab";
import { ArtifactsModal } from "./components/ArtifactsModal";
import { ExportModal } from "./components/ExportModal";
import type { AuditReport, DecisionCandidate, Finding, JobStatus } from "./types/report";
import { api } from "./services/api";
import { AlertCircle, FileSearch, TrendingUp, Users, FlaskConical, LayoutGrid } from "lucide-react";

export function App() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [report, setReport] = useState<AuditReport | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [activeTab, setActiveTab] = useState<string>("decisions");
  const [activeDecision, setActiveDecision] = useState<DecisionCandidate | null>(null);
  const [artifactsCandidate, setArtifactsCandidate] = useState<DecisionCandidate | null>(null);
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(null);
  const [showExportModal, setShowExportModal] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Sync theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Polling for ongoing audit job
  useEffect(() => {
    if (!currentJobId) return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(currentJobId);
        setJobStatus(status);

        if (status.status === "completed") {
          clearInterval(interval);
          const fullReport = await api.getReport(currentJobId);
          setReport(fullReport);
          setCurrentJobId(null);
        } else if (status.status === "failed") {
          clearInterval(interval);
          setErrorMessage(status.error_message || "Audit failed during processing.");
          setCurrentJobId(null);
        }
      } catch (err: any) {
        clearInterval(interval);
        setErrorMessage(err.message || "Lost connection to audit engine.");
        setCurrentJobId(null);
      }
    }, 800);

    return () => clearInterval(interval);
  }, [currentJobId]);

  const handleStartAnalysis = async (params: {
    file?: File;
    sampleId?: string;
    targetColumn?: string;
    timeColumn?: string;
    splitColumn?: string;
    geminiApiKey?: string;
  }) => {
    setErrorMessage(null);
    setReport(null);
    setActiveDecision(null);

    try {
      if (params.sampleId) {
        const res = await api.triggerSampleAnalysis(params.sampleId, params.geminiApiKey);
        setCurrentJobId(res.job_id);
        setJobStatus({
          job_id: res.job_id,
          status: "queued",
          progress_pct: 5,
          current_stage: "Queuing benchmark dataset...",
          created_at: new Date().toISOString(),
        });
      } else if (params.file) {
        const formData = new FormData();
        formData.append("file", params.file);
        if (params.targetColumn) formData.append("target_column", params.targetColumn);
        if (params.timeColumn) formData.append("time_column", params.timeColumn);
        if (params.splitColumn) formData.append("split_column", params.splitColumn);
        if (params.geminiApiKey) formData.append("gemini_api_key", params.geminiApiKey);

        const res = await api.uploadAndAnalyze(formData);
        setCurrentJobId(res.job_id);
        setJobStatus({
          job_id: res.job_id,
          status: "queued",
          progress_pct: 5,
          current_stage: "Ingesting file...",
          created_at: new Date().toISOString(),
        });
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to initiate audit.");
    }
  };

  const handleReset = () => {
    setReport(null);
    setCurrentJobId(null);
    setJobStatus(null);
    setErrorMessage(null);
    setActiveDecision(null);
    setActiveTab("decisions");
  };

  const handleDecisionUpdated = (updatedCandidate: DecisionCandidate) => {
    setActiveDecision(updatedCandidate);
    if (report) {
      const updatedList = (report.decision_candidates || []).map((c) =>
        c.decision_id === updatedCandidate.decision_id ? updatedCandidate : c
      );
      setReport({
        ...report,
        decision_candidates: updatedList,
      });
    }
  };

  const tabs = [
    { id: "decisions", label: "Decision Lab", icon: FlaskConical, count: report?.decision_candidates?.length },
    { id: "profiling", label: "Schema & Profiling", icon: FileSearch, count: report?.dataset_profile?.columns },
    { id: "findings", label: "All Observations (Raw)", icon: LayoutGrid, count: report?.findings?.length },
    { id: "drift", label: "Drift Analysis", icon: TrendingUp, count: report?.drift_summary?.filter((d) => d.is_drift_detected).length },
    { id: "fairness", label: "Slice Fairness", icon: Users, count: report?.slice_summary?.filter((s) => s.is_disparate).length },
  ];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Header
        theme={theme}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        onNewAudit={handleReset}
        hasReport={!!report}
        report={report}
        onOpenExport={() => setShowExportModal(true)}
      />

      <main style={{ flex: 1, padding: "24px", maxWidth: "1280px", margin: "0 auto", width: "100%" }}>
        {/* Error Alert */}
        {errorMessage && (
          <div
            className="glass-panel"
            style={{
              padding: "16px 20px",
              marginBottom: "24px",
              background: "var(--bg-critical)",
              border: "1px solid var(--border-critical)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px", color: "var(--color-critical)" }}>
              <AlertCircle size={20} />
              <span style={{ fontSize: "14px", fontWeight: 600 }}>{errorMessage}</span>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => setErrorMessage(null)}>
              Dismiss
            </button>
          </div>
        )}

        {/* View 1: Ingestion Zone */}
        {!report && !currentJobId && (
          <IngestionZone onStartAnalysis={handleStartAnalysis} isLoading={!!currentJobId} />
        )}

        {/* View 2: Progress Tracker */}
        {!report && currentJobId && jobStatus && (
          <ProgressTracker status={jobStatus} />
        )}

        {/* View 3: Decision Lab & Audit Workspace */}
        {report && (
          <div className="animate-fade-in">
            {/* Top Navigation Tabs Bar */}
            <div
              style={{
                display: "flex",
                gap: "8px",
                borderBottom: "1px solid var(--border-color)",
                marginBottom: "24px",
                overflowX: "auto",
                paddingBottom: "4px",
              }}
            >
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setActiveTab(tab.id);
                      if (tab.id !== "decisions") {
                        setActiveDecision(null);
                      }
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      padding: "10px 16px",
                      background: isActive ? "var(--accent-teal-subtle)" : "transparent",
                      border: "none",
                      borderBottom: `2px solid ${isActive ? "var(--accent-teal)" : "transparent"}`,
                      borderRadius: "6px 6px 0 0",
                      color: isActive ? "var(--accent-teal)" : "var(--text-secondary)",
                      fontWeight: isActive ? 700 : 500,
                      cursor: "pointer",
                      fontSize: "13.5px",
                      whiteSpace: "nowrap",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <Icon size={16} />
                    <span>{tab.label}</span>
                    {tab.count !== undefined && tab.count > 0 && (
                      <span
                        style={{
                          fontSize: "11px",
                          padding: "1px 6px",
                          borderRadius: "9999px",
                          background: isActive ? "var(--accent-teal)" : "var(--bg-subtle)",
                          color: isActive ? "#ffffff" : "var(--text-muted)",
                          fontWeight: 700,
                        }}
                      >
                        {tab.count}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Tab Contents */}
            <div>
              {activeTab === "decisions" && (
                activeDecision ? (
                  <DecisionLabView
                    report={report}
                    candidate={activeDecision}
                    onBack={() => setActiveDecision(null)}
                    onDecisionUpdated={handleDecisionUpdated}
                    onOpenArtifacts={(c) => setArtifactsCandidate(c)}
                  />
                ) : (
                  <DecisionWorkbench
                    report={report}
                    onSelectDecision={(c) => setActiveDecision(c)}
                  />
                )
              )}

              {activeTab === "profiling" && (
                <ProfilingTab profile={report.dataset_profile} />
              )}

              {activeTab === "findings" && (
                <FindingsTable
                  findings={report.findings}
                  onSelectFinding={(f) => setSelectedFinding(f)}
                />
              )}

              {activeTab === "drift" && (
                <DriftTab report={report} />
              )}

              {activeTab === "fairness" && (
                <FairnessTab report={report} />
              )}
            </div>
          </div>
        )}

        {/* Deep Dive Finding Drawer */}
        <FindingDrawer
          finding={selectedFinding}
          onClose={() => setSelectedFinding(null)}
        />

        {/* Reproducibility Artifacts Modal for a Specific Decision */}
        {artifactsCandidate && report && (
          <ArtifactsModal
            report={report}
            candidate={artifactsCandidate}
            onClose={() => setArtifactsCandidate(null)}
          />
        )}

        {/* Full Report Export Modal */}
        {showExportModal && report && (
          <ExportModal
            report={report}
            onClose={() => setShowExportModal(false)}
          />
        )}
      </main>
    </div>
  );
}

export default App;
