import React, { useState, useEffect } from "react";
import type { AuditReport, DecisionCandidate, ExperimentVariantResult } from "../types/report";
import { api } from "../services/api";

interface DecisionLabViewProps {
  report: AuditReport;
  candidate: DecisionCandidate;
  onBack: () => void;
  onDecisionUpdated: (updated: DecisionCandidate) => void;
  onOpenArtifacts: (candidate: DecisionCandidate) => void;
}

export const DecisionLabView: React.FC<DecisionLabViewProps> = ({
  report,
  candidate,
  onBack,
  onDecisionUpdated,
  onOpenArtifacts,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [executionStep, setExecutionStep] = useState<string>("idle");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [selectedStrategiesToTest, setSelectedStrategiesToTest] = useState<string[]>(
    candidate.available_strategies || []
  );
  const [selectedStrategy, setSelectedStrategy] = useState<string>(
    candidate.user_decision?.selected_strategy_id || candidate.available_strategies[0] || ""
  );
  const [rationale, setRationale] = useState<string>(
    candidate.user_decision?.decision_rationale || ""
  );
  const [customMethodNotes, setCustomMethodNotes] = useState<string>(
    candidate.user_decision?.custom_method_notes || ""
  );
  const [isSavingDecision, setIsSavingDecision] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);

  const results: ExperimentVariantResult[] = candidate.experiment_results || [];
  const hasResults = results.length > 0;
  const isDecided = candidate.status === "USER_DECISION";
  const ev = candidate.evidence_block;
  const targetCol = report.dataset_profile.detected_target || "target";
  const isTemporal = Boolean(report.analysis_context?.time_column);

  useEffect(() => {
    let interval: any;
    if (isRunning) {
      const startTime = Date.now();
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const toggleStrategySelection = (stratId: string) => {
    if (selectedStrategiesToTest.includes(stratId)) {
      if (selectedStrategiesToTest.length <= 2) {
        setFeedbackMessage({
          text: "At least 2 strategies are required for a controlled comparative experiment.",
          type: "error",
        });
        return;
      }
      setSelectedStrategiesToTest(selectedStrategiesToTest.filter((s) => s !== stratId));
    } else {
      setSelectedStrategiesToTest([...selectedStrategiesToTest, stratId]);
    }
  };

  const handleRunExperiments = async () => {
    try {
      setIsRunning(true);
      setFeedbackMessage(null);
      setExecutionStep("Preparing isolated training folds...");

      const timer1 = setTimeout(() => setExecutionStep("Fitting transformers strictly on Fold 1/5..."), 400);
      const timer2 = setTimeout(() => setExecutionStep("Evaluating out-of-fold validation on Fold 3/5..."), 1200);
      const timer3 = setTimeout(() => setExecutionStep("Comparing empirical metrics across variants..."), 2000);

      const updated = await api.runCandidateExperiments(
        report.report_id,
        candidate.decision_id,
        selectedStrategiesToTest
      );

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);

      onDecisionUpdated(updated);
      if (updated.experiment_results && updated.experiment_results.length > 0) {
        setSelectedStrategy(updated.experiment_results[0].strategy_id);
      }
      setFeedbackMessage({
        text: "Cross-validation evaluation complete. Review the empirical evidence matrix below.",
        type: "success",
      });
    } catch (err: any) {
      setFeedbackMessage({
        text: `Experiment execution failed: ${err.message}`,
        type: "error",
      });
    } finally {
      setIsRunning(false);
      setExecutionStep("idle");
    }
  };

  const handleSaveDecision = async () => {
    if (!selectedStrategy) {
      setFeedbackMessage({
        text: "Please select a strategy from the matrix to record your decision.",
        type: "error",
      });
      return;
    }
    try {
      setIsSavingDecision(true);
      setFeedbackMessage(null);
      const updated = await api.recordDecision(report.report_id, candidate.decision_id, {
        selected_strategy_id: selectedStrategy,
        decision_rationale: rationale || `Adopted '${selectedStrategy}' based on controlled empirical validation.`,
        custom_method_notes: customMethodNotes || undefined,
      });
      onDecisionUpdated(updated);
      setFeedbackMessage({
        text: "Decision successfully recorded. Reproducible code artifacts compiled and ready.",
        type: "success",
      });
    } catch (err: any) {
      setFeedbackMessage({
        text: `Failed to save decision: ${err.message}`,
        type: "error",
      });
    } finally {
      setIsSavingDecision(false);
    }
  };

  const formatElapsed = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }} className="animate-fade-in">
      {/* Top Breadcrumb & Action Tray */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <button
          onClick={onBack}
          style={{
            background: "transparent",
            border: "1px solid #cbd5e1",
            borderRadius: "6px",
            padding: "6px 12px",
            fontSize: "12.5px",
            fontWeight: 600,
            color: "#334155",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          ← Back to Decisions Workbench
        </button>

        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          {isDecided && (
            <button
              onClick={() => onOpenArtifacts(candidate)}
              style={{
                background: "#f0fdfa",
                border: "1px solid #0f766e",
                borderRadius: "6px",
                padding: "6px 14px",
                fontSize: "12.5px",
                fontWeight: 600,
                color: "#0f766e",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              📥 Download Artifacts (.ipynb, .py)
            </button>
          )}
          <span style={{ fontSize: "11.5px", fontWeight: 700, padding: "2px 8px", borderRadius: "4px", background: "#f1f5f9", color: "#334155" }}>
            {candidate.decision_id}
          </span>
          <span style={{ fontSize: "11.5px", fontWeight: 600, padding: "2px 8px", borderRadius: "4px", background: "#f8fafc", color: "#475569", border: "1px solid #e2e8f0" }}>
            {candidate.category.replace("_", " ")}
          </span>
        </div>
      </div>

      {/* Feedback Banner (In-App notification replacing alert()) */}
      {feedbackMessage && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: 500,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: feedbackMessage.type === "success" ? "#f0fdfa" : feedbackMessage.type === "error" ? "#fef2f2" : "#f8fafc",
            color: feedbackMessage.type === "success" ? "#0f766e" : feedbackMessage.type === "error" ? "#b91c1c" : "#334155",
            border: `1px solid ${feedbackMessage.type === "success" ? "#ccfbf1" : feedbackMessage.type === "error" ? "#fecaca" : "#e2e8f0"}`,
          }}
        >
          <span>{feedbackMessage.text}</span>
          <button
            onClick={() => setFeedbackMessage(null)}
            style={{ background: "transparent", border: "none", color: "inherit", cursor: "pointer", fontWeight: 700 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Decision Question Headline Banner */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          borderLeft: "4px solid #0f766e",
          padding: "20px 24px",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.02)",
        }}
      >
        <div style={{ fontSize: "11px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Investigation Objective & Decision Question
        </div>
        <h2 style={{ fontSize: "19px", fontWeight: 800, color: "#0f172a", margin: 0, lineHeight: "1.4" }}>
          {candidate.question}
        </h2>
        <p style={{ fontSize: "13.5px", color: "#475569", margin: 0, lineHeight: "1.5" }}>
          {candidate.rationale}
        </p>
        <div style={{ display: "flex", gap: "16px", alignItems: "center", marginTop: "2px", flexWrap: "wrap", fontSize: "12px" }}>
          <div>
            <span style={{ color: "#64748b", fontWeight: 600 }}>Target: </span>
            <code style={{ background: "#f1f5f9", padding: "2px 6px", borderRadius: "4px" }}>{targetCol}</code>
          </div>
          <div>
            <span style={{ color: "#64748b", fontWeight: 600 }}>Affected Features: </span>
            {candidate.affected_columns.map((c) => (
              <code key={c} style={{ background: "#f1f5f9", padding: "2px 6px", borderRadius: "4px", marginRight: "4px" }}>
                {c}
              </code>
            ))}
          </div>
        </div>
      </div>

      {/* Structured Evidence Brief */}
      {ev && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "14px",
            background: "#f8fafc",
            padding: "16px",
            borderRadius: "6px",
            border: "1px solid #e2e8f0",
            fontSize: "12px",
          }}
        >
          <div>
            <div style={{ fontWeight: 700, color: "#0f766e", textTransform: "uppercase", fontSize: "10.5px", marginBottom: "6px" }}>
              What We Know (Empirical Evidence)
            </div>
            <ul style={{ margin: 0, paddingLeft: "16px", color: "#334155", lineHeight: "1.5" }}>
              {ev.what_we_know.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>

          <div>
            <div style={{ fontWeight: 700, color: "#b45309", textTransform: "uppercase", fontSize: "10.5px", marginBottom: "6px" }}>
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

      {/* Controlled Parameter Ledger ("We Are Changing ONE Thing") */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          padding: "18px 24px",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "24px",
        }}
      >
        <div>
          <div style={{ fontSize: "11px", fontWeight: 700, color: "#0f766e", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "6px" }}>
            Controlled Protocol (What Stays Constant)
          </div>
          <ul style={{ margin: 0, paddingLeft: "18px", fontSize: "12.5px", color: "#334155", lineHeight: "1.6" }}>
            <li><strong>Model Architecture:</strong> Random Forest (50 trees, max_depth=6)</li>
            <li><strong>Validation Protocol:</strong> {isTemporal ? "Chronological TimeSeriesSplit (Zero lookahead)" : "5-Fold Stratified Cross-Validation (Seed=42)"}</li>
            <li><strong>Fold Isolation:</strong> Transformers fitted strictly inside training folds</li>
            <li><strong>Target Metric:</strong> {results[0]?.primary_metric_name || "ROC-AUC"}</li>
          </ul>
        </div>

        <div>
          <div style={{ fontSize: "11px", fontWeight: 700, color: "#b45309", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "6px" }}>
            Experimental Variable (What Changes)
          </div>
          <p style={{ margin: 0, fontSize: "12.5px", color: "#334155", lineHeight: "1.5" }}>
            <strong>Treatment Strategy Variant:</strong> Isolating the empirical performance delta of candidate preprocessing transformations.
          </p>
          <div style={{ marginTop: "8px", fontSize: "11.5px", color: "#64748b", background: "#f8fafc", padding: "6px 10px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
            Scientific Guarantee: Seeds, splits, feature subset definitions, and model hyperparameters remain identical across all variants.
          </div>
        </div>
      </div>

      {/* Strategy Selection & Execution Controller */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          padding: "18px 24px",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
        }}
      >
        <div>
          <div style={{ fontSize: "13.5px", fontWeight: 700, color: "#0f172a", marginBottom: "2px" }}>
            Select Strategies for Controlled Comparison
          </div>
          <p style={{ fontSize: "12.5px", color: "#64748b", margin: 0 }}>
            Choose which candidate strategies to evaluate under the locked cross-validation protocol (minimum 2 required):
          </p>
        </div>

        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {candidate.available_strategies.map((stratId) => {
            const isChecked = selectedStrategiesToTest.includes(stratId);
            return (
              <button
                key={stratId}
                onClick={() => toggleStrategySelection(stratId)}
                disabled={isRunning}
                style={{
                  padding: "6px 12px",
                  borderRadius: "6px",
                  fontSize: "12px",
                  fontWeight: 600,
                  cursor: isRunning ? "not-allowed" : "pointer",
                  border: isChecked ? "1px solid #0f766e" : "1px solid #cbd5e1",
                  background: isChecked ? "#f0fdfa" : "#ffffff",
                  color: isChecked ? "#0f766e" : "#475569",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span>{isChecked ? "✓" : "○"}</span>
                <span>{stratId.replace(/_/g, " ")}</span>
              </button>
            );
          })}
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid #f1f5f9", paddingTop: "12px" }}>
          <div style={{ fontSize: "12px", color: "#64748b" }}>
            {selectedStrategiesToTest.length} variants selected for evaluation
          </div>

          <button
            onClick={handleRunExperiments}
            disabled={isRunning || selectedStrategiesToTest.length < 2}
            style={{
              padding: "9px 18px",
              borderRadius: "6px",
              fontSize: "13px",
              fontWeight: 700,
              background: isRunning ? "#94a3b8" : "#0f766e",
              color: "#ffffff",
              border: "none",
              cursor: isRunning ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}
          >
            {isRunning ? `${executionStep} (${formatElapsed(elapsedSeconds)})` : hasResults ? "Re-Run Controlled A/B Test" : "Run Controlled A/B Test"}
          </button>
        </div>
      </div>

      {/* Real-Time Progress Visualizer */}
      {isRunning && (
        <div
          style={{
            background: "#f0fdfa",
            borderRadius: "8px",
            border: "1px solid #ccfbf1",
            padding: "16px 20px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "12.5px", fontWeight: 700, color: "#0f766e" }}>
              Controlled Cross-Validation in Progress
            </span>
            <span style={{ fontSize: "12px", fontWeight: 600, color: "#0f766e" }}>
              Elapsed {formatElapsed(elapsedSeconds)}
            </span>
          </div>
          <div style={{ fontSize: "13px", color: "#134e4a", fontWeight: 500 }}>
            {executionStep}
          </div>
          <div style={{ fontSize: "11.5px", color: "#64748b" }}>
            Seed-locked random forest training across {selectedStrategiesToTest.length} variants with strict fold isolation.
          </div>
        </div>
      )}

      {/* Empirical Results Matrix */}
      {hasResults && (
        <div
          style={{
            background: "#ffffff",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            padding: "20px 24px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          <div>
            <div style={{ fontSize: "11px", fontWeight: 700, color: "#0f766e", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "2px" }}>
              Evaluation Evidence
            </div>
            <h3 style={{ fontSize: "16.5px", fontWeight: 800, color: "#0f172a", margin: 0 }}>
              Empirical Strategy Comparison Matrix
            </h3>
            <p style={{ fontSize: "12.5px", color: "#64748b", margin: "2px 0 0 0" }}>
              Click any row to select a strategy. Review empirical performance deltas and pipeline complexity before recording your decision:
            </p>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12.5px" }}>
              <thead>
                <tr style={{ background: "#f8fafc", borderBottom: "1px solid #e2e8f0", textAlign: "left" }}>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Strategy Variant</th>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Primary Metric</th>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Δ from Baseline</th>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Features</th>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Complexity</th>
                  <th style={{ padding: "10px 12px", color: "#475569", fontWeight: 700 }}>Practical Significance</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => {
                  const isBaseline = r.is_baseline;
                  const isSelected = selectedStrategy === r.strategy_id;

                  const deltaColor =
                    r.delta_from_baseline > 0.005
                      ? "#047857"
                      : r.delta_from_baseline < -0.01
                      ? "#b91c1c"
                      : "#475569";

                  return (
                    <tr
                      key={r.strategy_id}
                      onClick={() => setSelectedStrategy(r.strategy_id)}
                      style={{
                        borderBottom: "1px solid #f1f5f9",
                        background: isSelected ? "#f0fdfa" : "transparent",
                        cursor: "pointer",
                        outline: isSelected ? "1px solid #0f766e" : "none",
                      }}
                    >
                      <td style={{ padding: "12px 12px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <input
                            type="radio"
                            id={`radio-${r.strategy_id}`}
                            name="strategy_selection"
                            checked={isSelected}
                            onChange={() => setSelectedStrategy(r.strategy_id)}
                            style={{ cursor: "pointer", accentColor: "#0f766e" }}
                          />
                          <label htmlFor={`radio-${r.strategy_id}`} style={{ cursor: "pointer" }}>
                            <div style={{ fontWeight: 700, color: "#0f172a" }}>
                              {r.strategy_name}
                              {isBaseline && (
                                <span
                                  style={{
                                    marginLeft: "8px",
                                    fontSize: "10px",
                                    fontWeight: 700,
                                    padding: "1px 6px",
                                    borderRadius: "4px",
                                    background: "#f1f5f9",
                                    color: "#334155",
                                    border: "1px solid #cbd5e1",
                                  }}
                                >
                                  Control Baseline
                                </span>
                              )}
                            </div>
                            <div style={{ fontSize: "11.5px", color: "#64748b", marginTop: "2px" }}>{r.description}</div>
                          </label>
                        </div>
                      </td>

                      <td style={{ padding: "12px 12px", fontWeight: 800, color: "#0f172a" }}>
                        {r.primary_score.toFixed(4)}{" "}
                        <span style={{ fontSize: "11px", fontWeight: 500, color: "#64748b" }}>
                          ({r.primary_metric_name})
                        </span>
                      </td>

                      <td style={{ padding: "12px 12px", fontWeight: 800, color: deltaColor }}>
                        {isBaseline
                          ? "—"
                          : `${r.delta_from_baseline > 0 ? "+" : ""}${r.delta_from_baseline.toFixed(4)}`}
                      </td>

                      <td style={{ padding: "12px 12px", color: "#334155" }}>
                        {r.feature_count}
                      </td>

                      <td style={{ padding: "12px 12px" }}>
                        <span
                          style={{
                            fontSize: "11px",
                            fontWeight: 700,
                            padding: "2px 7px",
                            borderRadius: "4px",
                            background: r.pipeline_complexity === "Low" ? "#ecfdf5" : "#fffbeb",
                            color: r.pipeline_complexity === "Low" ? "#047857" : "#b45309",
                            border: r.pipeline_complexity === "Low" ? "1px solid #a7f3d0" : "1px solid #fde68a",
                          }}
                        >
                          {r.pipeline_complexity}
                        </span>
                      </td>

                      <td style={{ padding: "12px 12px", fontSize: "12px", color: "#334155", fontWeight: 500 }}>
                        {r.practical_importance}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* System Evidence Interpretation */}
          {candidate.system_interpretation && (
            <div
              style={{
                background: "#f8fafc",
                borderRadius: "6px",
                border: "1px solid #e2e8f0",
                padding: "14px 16px",
              }}
            >
              <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#0f766e", textTransform: "uppercase", marginBottom: "3px" }}>
                System Evidence Interpretation
              </div>
              <p style={{ fontSize: "13px", color: "#1e293b", margin: 0, lineHeight: "1.5" }}>
                {candidate.system_interpretation}
              </p>
            </div>
          )}

          {/* Limitations & Caveats */}
          <div style={{ fontSize: "11.5px", color: "#64748b", lineHeight: "1.5", borderTop: "1px solid #f1f5f9", paddingTop: "10px" }}>
            <strong>Evaluation Limitations:</strong> Results reflect 5-fold cross-validation under the Random Forest inductive bias. Performance differences are model-family dependent; validate on out-of-time production distributions before final deployment.
          </div>
        </div>
      )}

      {/* Human Decision Confirmation Loop */}
      {hasResults && (
        <div
          style={{
            background: "#ffffff",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            padding: "20px 24px",
            display: "flex",
            flexDirection: "column",
            gap: "14px",
            boxShadow: "0 1px 3px rgba(0,0,0,0.02)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#0f766e", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "2px" }}>
                Human In The Loop
              </div>
              <h3 style={{ fontSize: "16px", fontWeight: 800, color: "#0f172a", margin: 0 }}>
                Record Human Decision & Rationale
              </h3>
            </div>
            {isDecided && (
              <span style={{ fontSize: "11.5px", fontWeight: 700, padding: "2px 8px", borderRadius: "4px", background: "#ecfdf5", color: "#047857", border: "1px solid #a7f3d0" }}>
                ✓ Decision Recorded
              </span>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
            <div>
              <label style={{ display: "block", fontSize: "12.5px", fontWeight: 600, color: "#334155", marginBottom: "4px" }}>
                Decision Rationale (Why this strategy is selected)
              </label>
              <textarea
                value={rationale}
                onChange={(e) => setRationale(e.target.value)}
                placeholder="e.g. Missing indicator preserves predictive signal with minimal added pipeline complexity in production."
                rows={3}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "12.5px",
                  color: "#0f172a",
                  boxSizing: "border-box",
                  fontFamily: "inherit",
                }}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12.5px", fontWeight: 600, color: "#334155", marginBottom: "4px" }}>
                Custom Team Method Notes (Optional)
              </label>
              <textarea
                value={customMethodNotes}
                onChange={(e) => setCustomMethodNotes(e.target.value)}
                placeholder="e.g. In downstream production, ETL applies iterative imputer fallback for edge cases."
                rows={3}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "12.5px",
                  color: "#0f172a",
                  boxSizing: "border-box",
                  fontFamily: "inherit",
                }}
              />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid #f1f5f9", paddingTop: "12px" }}>
            <div style={{ fontSize: "12.5px", color: "#475569" }}>
              Active Selection: <strong>{selectedStrategy.replace(/_/g, " ")}</strong>
            </div>

            <button
              onClick={handleSaveDecision}
              disabled={isSavingDecision || !selectedStrategy}
              style={{
                padding: "9px 18px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: 700,
                background: isSavingDecision ? "#94a3b8" : "#0f766e",
                color: "#ffffff",
                border: "none",
                cursor: isSavingDecision ? "not-allowed" : "pointer",
                boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
              }}
            >
              {isSavingDecision ? "Recording Decision..." : isDecided ? "Update Decision" : "Confirm Decision & Compile Artifacts"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
