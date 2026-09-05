import React, { useState } from "react";
import type { AuditReport } from "../types/report";
import { TrendingUp } from "lucide-react";

interface DriftTabProps {
  report: AuditReport;
}

export const DriftTab: React.FC<DriftTabProps> = ({ report }) => {
  const driftList = report.drift_summary || [];
  const [selectedFeature, setSelectedFeature] = useState<string>(
    driftList[0]?.feature || ""
  );

  const activeDrift = driftList.find((d) => d.feature === selectedFeature);
  const driftedCount = driftList.filter((d) => d.is_drift_detected).length;

  return (
    <div>
      {/* Top Banner */}
      <div
        className="glass-panel"
        style={{
          padding: "20px 24px",
          marginBottom: "24px",
          borderLeft: `4px solid ${driftedCount > 0 ? "var(--color-warning)" : "var(--color-success)"}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <TrendingUp size={20} color={driftedCount > 0 ? "var(--color-warning)" : "var(--color-success)"} />
            <h3 style={{ fontSize: "17px", fontWeight: 700 }}>
              {driftedCount > 0
                ? `${driftedCount} Feature(s) Showing Significant Distribution Drift`
                : "Distribution Stability Confirmed Across Features"}
            </h3>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Population Stability Index (PSI &gt; 0.25 critical, &gt; 0.10 warning) and Kolmogorov-Smirnov 2-sample tests.
          </p>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Drift Score</div>
          <div style={{ fontSize: "24px", fontWeight: 800, color: report.score_breakdown.drift_score >= 80 ? "var(--color-success)" : "var(--color-warning)" }}>
            {report.score_breakdown.drift_score}/100
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: "20px" }}>
        {/* Feature PSI Leaderboard */}
        <div className="glass-panel" style={{ padding: "16px", maxHeight: "640px", overflowY: "auto" }}>
          <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "12px", padding: "0 4px" }}>
            Audited Features ({driftList.length})
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {driftList.map((d) => {
              const isSelected = selectedFeature === d.feature;
              const psiColor = d.psi >= 0.25 ? "var(--color-critical)" : (d.psi >= 0.10 ? "var(--color-warning)" : "var(--color-success)");
              return (
                <div
                  key={d.feature}
                  onClick={() => setSelectedFeature(d.feature)}
                  style={{
                    padding: "10px 12px",
                    borderRadius: "var(--radius-sm)",
                    background: isSelected ? "rgba(6, 182, 212, 0.15)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${isSelected ? "var(--accent-cyan)" : "transparent"}`,
                    cursor: "pointer",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: "13px", color: isSelected ? "var(--accent-cyan)" : "var(--text-primary)" }}>
                      {d.feature}
                    </div>
                    {d.ks_p_value !== null && d.ks_p_value !== undefined && (
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        KS p-val: {d.ks_p_value < 0.001 ? "< 0.001" : d.ks_p_value.toFixed(3)}
                      </div>
                    )}
                  </div>

                  <div style={{ textAlign: "right" }}>
                    <span style={{ fontSize: "12px", fontWeight: 800, fontFamily: "var(--font-mono)", color: psiColor }}>
                      PSI {d.psi.toFixed(3)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Deep Dive Bin-by-Bin Distribution Overlay */}
        {activeDrift ? (
          <div className="glass-panel" style={{ padding: "28px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <h3 style={{ fontSize: "20px", fontWeight: 800 }}>{activeDrift.feature}</h3>
                  <span className={`badge badge-${activeDrift.severity.toLowerCase()}`}>
                    {activeDrift.severity}
                  </span>
                </div>
                <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "4px" }}>
                  PSI: <strong>{activeDrift.psi.toFixed(4)}</strong> • {activeDrift.psi >= 0.25 ? "Significant population shift" : (activeDrift.psi >= 0.10 ? "Moderate shift" : "Stable baseline")}
                </p>
              </div>

              {activeDrift.ks_statistic !== null && activeDrift.ks_statistic !== undefined && (
                <div style={{ textAlign: "right", background: "rgba(255,255,255,0.03)", padding: "8px 14px", borderRadius: "var(--radius-sm)" }}>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>KS 2-Sample Statistic</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)" }}>
                    {activeDrift.ks_statistic.toFixed(3)} (p={activeDrift.ks_p_value})
                  </div>
                </div>
              )}
            </div>

            {/* Side-by-Side Bins Histogram */}
            {activeDrift.bin_details && activeDrift.bin_details.length > 0 && (
              <div>
                <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "12px" }}>
                  Baseline (Expected) vs Current (Actual) Bin Discretization
                </span>

                <div style={{ display: "flex", gap: "16px", marginBottom: "16px", fontSize: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <div style={{ width: "12px", height: "12px", background: "var(--accent-blue)", borderRadius: "2px" }} />
                    <span style={{ color: "var(--text-secondary)" }}>Baseline Expected %</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <div style={{ width: "12px", height: "12px", background: "var(--accent-cyan)", borderRadius: "2px" }} />
                    <span style={{ color: "var(--text-secondary)" }}>Current Actual %</span>
                  </div>
                </div>

                <div style={{ height: "160px", display: "flex", alignItems: "flex-end", gap: "10px", background: "rgba(0,0,0,0.2)", padding: "16px", borderRadius: "var(--radius-sm)" }}>
                  {activeDrift.bin_details.map((b, idx) => {
                    const maxPct = 0.5; // normalized scale
                    const expHeight = (b.expected_pct / maxPct) * 100;
                    const actHeight = (b.actual_pct / maxPct) * 100;

                    return (
                      <div
                        key={idx}
                        style={{
                          flex: 1,
                          height: "100%",
                          display: "flex",
                          flexDirection: "column",
                          justifyContent: "flex-end",
                          alignItems: "center",
                        }}
                        title={`Bin: ${b.bin_label} | Expected: ${(b.expected_pct * 100).toFixed(1)}% | Actual: ${(b.actual_pct * 100).toFixed(1)}% | PSI Contribution: ${b.psi_contribution.toFixed(4)}`}
                      >
                        <div style={{ display: "flex", alignItems: "flex-end", gap: "3px", width: "100%", height: "100%" }}>
                          {/* Expected Bar */}
                          <div
                            style={{
                              flex: 1,
                              height: `${Math.min(expHeight, 100)}%`,
                              background: "var(--accent-blue)",
                              borderRadius: "2px 2px 0 0",
                              opacity: 0.8,
                            }}
                          />
                          {/* Actual Bar */}
                          <div
                            style={{
                              flex: 1,
                              height: `${Math.min(actHeight, 100)}%`,
                              background: "var(--accent-cyan)",
                              borderRadius: "2px 2px 0 0",
                            }}
                          />
                        </div>
                        <span style={{ fontSize: "9px", color: "var(--text-muted)", marginTop: "6px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "45px" }}>
                          {b.bin_label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="glass-panel" style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
            Select a feature from the sidebar to inspect distribution drift.
          </div>
        )}
      </div>
    </div>
  );
};
