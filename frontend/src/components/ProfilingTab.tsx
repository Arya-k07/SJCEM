import React, { useState } from "react";
import type { DatasetProfile, ColumnProfile } from "../types/report";

interface ProfilingTabProps {
  profile: DatasetProfile;
}

export const ProfilingTab: React.FC<ProfilingTabProps> = ({ profile }) => {
  const [selectedColumn, setSelectedColumn] = useState<string>(
    Object.keys(profile.columns_profile)[0] || ""
  );

  const activeCol: ColumnProfile | undefined = profile.columns_profile[selectedColumn];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: "20px" }}>
      {/* Column List Sidebar */}
      <div className="glass-panel" style={{ padding: "16px", maxHeight: "680px", overflowY: "auto" }}>
        <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "12px", padding: "0 4px" }}>
          Columns ({profile.columns})
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          {Object.entries(profile.columns_profile).map(([colName, colProf]) => {
            const isSelected = selectedColumn === colName;
            return (
              <div
                key={colName}
                onClick={() => setSelectedColumn(colName)}
                style={{
                  padding: "10px 12px",
                  borderRadius: "var(--radius-sm)",
                  background: isSelected ? "rgba(6, 182, 212, 0.15)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${isSelected ? "var(--accent-cyan)" : "transparent"}`,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <span style={{ fontWeight: 600, fontSize: "13.5px", color: isSelected ? "var(--accent-cyan)" : "var(--text-primary)" }}>
                    {colName}
                  </span>
                  <span style={{ fontSize: "10px", padding: "1px 5px", borderRadius: "3px", background: "rgba(255,255,255,0.06)", color: "var(--text-muted)" }}>
                    {colProf.dtype}
                  </span>
                </div>

                <div style={{ display: "flex", gap: "8px", fontSize: "11px", color: "var(--text-muted)" }}>
                  <span>{colProf.inferred_role.replace(/_/g, " ")}</span>
                  {colProf.missing_count > 0 && (
                    <span style={{ color: "var(--color-warning)" }}>
                      {(colProf.missing_pct * 100).toFixed(1)}% null
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Column Deep-Dive Panel */}
      {activeCol && (
        <div className="glass-panel" style={{ padding: "28px" }}>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <h3 style={{ fontSize: "22px", fontWeight: 800 }}>{activeCol.name}</h3>
                <span className="badge badge-info" style={{ textTransform: "uppercase" }}>
                  {activeCol.semantic_type}
                </span>
                <span className="badge badge-secondary" style={{ textTransform: "capitalize" }}>
                  {activeCol.inferred_role.replace(/_/g, " ")}
                </span>
              </div>
              <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "4px" }}>
                Type: <code>{activeCol.dtype}</code> • {activeCol.unique_count.toLocaleString()} unique values ({(activeCol.unique_ratio * 100).toFixed(1)}% cardinality)
              </p>
            </div>

            {/* Missingness Badge */}
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>Missing Rate</div>
              <div style={{ fontSize: "18px", fontWeight: 800, color: activeCol.missing_count > 0 ? "var(--color-warning)" : "var(--color-success)" }}>
                {(activeCol.missing_pct * 100).toFixed(2)}% ({activeCol.missing_count} rows)
              </div>
            </div>
          </div>

          {/* Numeric Summary Stats Grid */}
          {activeCol.min !== null && activeCol.min !== undefined && (
            <div style={{ marginBottom: "24px" }}>
              <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "8px" }}>
                Descriptive Numerical Statistics
              </span>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: "10px" }}>
                {[
                  { label: "Min", val: activeCol.min },
                  { label: "p25 (Q1)", val: activeCol.p25 },
                  { label: "Median", val: activeCol.median },
                  { label: "Mean", val: activeCol.mean },
                  { label: "p75 (Q3)", val: activeCol.p75 },
                  { label: "Max", val: activeCol.max },
                ].map((stat) => (
                  <div key={stat.label} style={{ background: "rgba(255,255,255,0.03)", padding: "10px", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>{stat.label}</div>
                    <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)", marginTop: "2px" }}>
                      {stat.val !== null && stat.val !== undefined ? stat.val : "—"}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", marginTop: "10px" }}>
                <div style={{ background: "rgba(255,255,255,0.03)", padding: "10px", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Standard Deviation (σ)</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)" }}>{activeCol.std || "—"}</div>
                </div>
                <div style={{ background: "rgba(255,255,255,0.03)", padding: "10px", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Skewness</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)" }}>{activeCol.skewness || "—"}</div>
                </div>
                <div style={{ background: "rgba(255,255,255,0.03)", padding: "10px", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Kurtosis</div>
                  <div style={{ fontSize: "15px", fontWeight: 700, fontFamily: "var(--font-mono)" }}>{activeCol.kurtosis || "—"}</div>
                </div>
              </div>
            </div>
          )}

          {/* Histogram Visualization */}
          {activeCol.histogram && activeCol.histogram.length > 0 && (
            <div style={{ marginBottom: "24px" }}>
              <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "8px" }}>
                Distribution Histogram (10 Bins)
              </span>

              <div style={{ height: "130px", display: "flex", alignItems: "flex-end", gap: "6px", background: "rgba(0,0,0,0.2)", padding: "12px", borderRadius: "var(--radius-sm)" }}>
                {activeCol.histogram.map((bin, idx) => {
                  const maxPct = Math.max(...activeCol.histogram.map((b) => b.pct), 0.01);
                  const heightPct = (bin.pct / maxPct) * 100;
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
                      title={`Range: [${bin.bin_start}, ${bin.bin_end}] | Count: ${bin.count} (${(bin.pct * 100).toFixed(1)}%)`}
                    >
                      <div
                        style={{
                          width: "100%",
                          height: `${Math.max(heightPct, 4)}%`,
                          background: "var(--accent-gradient)",
                          borderRadius: "3px 3px 0 0",
                          transition: "height 0.3s ease",
                        }}
                      />
                      <span style={{ fontSize: "9px", color: "var(--text-muted)", marginTop: "4px", whiteSpace: "nowrap" }}>
                        {bin.bin_start}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Top Frequent Values */}
          {activeCol.top_values && activeCol.top_values.length > 0 && (
            <div>
              <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "8px" }}>
                Most Frequent Values
              </span>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {activeCol.top_values.slice(0, 6).map((val, idx) => (
                  <div key={idx} style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "13px" }}>
                    <span style={{ width: "160px", color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      <code>{val.value || "(empty)"}</code>
                    </span>
                    <div style={{ flex: 1, height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${val.pct * 100}%`, height: "100%", background: "var(--accent-cyan)", borderRadius: "4px" }} />
                    </div>
                    <span style={{ width: "75px", textAlign: "right", fontSize: "12px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {(val.pct * 100).toFixed(1)}% ({val.count})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
