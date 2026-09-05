import React, { useState } from "react";
import { Search, Eye, AlertCircle, CheckCircle2, ArrowRight } from "lucide-react";
import type { Finding, FindingStatus } from "../types/report";

interface FindingsTableProps {
  findings: Finding[];
  onSelectFinding: (finding: Finding) => void;
}

export const FindingsTable: React.FC<FindingsTableProps> = ({ findings, onSelectFinding }) => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");

  const categories = Array.from(new Set(findings.map((f) => f.type)));

  const filtered = findings.filter((f) => {
    const matchesSearch =
      f.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (f.column && f.column.toLowerCase().includes(searchTerm.toLowerCase())) ||
      f.id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = statusFilter === "ALL" || (f.status || "OBSERVED") === statusFilter;
    const matchesCategory = categoryFilter === "ALL" || f.type === categoryFilter;

    return matchesSearch && matchesStatus && matchesCategory;
  });

  const getStatusBadge = (status: FindingStatus) => {
    if (status === "SUSPECTED") {
      return (
        <span className="badge badge-warning" style={{ gap: "4px" }}>
          <AlertCircle size={12} /> SUSPECTED
        </span>
      );
    }
    if (status === "CONFIRMED") {
      return (
        <span className="badge badge-critical" style={{ gap: "4px" }}>
          <CheckCircle2 size={12} /> CONFIRMED
        </span>
      );
    }
    return (
      <span className="badge badge-info" style={{ gap: "4px" }}>
        <Eye size={12} /> OBSERVED
      </span>
    );
  };

  return (
    <div className="glass-panel" style={{ padding: "24px", marginBottom: "28px" }}>
      {/* Controls Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "16px", marginBottom: "20px", flexWrap: "wrap" }}>
        {/* Search */}
        <div style={{ position: "relative", minWidth: "260px", flex: 1 }}>
          <input
            type="text"
            placeholder="Search observations, columns, IDs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 14px 8px 34px",
              borderRadius: "var(--radius-sm)",
              background: "var(--bg-primary)",
              border: "1px solid var(--border-color)",
              color: "var(--text-primary)",
              outline: "none",
            }}
          />
          <Search size={15} color="var(--text-muted)" style={{ position: "absolute", left: "10px", top: "11px" }} />
        </div>

        {/* Status Filters */}
        <div style={{ display: "flex", gap: "6px" }}>
          {["ALL", "OBSERVED", "SUSPECTED", "CONFIRMED"].map((st) => {
            const isActive = statusFilter === st;
            const count = st === "ALL" ? findings.length : findings.filter((f) => (f.status || "OBSERVED") === st).length;
            return (
              <button
                key={st}
                onClick={() => setStatusFilter(st)}
                className={`btn btn-sm ${isActive ? "btn-primary" : "btn-secondary"}`}
                style={{ fontSize: "12px", padding: "6px 10px" }}
              >
                {st} ({count})
              </button>
            );
          })}
        </div>

        {/* Category Dropdown */}
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "var(--radius-sm)",
            background: "var(--bg-primary)",
            border: "1px solid var(--border-color)",
            color: "var(--text-primary)",
            outline: "none",
            fontSize: "12.5px",
          }}
        >
          <option value="ALL">All Categories ({findings.length})</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c.replace(/_/g, " ").toUpperCase()} ({findings.filter((f) => f.type === c).length})
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13.5px" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left" }}>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase", width: "90px" }}>ID</th>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase", width: "120px" }}>Status</th>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase", width: "140px" }}>Category</th>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase", width: "140px" }}>Column</th>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase" }}>Observation & Decision Question</th>
              <th style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: "11.5px", textTransform: "uppercase", width: "110px", textAlign: "right" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", padding: "40px", color: "var(--text-muted)" }}>
                  No observations matching current filters.
                </td>
              </tr>
            ) : (
              filtered.map((f) => (
                <tr
                  key={f.id}
                  style={{
                    borderBottom: "1px solid var(--border-color)",
                    cursor: "pointer",
                    transition: "background 0.15s ease",
                  }}
                  onClick={() => onSelectFinding(f)}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-subtle)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <td style={{ padding: "14px 12px", fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--text-muted)" }}>
                    {f.id}
                  </td>
                  <td style={{ padding: "14px 12px" }}>
                    {getStatusBadge(f.status || "OBSERVED")}
                  </td>
                  <td style={{ padding: "14px 12px", fontSize: "12.5px", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                    {f.type.replace(/_/g, " ")}
                  </td>
                  <td style={{ padding: "14px 12px" }}>
                    {f.column ? <code>{f.column}</code> : <span style={{ color: "var(--text-muted)", fontSize: "12px" }}>Dataset</span>}
                  </td>
                  <td style={{ padding: "14px 12px" }}>
                    <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "3px" }}>{f.title}</div>
                    <div style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                      {f.decision_question || f.description}
                    </div>
                  </td>
                  <td style={{ padding: "14px 12px", textAlign: "right" }}>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectFinding(f);
                      }}
                      style={{ fontSize: "12px", padding: "4px 8px" }}
                    >
                      Evidence <ArrowRight size={12} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
