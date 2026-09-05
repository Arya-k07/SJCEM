"""
Reporting Engine: computes composite Health Scores, formats findings, generates
human-readable summaries (via Gemini API with deterministic fallback), and builds standalone HTML reports.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import requests
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    FindingCategory,
    Severity,
)
from dataset_autopilot.schemas import (
    AuditReport,
    DatasetProfile,
    DriftMetric,
    ExperimentResult,
    Finding,
    HealthScoreBreakdown,
    SliceMetric,
)


def compute_health_scores(
    findings: List[Finding],
    has_target: bool = True,
    config: AutopilotConfig = DEFAULT_CONFIG
) -> HealthScoreBreakdown:
    """Compute overall and category-specific data health scores (0-100) and letter grade."""
    category_penalties = {
        "quality": 0.0,
        "leakage": 0.0,
        "drift": 0.0,
        "fairness": 0.0,
    }

    penalty_weights = {
        Severity.CRITICAL: 15.0,
        Severity.WARNING: 5.0,
        Severity.INFO: 1.0,
    }

    for f in findings:
        pts = penalty_weights.get(f.severity, 3.0)
        if f.type in [FindingCategory.TARGET_LEAKAGE, FindingCategory.TRAIN_TEST_CONTAMINATION]:
            category_penalties["leakage"] += pts
        elif f.type == FindingCategory.DISTRIBUTION_DRIFT:
            category_penalties["drift"] += pts
        elif f.type == FindingCategory.SLICE_BIAS:
            category_penalties["fairness"] += pts
        else:
            category_penalties["quality"] += pts

    quality_score = max(0.0, 100.0 - category_penalties["quality"])
    drift_score = max(0.0, 100.0 - category_penalties["drift"])

    if has_target:
        leakage_score = max(0.0, 100.0 - category_penalties["leakage"])
        fairness_score = max(0.0, 100.0 - category_penalties["fairness"])
        overall = (
            config.weight_quality * quality_score +
            config.weight_leakage * leakage_score +
            config.weight_drift * drift_score +
            config.weight_fairness * fairness_score
        )
    else:
        # For unsupervised datasets, leakage and supervised fairness were not assessed
        leakage_score = 0.0
        fairness_score = 0.0
        applicable_weights = config.weight_quality + config.weight_drift
        overall = (
            config.weight_quality * quality_score +
            config.weight_drift * drift_score
        ) / applicable_weights

    overall = round(max(0.0, min(100.0, overall)), 1)

    if overall >= 90:
        grade = "A+"
    elif overall >= 80:
        grade = "A"
    elif overall >= 70:
        grade = "B"
    elif overall >= 55:
        grade = "C"
    elif overall >= 40:
        grade = "D"
    else:
        grade = "F"

    return HealthScoreBreakdown(
        overall_score=overall,
        quality_score=round(quality_score, 1),
        leakage_score=round(leakage_score, 1),
        drift_score=round(drift_score, 1),
        fairness_score=round(fairness_score, 1),
        grade=grade
    )


def generate_executive_summary_deterministic(
    profile: DatasetProfile,
    findings: List[Finding],
    score_breakdown: HealthScoreBreakdown,
    experiments: List[ExperimentResult]
) -> str:
    """Generate a comprehensive, deterministic plain-English executive summary."""
    crit_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    warn_count = sum(1 for f in findings if f.severity == Severity.WARNING)

    lines = []
    lines.append(
        f"Dataset Autopilot audited {profile.rows:,} rows and {profile.columns} columns "
        f"with overall Data Health Score of {score_breakdown.overall_score}/100 (Grade: {score_breakdown.grade})."
    )

    if crit_count > 0:
        lines.append(f"CRITICAL ATTENTION: {crit_count} high-severity issue(s) identified that pose immediate risk to model integrity.")
    elif warn_count > 0:
        lines.append(f"Notice: {warn_count} warning(s) detected. Recommended remediations should be addressed before deployment.")
    else:
        lines.append("Pristine status: No severe data quality or target leakage risks detected.")

    leak_findings = [f for f in findings if f.type == FindingCategory.TARGET_LEAKAGE]
    if leak_findings:
        leaky_cols = [f.column for f in leak_findings if f.column]
        lines.append(f"Target Leakage: Features {leaky_cols} show suspicious predictive dominance or post-event properties.")

    drift_findings = [f for f in findings if f.type == FindingCategory.DISTRIBUTION_DRIFT]
    if drift_findings:
        lines.append(f"Distribution Drift: Significant shift (PSI >= 0.25) observed across {len(drift_findings)} feature(s).")

    bias_findings = [f for f in findings if f.type == FindingCategory.SLICE_BIAS]
    if bias_findings:
        lines.append(f"Fairness Slices: {len(bias_findings)} demographic or categorical subgroup(s) exhibit performance disparity >15%.")

    if experiments:
        leak_exp = next((e for e in experiments if e.category == "target_leakage"), None)
        if leak_exp:
            lines.append(
                f"Remediation Impact: Removing suspect leaky features caused baseline {leak_exp.baseline_metric_name} "
                f"to shift from {leak_exp.baseline_score:.3f} to {leak_exp.modified_score:.3f} ({leak_exp.notes})."
            )

    return " ".join(lines)


def generate_executive_summary_gemini(
    profile: DatasetProfile,
    findings: List[Finding],
    score_breakdown: HealthScoreBreakdown,
    experiments: List[ExperimentResult],
    api_key: str,
    model_name: str = "gemini-2.5-flash"
) -> Optional[str]:
    """Generate natural language summary using Google Gemini API."""
    if not api_key:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        prompt = (
            f"You are Dataset Autopilot, an expert ML data auditing engine. "
            f"Summarize the following tabular audit in 3-4 professional, concise sentences for a data scientist.\n"
            f"Dataset: {profile.rows} rows, {profile.columns} columns, detected target: '{profile.detected_target}'.\n"
            f"Health Score: {score_breakdown.overall_score}/100 (Grade: {score_breakdown.grade}).\n"
            f"Category Scores: Quality={score_breakdown.quality_score}, Leakage={score_breakdown.leakage_score}, Drift={score_breakdown.drift_score}, Fairness={score_breakdown.fairness_score}.\n"
            f"Top Findings: {[{'id': f.id, 'type': f.type.value, 'col': f.column, 'title': f.title, 'sev': f.severity.value} for f in findings[:6]]}.\n"
            f"Experiments: {[{'name': e.name, 'delta': e.delta, 'status': e.status} for e in experiments[:3]]}."
        )

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 300
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
    except Exception:
        pass

    return None


def generate_html_report(report: AuditReport) -> str:
    """Build a standalone, beautifully styled self-contained HTML audit report."""
    grade_color = "#10b981" if report.health_grade.startswith("A") else (
        "#3b82f6" if report.health_grade == "B" else (
            "#f59e0b" if report.health_grade == "C" else "#ef4444"
        )
    )

    findings_rows = ""
    for f in report.findings:
        badge_class = "badge-crit" if f.severity == Severity.CRITICAL else (
            "badge-warn" if f.severity == Severity.WARNING else "badge-info"
        )
        findings_rows += f"""
        <tr>
            <td><code>{f.id}</code></td>
            <td><span class="badge {badge_class}">{f.severity.value}</span></td>
            <td><strong>{f.type.value.replace('_', ' ').title()}</strong></td>
            <td><code>{f.column or 'Dataset-level'}</code></td>
            <td>
                <div style="font-weight:600; margin-bottom:4px;">{f.title}</div>
                <div style="font-size:13px; color:#475569;">{f.description}</div>
                <div style="margin-top:6px; font-size:12px; color:#0f766e; background:#f0fdfa; padding:4px 8px; border-radius:4px; border-left:3px solid #0d9488;">
                    <strong>Recommendation:</strong> {f.recommendation}
                </div>
            </td>
        </tr>
        """

    exp_rows = ""
    for e in report.experiments:
        delta_color = "#10b981" if e.delta > 0 else ("#ef4444" if e.delta < 0 else "#64748b")
        exp_rows += f"""
        <tr>
            <td><code>{e.id}</code></td>
            <td><strong>{e.name}</strong><br><small style="color:#64748b;">{e.description}</small></td>
            <td>{e.model_name}</td>
            <td>{e.baseline_metric_name}</td>
            <td>{e.baseline_score:.3f}</td>
            <td>{e.modified_score:.3f}</td>
            <td style="color:{delta_color}; font-weight:bold;">{e.delta:+.3f} ({e.delta_pct:+.1f}%)</td>
            <td><em>{e.notes}</em></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Autopilot Report - {report.dataset_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --accent: #38bdf8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 32px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #ffffff; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05); padding: 36px; border: 1px solid #e2e8f0; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 24px; margin-bottom: 28px; }}
        .brand {{ font-size: 26px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px; }}
        .brand span {{ color: #0284c7; }}
        .meta {{ color: #64748b; font-size: 14px; text-align: right; }}
        .scorecard {{ display: grid; grid-template-columns: 220px 1fr; gap: 24px; background: linear-gradient(135deg, #0f172a, #1e293b); color: #fff; padding: 28px; border-radius: 14px; margin-bottom: 32px; }}
        .score-circle {{ display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.05); border: 3px solid {grade_color}; border-radius: 12px; padding: 20px; }}
        .score-num {{ font-size: 48px; font-weight: 900; color: {grade_color}; line-height: 1; }}
        .grade-badge {{ font-size: 18px; font-weight: 700; background: {grade_color}; color: #fff; padding: 2px 10px; border-radius: 6px; margin-top: 8px; }}
        .score-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .score-item {{ background: rgba(255,255,255,0.06); padding: 12px; border-radius: 8px; text-align: center; }}
        .score-item-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
        .score-item-val {{ font-size: 22px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
        .summary-box {{ background: #f8fafc; border-left: 4px solid #0284c7; padding: 16px 20px; border-radius: 6px; margin-bottom: 32px; font-size: 15px; color: #334155; }}
        h2 {{ font-size: 20px; color: #0f172a; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 14px; }}
        th {{ background: #f1f5f9; text-align: left; padding: 12px 14px; font-weight: 700; color: #475569; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
        tr:hover td {{ background: #fafafa; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .badge-crit {{ background: #fee2e2; color: #dc2626; }}
        .badge-warn {{ background: #fef3c7; color: #d97706; }}
        .badge-info {{ background: #e0f2fe; color: #0284c7; }}
        code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; color: #0f172a; }}
        .footer {{ text-align: center; color: #94a3b8; font-size: 13px; border-top: 1px solid #f1f5f9; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand">🚀 Dataset <span>Autopilot</span></div>
            <div class="meta">
                <div><strong>Report ID:</strong> {report.report_id}</div>
                <div><strong>Dataset:</strong> {report.dataset_name} | {report.dataset_profile.rows:,} rows, {report.dataset_profile.columns} cols</div>
                <div><strong>Generated:</strong> {report.created_at}</div>
            </div>
        </div>

        <div class="scorecard">
            <div class="score-circle">
                <div class="score-num">{report.health_score}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">DATA HEALTH SCORE</div>
                <div class="grade-badge">GRADE {report.health_grade}</div>
            </div>
            <div>
                <h3 style="font-size:18px; margin-bottom:8px;">Executive Health Assessment</h3>
                <p style="font-size:14px; color:#cbd5e1; line-height:1.5;">{report.executive_summary}</p>
                <div class="score-grid">
                    <div class="score-item">
                        <div class="score-item-title">Data Quality</div>
                        <div class="score-item-val">{report.score_breakdown.quality_score}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-title">Target Leakage</div>
                        <div class="score-item-val">{report.score_breakdown.leakage_score}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-title">Distribution Drift</div>
                        <div class="score-item-val">{report.score_breakdown.drift_score}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-title">Slice Fairness</div>
                        <div class="score-item-val">{report.score_breakdown.fairness_score}</div>
                    </div>
                </div>
            </div>
        </div>

        <h2>🔍 Audit Findings & Actionable Recommendations ({len(report.findings)})</h2>
        <table>
            <thead>
                <tr>
                    <th style="width:70px;">ID</th>
                    <th style="width:90px;">Severity</th>
                    <th style="width:140px;">Category</th>
                    <th style="width:130px;">Column</th>
                    <th>Details & Remediation</th>
                </tr>
            </thead>
            <tbody>
                {findings_rows if findings_rows else '<tr><td colspan="5" style="text-align:center; padding:20px; color:#64748b;">No issues detected. Dataset passed all validation checks!</td></tr>'}
            </tbody>
        </table>

        <h2>🧪 Controlled Remediation Experiments ({len(report.experiments)})</h2>
        <table>
            <thead>
                <tr>
                    <th style="width:80px;">ID</th>
                    <th>Experiment Name</th>
                    <th>Oracle Model</th>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Modified</th>
                    <th>Impact (Δ)</th>
                    <th>Empirical Outcome</th>
                </tr>
            </thead>
            <tbody>
                {exp_rows if exp_rows else '<tr><td colspan="8" style="text-align:center; padding:20px; color:#64748b;">No experiments configured.</td></tr>'}
            </tbody>
        </table>

        <div class="footer">
            Generated autonomously by Dataset Autopilot Engine • Scikit-learn Oracle Benchmarking
        </div>
    </div>
</body>
</html>
"""
    return html
