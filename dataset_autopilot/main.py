"""
CLI Entrypoint and Server launcher for Dataset Autopilot.
"""

import argparse
import json
import os
import sys
import uvicorn
from dataset_autopilot.api import app
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.generate_samples import generate_benchmark_datasets
from dataset_autopilot.ingestion import load_dataset_from_path
from dataset_autopilot.reporting import generate_html_report


def main():
    parser = argparse.ArgumentParser(
        prog="dataset-autopilot",
        description="Dataset Autopilot - Autonomous Data-Auditing Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST backend server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Run autonomous audit on a local file")
    audit_parser.add_argument("--file", "-f", required=True, help="Path to CSV / Parquet / JSON file")
    audit_parser.add_argument("--target", "-t", default=None, help="Target column name")
    audit_parser.add_argument("--split", "-s", default=None, help="Train/test split column name")
    audit_parser.add_argument("--time", default=None, help="Timestamp column name")
    audit_parser.add_argument("--output", "-o", default=None, help="Path to save output JSON report")
    audit_parser.add_argument("--html", default=None, help="Path to save output HTML report")
    audit_parser.add_argument("--gemini-key", default=None, help="Gemini API Key for AI summaries")

    # Generate samples command
    subparsers.add_parser("generate-samples", help="Generate sample benchmark datasets in data/samples")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"[Dataset Autopilot] Launching Server at http://{args.host}:{args.port}")
        uvicorn.run("dataset_autopilot.api:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "generate-samples":
        generate_benchmark_datasets()

    elif args.command == "audit":
        print(f"[*] Auditing dataset: {args.file}...")
        df, _ = load_dataset_from_path(args.file)
        engine = DatasetAutopilotEngine()

        def cli_progress(pct: int, msg: str):
            print(f"[{pct:3d}%] {msg}")

        report = engine.run_audit(
            df=df,
            dataset_name=os.path.basename(args.file),
            target_col=args.target,
            split_col=args.split,
            time_col=args.time,
            gemini_api_key=args.gemini_key,
            progress_callback=cli_progress
        )

        print("\n" + "=" * 55)
        print(f"   DATA HEALTH SCORE: {report.health_score}/100  (Grade: {report.health_grade})")
        print("=" * 55)
        print(f"Rows: {report.dataset_profile.rows:,} | Columns: {report.dataset_profile.columns} | Target: {report.dataset_profile.detected_target}")
        print(f"Quality: {report.score_breakdown.quality_score} | Leakage: {report.score_breakdown.leakage_score} | Drift: {report.score_breakdown.drift_score} | Fairness: {report.score_breakdown.fairness_score}")
        print(f"\nExecutive Summary:\n{report.executive_summary}\n")
        print(f"Total Findings: {len(report.findings)}")
        for f in report.findings[:5]:
            print(f"  * [{f.severity.value}] {f.title} ({f.type.value})")
        if len(report.findings) > 5:
            print(f"  ... and {len(report.findings) - 5} more findings.")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            print(f"\n[+] Saved JSON report to: {args.output}")

        if args.html:
            html_content = generate_html_report(report)
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"[+] Saved HTML report to: {args.html}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
