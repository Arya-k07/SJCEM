"""
FastAPI REST Server for Dataset Autopilot with asynchronous job queue,
progress tracking, sample dataset triggers, and report downloads.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
import uuid
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dataset_autopilot.config import (
    DEFAULT_CONFIG,
    AutopilotConfig,
    DecisionStatus,
)
from dataset_autopilot.engine import DatasetAutopilotEngine
from dataset_autopilot.ingestion import (
    load_dataset_from_base64,
    load_dataset_from_bytes,
    load_dataset_from_path,
    load_dataset_from_url,
)
from dataset_autopilot.reporting import generate_html_report
from dataset_autopilot.schemas import (
    AnalysisContext,
    AnalysisRequest,
    AuditReport,
    DecisionCandidate,
    ExperimentResult,
    ExperimentVariantResult,
    JobStatusResponse,
    UserDecision,
)


app = FastAPI(
    title="Dataset Autopilot API",
    description="Autonomous Data-Auditing and Leakage Detection Engine",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job, report, and dataset store
JOBS: Dict[str, Dict[str, Any]] = {}
REPORTS: Dict[str, AuditReport] = {}
DATASETS: Dict[str, Any] = {}
executor = ThreadPoolExecutor(max_workers=4)

SAMPLES = {
    "telecom_churn_leakage": {
        "id": "telecom_churn_leakage",
        "title": "Telecom Customer Churn",
        "description": "Customer churn dataset containing post-event refund leakage and informative missingness.",
        "target": "churn",
        "file_path": "data/samples/telecom_churn_leakage.csv",
        "badge": "Target Leakage Demo",
        "color": "#ef4444"
    },
    "credit_risk_drift": {
        "id": "credit_risk_drift",
        "title": "Credit Risk & Loan Default",
        "description": "Loan application records with macroeconomic time-based distribution shift (PSI > 0.28).",
        "target": "default_risk",
        "time_col": "application_date",
        "file_path": "data/samples/credit_risk_drift.csv",
        "badge": "Distribution Drift Demo",
        "color": "#f59e0b"
    },
    "titanic_quality_audit": {
        "id": "titanic_quality_audit",
        "title": "Titanic Passenger Survival",
        "description": "Classical survival dataset featuring structured missingness (MNAR) and extreme fare outliers.",
        "target": "survived",
        "file_path": "data/samples/titanic_quality_audit.csv",
        "badge": "Data Quality & Outliers",
        "color": "#3b82f6"
    },
    "recruitment_bias_slices": {
        "id": "recruitment_bias_slices",
        "title": "Recruitment & Hiring Pipeline",
        "description": "Job applicant screening data with candidate demographic slices and disparate impact.",
        "target": "hired",
        "file_path": "data/samples/recruitment_bias_slices.csv",
        "badge": "Slice Fairness & Bias",
        "color": "#8b5cf6"
    }
}


def _process_audit_job(
    job_id: str,
    df,
    dataset_name: str,
    target_col: Optional[str] = None,
    split_col: Optional[str] = None,
    time_col: Optional[str] = None,
    context: Optional[AnalysisContext] = None,
    gemini_api_key: Optional[str] = None
):
    """Worker function to execute audit in background."""
    engine = DatasetAutopilotEngine()

    def progress_callback(pct: int, stage: str):
        if job_id in JOBS:
            JOBS[job_id]["progress_pct"] = pct
            JOBS[job_id]["current_stage"] = stage

    try:
        JOBS[job_id]["status"] = "processing"
        DATASETS[job_id] = df
        report = engine.run_audit(
            df=df,
            dataset_name=dataset_name,
            target_col=target_col,
            split_col=split_col,
            time_col=time_col,
            context=context,
            gemini_api_key=gemini_api_key,
            progress_callback=progress_callback
        )
        REPORTS[job_id] = report
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress_pct"] = 100
        JOBS[job_id]["current_stage"] = "Audit completed successfully."
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error_message"] = str(e)
        JOBS[job_id]["current_stage"] = f"Failed: {str(e)}"


@app.get("/")
def root():
    return {
        "service": "Dataset Autopilot API",
        "status": "online",
        "version": "2.0.0 (Decision Lab)",
        "docs": "/docs"
    }


@app.get("/api/samples")
def list_samples():
    """Return available pre-loaded benchmark datasets."""
    return list(SAMPLES.values())


@app.post("/api/samples/{sample_id}/analyze")
def analyze_sample(sample_id: str, background_tasks: BackgroundTasks, gemini_api_key: Optional[str] = None):
    """Trigger 1-click audit on a sample benchmark dataset."""
    sample_info = SAMPLES.get(sample_id)
    if not sample_info:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")

    file_path = sample_info["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Sample file missing on server.")

    df, _ = load_dataset_from_path(file_path)
    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0,
        "current_stage": "Queued for analysis...",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "error_message": None
    }

    background_tasks.add_task(
        _process_audit_job,
        job_id=job_id,
        df=df,
        dataset_name=f"{sample_info['title']}.csv",
        target_col=sample_info.get("target"),
        time_col=sample_info.get("time_col"),
        gemini_api_key=gemini_api_key
    )

    return {"job_id": job_id, "status": "queued", "message": f"Analyzing sample '{sample_info['title']}'"}


@app.post("/api/analyze")
async def analyze_dataset(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None),
    split_column: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    dataset_url: Optional[str] = Form(None),
    dataset_base64: Optional[str] = Form(None),
    file_name: Optional[str] = Form("uploaded_dataset.csv"),
    primary_objective: Optional[str] = Form("predictive_performance")
):
    """Upload and start auditing a dataset via multipart form or parameters."""
    try:
        if file:
            content = await file.read()
            df, _ = load_dataset_from_bytes(content, filename=file.filename)
            dname = file.filename
        elif dataset_url:
            df, _ = load_dataset_from_url(dataset_url)
            dname = dataset_url.split("/")[-1] or "downloaded_data.csv"
        elif dataset_base64:
            df, _ = load_dataset_from_base64(dataset_base64, filename=file_name or "data.csv")
            dname = file_name or "data.csv"
        else:
            raise HTTPException(status_code=400, detail="Must provide a file, dataset_url, or dataset_base64.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {str(e)}")

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0,
        "current_stage": "Queued for analysis...",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "error_message": None
    }

    context = AnalysisContext(
        target_column=target_column,
        time_column=time_column,
        primary_objective=primary_objective or "predictive_performance"
    )

    background_tasks.add_task(
        _process_audit_job,
        job_id=job_id,
        df=df,
        dataset_name=dname,
        target_col=target_column,
        split_col=split_column,
        time_col=time_column,
        context=context,
        gemini_api_key=gemini_api_key
    )

    return {"job_id": job_id, "status": "queued", "message": "Dataset analysis queued."}


@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll status and progress of an ongoing audit job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return JobStatusResponse(**job)


@app.get("/api/results/{job_id}", response_model=AuditReport)
def get_job_results(job_id: str):
    """Retrieve full audit report JSON for a completed job."""
    report = REPORTS.get(job_id)
    if not report:
        job = JOBS.get(job_id)
        if job and job["status"] == "processing":
            raise HTTPException(status_code=202, detail="Audit is still in progress.")
        elif job and job["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"Audit failed: {job.get('error_message')}")
        raise HTTPException(status_code=404, detail="Report not found for this Job ID.")
    return report


# =========================================================================
# Decision Lab REST Endpoints
# =========================================================================

@app.get("/api/runs/{run_id}/decision-candidates", response_model=List[DecisionCandidate])
def get_decision_candidates(run_id: str):
    """Retrieve prioritized Decision Candidates for a completed run."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    return report.decision_candidates


@app.get("/api/runs/{run_id}/decisions/{decision_id}", response_model=DecisionCandidate)
def get_decision_details(run_id: str, decision_id: str):
    """Retrieve details for a specific decision candidate."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    cand = next((c for c in report.decision_candidates if c.decision_id == decision_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Decision candidate not found.")
    return cand


class ExperimentExecutionRequest(BaseModel):
    strategy_ids: Optional[List[str]] = None


@app.post("/api/runs/{run_id}/decisions/{decision_id}/experiments", response_model=DecisionCandidate)
def run_decision_experiments(run_id: str, decision_id: str, request: Optional[ExperimentExecutionRequest] = None):
    """Run controlled A/B evaluation across registered strategies for a decision candidate."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    df = DATASETS.get(run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset for this run not found in memory.")

    cand = next((c for c in report.decision_candidates if c.decision_id == decision_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Decision candidate not found.")

    target_col = report.dataset_profile.detected_target
    if not target_col:
        raise HTTPException(status_code=400, detail="Cannot run supervised experiments on unsupervised dataset.")

    time_col = report.analysis_context.time_column if report.analysis_context else None
    strategy_ids = request.strategy_ids if request and request.strategy_ids else None
    from dataset_autopilot.experiment_registry import execute_candidate_investigation

    updated_cand = execute_candidate_investigation(
        candidate=cand,
        df=df,
        profile=report.dataset_profile,
        target_col=target_col,
        time_col=time_col,
        selected_strategies=strategy_ids
    )

    return updated_cand


@app.get("/api/runs/{run_id}/decisions/{decision_id}/results", response_model=List[ExperimentVariantResult])
def get_decision_results(run_id: str, decision_id: str):
    """Retrieve multi-metric comparison results for a decision candidate."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    cand = next((c for c in report.decision_candidates if c.decision_id == decision_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Decision candidate not found.")
    if not cand.experiment_results:
        raise HTTPException(status_code=404, detail="Experiments have not yet been run for this decision.")
    return cand.experiment_results


class UserDecisionPayload(BaseModel):
    selected_strategy_id: str
    decision_rationale: Optional[str] = None
    custom_method_notes: Optional[str] = None


@app.post("/api/runs/{run_id}/decisions/{decision_id}/decision", response_model=DecisionCandidate)
def record_human_decision(run_id: str, decision_id: str, payload: UserDecisionPayload):
    """Record human decision and rationale on a Decision Candidate."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    cand = next((c for c in report.decision_candidates if c.decision_id == decision_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Decision candidate not found.")

    decision_obj = UserDecision(
        selected_strategy_id=payload.selected_strategy_id,
        decision_rationale=payload.decision_rationale,
        decided_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        custom_method_notes=payload.custom_method_notes
    )

    cand.user_decision = decision_obj
    cand.status = DecisionStatus.USER_DECISION

    return cand


@app.get("/api/runs/{run_id}/decisions/{decision_id}/artifacts/{artifact_type}")
def get_decision_artifact(run_id: str, decision_id: str, artifact_type: str):
    """Download reproducible artifact: notebook (.ipynb), pipeline (.py), html, or json."""
    report = REPORTS.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit run not found.")
    cand = next((c for c in report.decision_candidates if c.decision_id == decision_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Decision candidate not found.")

    from dataset_autopilot.artifacts import (
        generate_decision_html,
        generate_jupyter_notebook,
        generate_pipeline_script,
    )

    safe_dec_id = "".join(c for c in cand.decision_id if c.isalnum() or c in "-_") or "decision"

    if artifact_type == "notebook":
        content = generate_jupyter_notebook(cand, report)
        return Response(
            content=content,
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{safe_dec_id}_workflow.ipynb"'}
        )
    elif artifact_type == "pipeline":
        content = generate_pipeline_script(cand, report)
        return Response(
            content=content,
            media_type="text/x-python",
            headers={"Content-Disposition": f'attachment; filename="{safe_dec_id}_pipeline.py"'}
        )
    elif artifact_type == "html":
        content = generate_decision_html(cand, report)
        return HTMLResponse(
            content=content,
            headers={"Content-Disposition": f'attachment; filename="{safe_dec_id}_report.html"'}
        )
    elif artifact_type == "json":
        content = cand.model_dump_json(indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_dec_id}_results.json"'}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid artifact_type. Supported: notebook, pipeline, html, json.")


@app.get("/api/export/{job_id}/html", response_class=HTMLResponse)
def export_html_report(job_id: str):
    """Download standalone interactive HTML audit report."""
    report = REPORTS.get(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    html_content = generate_html_report(report)
    return HTMLResponse(
        content=html_content,
        headers={"Content-Disposition": f"attachment; filename=autopilot_report_{job_id}.html"}
    )
