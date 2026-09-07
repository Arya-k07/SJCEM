"""HTTP and Python entry points for the preprocessing foundation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.preprocessing import (
	export_cleaned_dataset as write_cleaned_dataset,
	export_preprocessing_report,
	export_rejected_dataset,
	DatasetInputError,
	clean_dataset,
)
from app.nlp import analyze_dataset, export_analyzed_dataset

app = FastAPI(title="FeedbackIQ Preprocessing API", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
	return {
		"service": "feedbackiq-preprocessing",
		"status": "ok",
		"docs": "/docs",
		"health": "/health",
	}


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok", "service": "feedbackiq-preprocessing"}


@app.post("/api/preprocess")
async def preprocess(
	file: UploadFile = File(...),
	include_cleaned_data: bool = False,
	include_rejected_data: bool = False,
	export_cleaned_data: bool = False,
) -> dict:
	"""Clean a CSV/XLSX upload and optionally return its processed rows."""
	suffix = Path(file.filename or "").suffix.casefold()
	if suffix not in {".csv", ".xlsx", ".xlsm"}:
		raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLSM files are supported.")

	content = await file.read()
	temporary_file = NamedTemporaryFile(suffix=suffix, delete=False)
	temporary_path = Path(temporary_file.name)
	try:
		temporary_file.write(content)
		temporary_file.close()
		result = clean_dataset(temporary_path)
	except DatasetInputError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	finally:
		temporary_path.unlink(missing_ok=True)

	response = {
		"file_name": file.filename,
		"report": result["report"],
		"column_mapping": result["column_mapping"],
	}
	if include_cleaned_data:
		response["cleaned_data"] = json.loads(
			result["clean_data"].to_json(orient="records", date_format="iso")
		)
	if include_rejected_data:
		response["rejected_data"] = json.loads(
			result["rejected_data"].to_json(orient="records", date_format="iso")
		)
	if export_cleaned_data:
		output_directory = Path("outputs")
		file_token = uuid4().hex
		cleaned_path = write_cleaned_dataset(
			result["clean_data"], output_directory / f"cleaned_dataset_{file_token}.csv"
		)
		rejected_path = export_rejected_dataset(
			result["rejected_data"], output_directory / f"rejected_dataset_{file_token}.csv"
		)
		report_path = export_preprocessing_report(
			result["report"], output_directory / f"preprocessing_report_{file_token}.json"
		)
		response["export"] = {
			"cleaned_file": cleaned_path.as_posix(),
			"rejected_file": rejected_path.as_posix(),
			"report_file": report_path.as_posix(),
		}
	return response


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
	"""Preprocess an upload and return domain-agnostic NLP enrichment."""
	suffix = Path(file.filename or "").suffix.casefold()
	if suffix not in {".csv", ".xlsx", ".xlsm"}:
		raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLSM files are supported.")
	content = await file.read()
	temporary_file = NamedTemporaryFile(suffix=suffix, delete=False)
	temporary_path = Path(temporary_file.name)
	try:
		temporary_file.write(content)
		temporary_file.close()
		result = clean_dataset(temporary_path)
		analyzed = analyze_dataset(result["clean_data"])
		file_token = uuid4().hex
		analyzed_path = export_analyzed_dataset(analyzed, f"outputs/analyzed_dataset_{file_token}.csv")
	except (DatasetInputError, ValueError) as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	finally:
		temporary_path.unlink(missing_ok=True)
	return {
		"file_name": file.filename,
		"rows_analyzed": len(analyzed),
		"analyzed_file": analyzed_path,
		"data": json.loads(analyzed.to_json(orient="records", date_format="iso")),
	}


__all__ = ["app", "clean_dataset"]
