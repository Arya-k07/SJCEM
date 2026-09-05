"""
Data ingestion module supporting CSV, TSV, Parquet, JSON, base64 data, and URLs.
"""

import base64
import io
import os
from typing import Optional, Tuple, Union
import pandas as pd
import requests


def load_dataset_from_bytes(
    content: bytes,
    filename: str = "dataset.csv",
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """Load a pandas DataFrame from raw bytes, detecting format from filename or content."""
    lower_name = filename.lower()
    
    if lower_name.endswith(".parquet") or lower_name.endswith(".pq"):
        df = pd.read_parquet(io.BytesIO(content))
        file_format = "parquet"
    elif lower_name.endswith(".json") or lower_name.endswith(".jsonl"):
        try:
            df = pd.read_json(io.BytesIO(content))
        except ValueError:
            df = pd.read_json(io.BytesIO(content), lines=True)
        file_format = "json"
    elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
        file_format = "excel"
    else:
        # Default to CSV/TSV with separator sniffing
        file_format = "csv"
        # Sample first few lines to detect delimiter
        sample = content[:4096].decode("utf-8", errors="ignore")
        sep = ","
        if "\t" in sample and sample.count("\t") > sample.count(","):
            sep = "\t"
        elif ";" in sample and sample.count(";") > sample.count(","):
            sep = ";"
        elif "|" in sample and sample.count("|") > sample.count(","):
            sep = "|"

        df = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            nrows=max_rows,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip"
        )

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    # Deduplicate column names if any
    cols = pd.Series(df.columns)
    for dup in df.columns[cols.duplicated()].unique():
        cols[df.columns == dup] = [
            f"{dup}_{i}" if i != 0 else dup
            for i in range((df.columns == dup).sum())
        ]
    df.columns = cols

    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df, file_format


def load_dataset_from_path(
    file_path: str,
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """Load DataFrame from local file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "rb") as f:
        content = f.read()
    return load_dataset_from_bytes(content, filename=os.path.basename(file_path), max_rows=max_rows)


def load_dataset_from_url(
    url: str,
    max_rows: Optional[int] = None,
    timeout: int = 30
) -> Tuple[pd.DataFrame, str]:
    """Download and load DataFrame from a remote URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    filename = url.split("/")[-1].split("?")[0] or "downloaded_data.csv"
    return load_dataset_from_bytes(response.content, filename=filename, max_rows=max_rows)


def load_dataset_from_base64(
    base64_str: str,
    filename: str = "dataset.csv",
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """Decode base64 string and load DataFrame."""
    # Strip data URL prefix if present (e.g., 'data:text/csv;base64,...')
    if "," in base64_str and "base64" in base64_str[:50]:
        base64_str = base64_str.split(",", 1)[1]
    
    raw_bytes = base64.b64decode(base64_str)
    return load_dataset_from_bytes(raw_bytes, filename=filename, max_rows=max_rows)
