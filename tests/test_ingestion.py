import io
import pandas as pd
import pytest
from dataset_autopilot.ingestion import (
    load_dataset_from_base64,
    load_dataset_from_bytes,
    load_dataset_from_path,
)


def test_load_dataset_from_bytes_csv():
    csv_data = b"id,name,age,salary\n1,Alice,30,70000\n2,Bob,40,90000\n"
    df, fmt = load_dataset_from_bytes(csv_data, filename="test.csv")
    assert fmt == "csv"
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "age", "salary"]


def test_load_dataset_from_bytes_tsv():
    tsv_data = b"id\tval\n1\t100\n2\t200\n"
    df, fmt = load_dataset_from_bytes(tsv_data, filename="test.tsv")
    assert len(df) == 2
    assert "val" in df.columns


def test_load_dataset_from_base64():
    import base64
    csv_data = "a,b,c\n1,2,3\n4,5,6"
    b64_str = base64.b64encode(csv_data.encode()).decode()
    df, fmt = load_dataset_from_base64(b64_str)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]
