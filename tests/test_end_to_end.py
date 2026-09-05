import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from dataset_autopilot.api import app
from dataset_autopilot.engine import DatasetAutopilotEngine


client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Dataset Autopilot API"


def test_api_samples_list():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    ids = [d["id"] for d in data]
    assert "telecom_churn_leakage" in ids


def test_end_to_end_audit():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "user_id": [f"ID-{i}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.normal(50000, 10000, size=n),
        "leaky_feature": np.random.choice([0, 1], size=n),
        "churn": np.random.choice([0, 1], size=n)
    })
    # Make leaky_feature 95% correlated with churn
    df["leaky_feature"] = np.where(df["churn"] == 1, 1, 0)

    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="synthetic_audit.csv", target_col="churn")

    assert report.health_score > 0
    assert report.dataset_profile.rows == 150
    assert len(report.findings) > 0
    assert any(f.type.value == "target_leakage" for f in report.findings)
    assert len(report.experiments) >= 1
