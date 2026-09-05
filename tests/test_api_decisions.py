import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from dataset_autopilot.api import DATASETS, REPORTS, app
from dataset_autopilot.engine import DatasetAutopilotEngine


client = TestClient(app)


@pytest.fixture
def setup_completed_run():
    np.random.seed(42)
    n = 120
    salary = np.random.normal(50000, 10000, size=n)
    salary[:20] = np.nan # 16% missing
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, size=n),
        "salary": salary,
        "churn": np.random.choice([0, 1], size=n)
    })

    run_id = "TEST-RUN-001"
    engine = DatasetAutopilotEngine()
    report = engine.run_audit(df, dataset_name="test_telecom.csv", target_col="churn")

    REPORTS[run_id] = report
    DATASETS[run_id] = df
    return run_id, report


def test_get_decision_candidates(setup_completed_run):
    run_id, _ = setup_completed_run
    response = client.get(f"/api/runs/{run_id}/decision-candidates")
    assert response.status_code == 200
    candidates = response.json()
    assert len(candidates) >= 1
    assert any(c["category"] == "missingness_strategy" for c in candidates)

    first_cand = candidates[0]
    assert "evidence_block" in first_cand
    assert first_cand["evidence_block"] is not None
    assert len(first_cand["evidence_block"]["what_we_know"]) >= 1
    assert len(first_cand["evidence_block"]["what_is_unknown"]) >= 1
    assert "controlled_parameters" in first_cand["evidence_block"]


def test_execute_decision_experiments_and_decision_lifecycle(setup_completed_run):
    run_id, report = setup_completed_run
    cand_id = report.decision_candidates[0].decision_id

    # 1. Run Experiments
    resp_exp = client.post(
        f"/api/runs/{run_id}/decisions/{cand_id}/experiments",
        json={"strategy_ids": ["DROP_FEATURE", "MEDIAN_IMPUTATION", "MISSING_INDICATOR"]}
    )
    assert resp_exp.status_code == 200
    data = resp_exp.json()
    assert data["status"] == "EVIDENCE_AVAILABLE"
    assert len(data["experiment_results"]) == 3

    # 2. Get Results
    resp_res = client.get(f"/api/runs/{run_id}/decisions/{cand_id}/results")
    assert resp_res.status_code == 200
    results = resp_res.json()
    assert len(results) == 3
    assert results[0]["is_baseline"] is True

    # 3. Record Human Decision
    resp_dec = client.post(
        f"/api/runs/{run_id}/decisions/{cand_id}/decision",
        json={
            "selected_strategy_id": "MISSING_INDICATOR",
            "decision_rationale": "Retains all rows and adds predictive signal with minimal complexity.",
            "custom_method_notes": "Tested in 5-fold CV."
        }
    )
    assert resp_dec.status_code == 200
    dec_data = resp_dec.json()
    assert dec_data["status"] == "USER_DECISION"
    assert dec_data["user_decision"]["selected_strategy_id"] == "MISSING_INDICATOR"

    # 4. Download Reproducible Notebook
    resp_nb = client.get(f"/api/runs/{run_id}/decisions/{cand_id}/artifacts/notebook")
    assert resp_nb.status_code == 200
    nb_json = json.loads(resp_nb.content.decode("utf-8"))
    assert nb_json["nbformat"] == 4
    assert len(nb_json["cells"]) > 4

    # 5. Download Reproducible Pipeline
    resp_pipe = client.get(f"/api/runs/{run_id}/decisions/{cand_id}/artifacts/pipeline")
    assert resp_pipe.status_code == 200
    assert "class DecisionPreprocessor" in resp_pipe.text

    # 6. Download HTML Report
    resp_html = client.get(f"/api/runs/{run_id}/decisions/{cand_id}/artifacts/html")
    assert resp_html.status_code == 200
    assert "<!DOCTYPE html>" in resp_html.text

    # 7. Download JSON Contract
    resp_json = client.get(f"/api/runs/{run_id}/decisions/{cand_id}/artifacts/json")
    assert resp_json.status_code == 200
    cand_obj = json.loads(resp_json.text)
    assert cand_obj["decision_id"] == cand_id
