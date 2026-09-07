from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.decision.graph import run_decision
from app.decision.nodes import (
    action_decision, context_builder, decision_validator, department_router,
    impact_assessment, input_validator, persist_decision, priority_decision,
    recommendation_generator, severity_assessment,
)
from app.integrations.gemini import GeminiRecommendationProvider, RecommendationResult
from app.integrations.key_manager import GeminiKeyManager
from app.main import app


def record(**overrides) -> dict:
    value = {
        "feedback_id": "FB001",
        "feedback": "Campus Wi-Fi is down for all students.",
        "cleaned_text": "Campus Wi-Fi is down for all students.",
        "sentiment": "Negative",
        "sentiment_score": 0.95,
        "sentiment_confidence": 0.95,
        "topics": ["Wi-Fi"],
        "keywords": ["campus", "down", "students"],
        "emotion": "Frustration",
        "aspect_sentiments": [{"aspect": "Wi-Fi", "sentiment": "Negative"}],
    }
    value.update(overrides)
    return value


def test_input_validator_rejects_missing_analysis() -> None:
    result = input_validator({"feedback": "x"})
    assert result["validation_status"] == "INVALID"
    assert any("sentiment" in error for error in result["validation_errors"])


def test_each_decision_node_produces_explainable_output() -> None:
    state = record()
    state.update(input_validator(state))
    state.update(context_builder(state))
    state.update(severity_assessment(state))
    state.update(priority_decision(state))
    state.update(department_router(state))
    state.update(impact_assessment(state))
    state.update(action_decision(state))
    state.update(recommendation_generator(state, GeminiRecommendationProvider(api_key="")))
    state.update(decision_validator(state))
    assert state["severity"] == "CRITICAL"
    assert state["priority"] == "P1"
    assert state["department"] == "IT"
    assert state["impact_level"] == "CAMPUS_WIDE"
    assert state["action"] == "IMMEDIATE_ESCALATION"
    assert state["recommendation"]
    assert state["validation_status"] == "VALID"
    assert state["recommendation_status"] == "FALLBACK"


def test_persist_node_isolated_and_skips_invalid() -> None:
    result = persist_decision({"validation_status": "INVALID"})
    assert result["persistence_status"] == "SKIPPED_INVALID"


def test_end_to_end_graph_with_analyzed_row() -> None:
    result = run_decision(record())
    assert result["validation_status"] == "VALID"
    assert result["persistence_status"] == "NOT_CONFIGURED"
    assert result["final_decision"]["feedback_id"] == "FB001"


def test_decision_api_accepts_analyzed_record() -> None:
    with TestClient(app) as client:
        response = client.post("/api/decision", json=record())
    assert response.status_code == 200
    body = response.json()
    assert body["final_decision"]["department"] == "IT"
    assert body["persistence_status"] == "NOT_CONFIGURED"


def test_real_analyzed_csv_row_when_available() -> None:
    paths = list(Path(__file__).parents[2].rglob("analyzed_dataset_transformer(2).csv"))
    if not paths:
        pytest.skip("analyzed_dataset_transformer(2).csv is not present in this checkout")
    row = pd.read_csv(paths[0]).iloc[0].to_dict()
    for field in ("topics", "keywords", "aspect_sentiments"):
        value = row.get(field, [])
        if isinstance(value, str):
            try:
                row[field] = json.loads(value)
            except json.JSONDecodeError:
                row[field] = [value]
    result = run_decision(row)
    assert result["validation_status"] == "VALID"


class FakeProvider:
    def recommend(self, state: dict) -> RecommendationResult:
        return RecommendationResult("Recommendation", "Reason", "Next step", "GEMINI")


def test_recommendation_node_accepts_successful_gemini_result() -> None:
    state = record()
    state.update(recommendation_generator(state, FakeProvider()))
    assert state["recommendation_status"] == "GEMINI"
    assert state["suggested_next_step"] == "Next step"


def test_gemini_success_parses_structured_json(monkeypatch) -> None:
    import langchain_google_genai

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return {"recommendation": "Do it", "reason": "Because", "suggested_next_step": "Assign it"}

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", lambda **kwargs: Model())
    result = GeminiRecommendationProvider(api_key="test").recommend(record())
    assert result.status == "GEMINI"
    assert result.suggested_next_step == "Assign it"


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("API failed")])
def test_gemini_failures_use_fallback(monkeypatch, failure) -> None:
    import langchain_google_genai

    def failing_model(**kwargs):
        raise failure

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", failing_model)
    result = GeminiRecommendationProvider(api_key="test").recommend(
        record(action="ESCALATE", department="IT", severity="HIGH", priority="P2")
    )
    assert result.status == "FALLBACK"
    assert "IT" in result.recommendation


def test_invalid_gemini_json_uses_fallback(monkeypatch) -> None:
    import langchain_google_genai

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return {"recommendation": "missing reason"}

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", lambda **kwargs: Model())
    result = GeminiRecommendationProvider(api_key="test").recommend(record())
    assert result.status == "FALLBACK"


def test_row_recommendation_rotates_after_key_failure(monkeypatch) -> None:
    import langchain_google_genai

    seen = []

    class Model:
        def __init__(self, key):
            self.key = key

        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            seen.append(self.key)
            if self.key == "k1":
                raise RuntimeError("rate limited")
            return {"recommendation": "Do it", "reason": "Because", "suggested_next_step": "Assign it"}

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        lambda **kwargs: Model(kwargs["google_api_key"]),
    )
    result = GeminiRecommendationProvider(key_manager=GeminiKeyManager(["k1", "k2"])).recommend(record())
    assert result.status == "GEMINI"
    assert seen == ["k1", "k2"]


def test_fallback_without_api_key() -> None:
    result = GeminiRecommendationProvider(api_key="").recommend(record())
    assert result.status == "FALLBACK"


def test_decision_validator_rejects_contradictory_decisions() -> None:
    state = record(
        validation_status="VALID",
        severity="CRITICAL",
        priority="P4",
        impact_level="CAMPUS_WIDE",
        action="MONITOR",
        recommendation="x",
        recommendation_reason="x",
        suggested_next_step="x",
        recommendation_status="FALLBACK",
    )
    result = decision_validator(state)
    assert result["validation_status"] == "INVALID"
    assert any("inconsistent" in error for error in result["validation_errors"])
    assert any("escalation" in error for error in result["validation_errors"])


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1",
    reason="Set RUN_GEMINI_INTEGRATION=1 to run the optional Gemini integration test",
)
def test_optional_gemini_integration() -> None:
    result = GeminiRecommendationProvider().recommend(
        record(action="ESCALATE", department="IT", severity="HIGH", priority="P2")
    )
    assert result.status == "GEMINI"
    assert result.recommendation and result.reason and result.suggested_next_step
