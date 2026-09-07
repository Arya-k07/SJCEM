from __future__ import annotations

from fastapi.testclient import TestClient

from app.decision.domain_profile import GENERIC_PROFILE
from app.decision.graph import run_decision
from app.insights.chain import DatasetInsightChain
from app.insights.context_builder import InsightContextBuilder
from app.insights.node import generate_dataset_insights
from app.integrations.key_manager import GeminiKeyManager
from app.main import app


def college_records() -> list[dict]:
    return [
        {
            "feedback_id": "1",
            "feedback": "Network is slow",
            "cleaned_text": "Network is slow",
            "sentiment": "Negative",
            "sentiment_score": 0.9,
            "sentiment_confidence": 0.9,
            "topics": ["Network"],
            "keywords": ["network"],
            "emotion": "Frustration",
            "aspect_sentiments": [{"aspect": "Network", "sentiment": "Negative"}],
            "severity": "MEDIUM",
            "priority": "P3",
            "department": "IT",
            "impact_level": "INDIVIDUAL",
            "action": "CREATE_TASK",
            "validation_status": "VALID",
        },
        {
            "feedback_id": "2",
            "feedback": "Teaching is excellent",
            "cleaned_text": "Teaching is excellent",
            "sentiment": "Positive",
            "sentiment_score": 0.9,
            "sentiment_confidence": 0.9,
            "topics": ["Teaching Quality"],
            "keywords": ["teaching"],
            "emotion": "Satisfaction",
            "aspect_sentiments": [{"aspect": "Teaching", "sentiment": "Positive"}],
            "severity": "LOW",
            "priority": "P4",
            "department": "Academic Affairs",
            "impact_level": "INDIVIDUAL",
            "action": "MONITOR",
            "validation_status": "VALID",
        },
    ]


def test_context_builder_college_data() -> None:
    context = InsightContextBuilder().build(college_records(), domain="college")
    assert context["dataset"]["row_count"] == 2
    assert context["dataset"]["domain"] == "college"
    assert context["sentiment"]["distribution"]["Negative"] == 1
    assert context["top_topics"][0]["value"] in {"Network", "Teaching Quality"}


def test_context_builder_generic_data_without_department() -> None:
    records = [
        {"feedback": "Billing was confusing", "sentiment": "Negative", "topics": ["Billing"]},
        {"feedback": "Delivery was fast", "sentiment": "Positive", "topics": ["Delivery"]},
    ]
    context = InsightContextBuilder().build(records)
    assert context["dataset"]["domain"] == "generic"
    assert context["categories"] == []
    assert context["top_topics"]


def test_context_builder_missing_optional_fields_and_empty_dataset() -> None:
    builder = InsightContextBuilder()
    assert builder.build([])["dataset"]["row_count"] == 0
    context = builder.build([{"feedback": "hello"}])
    assert context["dataset"]["date_range"] is None
    assert context["impact"]["distribution"] == {}


def test_generic_profile_runs_same_decision_architecture() -> None:
    result = run_decision(
        {
            "feedback": "Billing is confusing",
            "cleaned_text": "Billing is confusing",
            "sentiment": "Negative",
            "sentiment_score": 0.8,
            "sentiment_confidence": 0.8,
            "topics": ["Billing"],
            "keywords": ["billing"],
            "emotion": "Frustration",
            "aspect_sentiments": [],
        },
        profile=GENERIC_PROFILE,
    )
    assert result["category"] == "General"
    assert result["department"] == "General"


def test_insight_fallback_is_structured() -> None:
    chain = DatasetInsightChain()
    result = chain.fallback(InsightContextBuilder().build(college_records(), domain="college"))
    assert result.executive_summary
    assert result.generation_status in {"GEMINI", "FALLBACK"}


def test_insights_api_rejects_empty_records() -> None:
    with TestClient(app) as client:
        response = client.post("/api/insights", json={"records": []})
    assert response.status_code == 400


def test_dataset_insights_rotates_after_key_failure(monkeypatch) -> None:
    import langchain_core.prompts
    import langchain_google_genai

    seen = []

    class Pipeline:
        def invoke(self, values):
            return {"executive_summary": "ok"}

    class Prompt:
        def __or__(self, model):
            return Pipeline()

    class Model:
        def __init__(self, key):
            self.key = key

        def with_structured_output(self, schema):
            if self.key == "k1":
                raise RuntimeError("quota")
            return self

    def make_model(**kwargs):
        seen.append(kwargs["google_api_key"])
        return Model(kwargs["google_api_key"])

    monkeypatch.setattr(langchain_core.prompts.ChatPromptTemplate, "from_messages", lambda messages: Prompt())
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", make_model)
    from app.insights.chain import DatasetInsightChain

    result = DatasetInsightChain(GeminiKeyManager(["k1", "k2"])).generate(
        InsightContextBuilder().build(college_records())
    )
    assert seen == ["k1", "k2"]
    assert result.generation_status == "GEMINI"
