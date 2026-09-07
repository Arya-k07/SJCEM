from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.nlp import SENTIMENT_LABELS, analyze_dataset, analyze_feedback


def test_analyze_dataset_adds_domain_agnostic_fields() -> None:
    cleaned = pd.DataFrame(
        {
            "feedback_id": ["FB001", "FB002", "FB003"],
            "feedback": [
                "The delivery is SOOOO slow!!! 😡",
                "The product is beautiful but support is disappointing.",
                ":(",
            ],
            "cleaned_text": [
                "The delivery is SOOOO slow!!! 😡",
                "The product is beautiful but support is disappointing.",
                ":(",
            ],
        }
    )

    analyzed = analyze_dataset(cleaned)

    assert list(analyzed["feedback_id"]) == ["FB001", "FB002", "FB003"]
    assert set(analyzed["sentiment"]).issubset(SENTIMENT_LABELS)
    assert analyzed["sentiment_score"].between(0, 1).all()
    assert analyzed["sentiment_confidence"].between(0, 1).all()
    assert analyzed["cleaned_text"].equals(cleaned["cleaned_text"])
    assert analyzed["topics"].astype(str).str.len().gt(0).all()
    assert analyzed["keywords"].notna().all()
    assert analyzed["emotion"].astype(str).str.len().gt(0).all()


def test_mixed_sentiment_and_short_feedback_do_not_crash() -> None:
    mixed = analyze_feedback("The service is beautiful but the delivery is terrible.")
    short = analyze_feedback("👍")

    assert mixed["sentiment"] in SENTIMENT_LABELS
    assert mixed["sentiment"] == "Mixed"
    assert 0 <= mixed["sentiment_score"] <= 1
    assert short["sentiment"] in SENTIMENT_LABELS


def test_analyze_api_returns_enriched_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "feedback.csv"
    pd.DataFrame(
        {
            "feedback_id": ["1", "2"],
            "feedback": ["Delivery was great!!! 😊", "Billing is confusing :/"],
        }
    ).to_csv(dataset_path, index=False)

    with TestClient(app) as client:
        with dataset_path.open("rb") as dataset:
            response = client.post(
                "/api/analyze",
                files={"file": ("feedback.csv", dataset, "text/csv")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["rows_analyzed"] == 2
    assert Path(body["analyzed_file"]).exists()
    assert {"sentiment", "sentiment_score", "topics", "keywords", "emotion"}.issubset(
        body["data"][0]
    )


def test_topics_and_keywords_exclude_generic_terms() -> None:
    analyzed = analyze_dataset(
        pd.DataFrame(
            {
                "feedback": ["Teaching quality is actually great", "Wi-Fi is painfully slow in the evening"],
                "cleaned_text": ["Teaching quality is actually great", "Wi-Fi is painfully slow in the evening"],
            }
        )
    )

    for field in ("topics", "keywords"):
        for value in analyzed[field]:
            assert all(term not in {"is", "good", "great", "actually", "very", "the", "and"} for term in value.strip("[]").replace('"', "").split(", ") if term)


def test_topics_capture_aspects_without_sentiment_words() -> None:
    cleaned = pd.DataFrame(
        {
            "feedback": [
                "teaching quality is actually great",
                "Wi-Fi is painfully slow in the evening",
                "The library is great but the Wi-Fi is terrible.",
            ],
            "cleaned_text": [
                "teaching quality is actually great",
                "Wi-Fi is painfully slow in the evening",
                "The library is great but the Wi-Fi is terrible.",
            ],
        }
    )

    analyzed = analyze_dataset(cleaned)
    topic_text = " ".join(analyzed["topics"])

    assert "Teaching Quality" in topic_text
    assert "Wi-Fi" in topic_text
    assert "great" not in topic_text.casefold()
    assert "is" not in topic_text.casefold().split()


def test_topic_candidates_do_not_leak_discourse_or_time_fragments() -> None:
    cleaned = pd.DataFrame(
        {
            "cleaned_text": [
                "Not sure what happened but new classrooms are really good and faculty is approachable. imo",
                "network dies during online tests",
                "fans make weird noise achha hai",
            ]
        }
    )

    analyzed = analyze_dataset(cleaned)
    values = " ".join(analyzed["topics"].tolist()).casefold()

    assert "sure what" not in values
    assert "sometimes" not in values
    assert "today" not in values
    assert "achha" not in values
    assert "classrooms" in values
    assert "network" in values
    assert "noise" in values


def test_indirect_complaints_get_negative_sentiment_and_entity_aspects() -> None:
    cases = {
        "network dies during online tests": "Network",
        "washrooms need better maintenance": "Washrooms",
        "not enough healthy options": "Healthy Options",
        "too many last minute changes": "Last Minute Changes",
        "prices are too high": "Prices",
        "lift doesnt work sometimes": "Lift",
        "projector wasnt working in class": "Projector",
        "not enough working PCs": "Working Pcs",
        "library closes too early": "Library",
        "wifi in block B barely works": "Wifi",
    }
    analyzed = analyze_dataset(pd.DataFrame({"cleaned_text": list(cases)}))

    for text, expected_aspect in cases.items():
        row = analyzed.loc[analyzed["cleaned_text"] == text].squeeze()
        assert row["sentiment"] == "Negative", text
        assert expected_aspect in row["aspect_sentiments"], text

    washroom_keywords = analyzed.loc[
        analyzed["cleaned_text"] == "washrooms need better maintenance", "keywords"
    ].squeeze()
    assert "maintenance" in washroom_keywords


def test_neutrality_gate_preserves_facts_short_neutral_and_negation() -> None:
    cases = {
        "The library is open until 8 PM.": "Neutral",
        "Classes are scheduled from 10 to 4.": "Neutral",
        "The Wi-Fi is available in block A.": "Neutral",
        "okay": "Neutral",
        "fine": "Neutral",
        "average": "Neutral",
        "nothing special": "Neutral",
        "not bad": "Positive",
        "not good": "Negative",
        "network dies during online tests": "Negative",
        "teaching quality is actually great": "Positive",
        "The library is great but the Wi-Fi is terrible.": "Mixed",
    }
    analyzed = analyze_dataset(pd.DataFrame({"cleaned_text": list(cases)}))

    for text, expected in cases.items():
        row = analyzed.loc[analyzed["cleaned_text"] == text].squeeze()
        assert row["sentiment"] == expected, text
        assert 0 <= row["sentiment_confidence"] <= 1
