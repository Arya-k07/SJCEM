"""Domain-agnostic, provider-based NLP enrichment for cleaned feedback."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


SENTIMENT_LABELS = {"Positive", "Negative", "Neutral", "Mixed"}
GENERIC_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "could", "did", "do", "does",
    "for", "from", "good", "great", "bad", "better", "in", "is", "it", "of", "on", "or", "our",
    "really", "so", "that", "the", "their", "this", "to", "too", "very", "was", "were", "with",
    "would", "actually", "not", "no", "tbh", "imo", "idk", "lol", "pls", "okay", "fine", "terrible",
}
NON_TOPIC_TERMS = GENERIC_TERMS | {
    "am", "been", "being", "can", "cannot", "come", "coming", "didnt", "doesnt", "dont",
    "during", "feel", "feels", "find", "found", "get", "gets", "give", "go", "going", "gone",
    "has", "have", "having", "happened", "improve", "into", "keep", "keeps", "know", "make",
    "makes", "need", "needs", "please", "should", "slow", "fast", "late", "early", "confusing",
    "comfortable", "uncomfortable", "outdated", "painfully", "extremely", "highly", "kinda", "much",
    "work", "working", "works", "wasnt", "were", "will", "without", "look", "looks", "tell",
    "sure", "what", "happened", "sometimes", "today", "although", "though", "some", "many",
    "enough", "both", "well", "please", "into", "achha", "accha", "hai", "yaar", "kinda",
    "decent", "supportive", "helpful", "approachable", "beautiful", "amazing", "excellent",
    "dies", "fails", "failure", "barely", "closes", "early", "high", "low", "late", "slow",
    "enough", "many", "much", "few", "little", "need", "outdated", "missing", "confusing", "maintenance",
}
ISSUE_TERMS = {
    "maintenance", "high", "low", "late", "early", "slow", "dies", "fails", "failure", "barely",
    "outdated", "missing", "confusing", "boring", "disconnecting", "disconnected", "changes",
}
NEGATIVE_STRUCTURE_PATTERNS = (
    r"\bnot\s+enough\b",
    r"\b(?:too|very)\s+(?:many|much|high|low|late|early|slow|expensive)\b",
    r"\b(?:doesn['’]?t|dont|doesnt|didn['’]?t|wasn['’]?t|isn['’]?t|aren['’]?t)\s+(?:work|working|function)\b",
    r"\b(?:barely|hardly)\s+works?\b",
    r"\b(?:dies|fails|failure|outdated|missing|confusing|disappointing|terrible|awful|poor)\b",
    r"\bneeds?\s+(?:better\s+)?(?:maintenance|improvement|attention|repair)\b",
)
CONTENT_TOKEN = re.compile(r"(?u)[\w][\w'-]*")


class SentimentProvider(Protocol):
    def analyze(self, text: str) -> tuple[str, float, float]:
        """Return sentiment label, confidence, and polarity."""


_TRANSFORMER_MODEL_NAME = os.getenv(
    "FEEDBACKIQ_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
_TRANSFORMER_CACHE: dict[str, Any] = {}
_EMBEDDING_CACHE: dict[str, Any] = {}


POSITIVE_EVIDENCE = re.compile(
    r"\b(?:great|good|excellent|amazing|helpful|improved|beautiful|supportive|love|loved|enjoy)\b|[😊😄😁😍👍]|(?::\)|:d|;\))",
    re.IGNORECASE,
)
NEGATIVE_EVIDENCE = re.compile(
    r"\b(?:bad|terrible|awful|poor|outdated|confusing|fails?|failure|dies|slow|barely|disappointing|angry|hate|hated)\b|[😡😠😭]|(?:\:\(|\:\/)",
    re.IGNORECASE,
)
FACTUAL_PATTERNS = (
    r"\b(?:is|are)\s+(?:available|open|located|closed)\b",
    r"\b(?:scheduled|held|runs?)\s+(?:from|at|on)\b",
    r"\bfrom\s+\d+\s*(?:am|pm)?\s+to\s+\d+\s*(?:am|pm)?\b",
    r"\b(?:there is|there are)\b",
)
NEUTRAL_SHORT_TEXT = {"okay", "fine", "average", "nothing special"}


def _contextual_sentiment(
    text: str,
    label: str,
    confidence: float,
    polarity: float,
    margin: float | None = None,
) -> tuple[str, float, float]:
    """Apply the existing contrast/problem layer to model output."""
    clauses = [part.strip() for part in re.split(r"\b(?:but|however|although|though)\b|[;]", text, flags=re.I) if part.strip()]
    clause_polarities = [TextBlob(clause).sentiment.polarity for clause in clauses] or [0.0]
    problem_signal = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NEGATIVE_STRUCTURE_PATTERNS)
    positive_signal = bool(POSITIVE_EVIDENCE.search(text))
    negative_signal = bool(NEGATIVE_EVIDENCE.search(text)) or problem_signal
    negated_positive = bool(re.search(r"\bnot\s+(?:good|great|helpful|excellent)\b", text, re.I))
    negated_negative = bool(re.search(r"\bnot\s+(?:bad|terrible|awful)\b", text, re.I))
    positive_signal = (positive_signal or negated_negative) and not negated_positive
    negative_signal = (negative_signal or negated_positive) and not negated_negative
    if problem_signal:
        label = "Negative"
        confidence = max(confidence, 0.78)
        polarity = min(-0.35, polarity)
    if any(value > 0.12 for value in clause_polarities) and any(value < -0.12 for value in clause_polarities):
        label = "Mixed"
        confidence = max(confidence, 0.78)
    weak_model = confidence < 0.78 or (margin is not None and margin < 0.56)
    factual = any(re.search(pattern, text, re.IGNORECASE) for pattern in FACTUAL_PATTERNS)
    short_neutral = text.strip().casefold() in NEUTRAL_SHORT_TEXT
    if not positive_signal and not negative_signal and (weak_model or factual or short_neutral):
        label = "Neutral"
        confidence = max(0.5, min(0.8, 1.0 - (margin or 0.0)))
    return label, round(max(0.0, min(1.0, confidence)), 4), polarity


class TransformerSentimentProvider:
    """Lazy, CPU-friendly Hugging Face sentiment provider with TextBlob fallback."""

    provider_name = "transformer"

    def _load(self) -> bool:
        if "loaded" in _TRANSFORMER_CACHE:
            return bool(_TRANSFORMER_CACHE["loaded"])
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(_TRANSFORMER_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(_TRANSFORMER_MODEL_NAME)
            model.to(device)
            model.eval()
            _TRANSFORMER_CACHE.update({"loaded": True, "tokenizer": tokenizer, "model": model, "device": device})
        except Exception as error:  # model downloads and optional dependencies are environmental
            _TRANSFORMER_CACHE.update({"loaded": False, "error": str(error)})
        return bool(_TRANSFORMER_CACHE["loaded"])

    @property
    def provider_name(self) -> str:
        return "transformer" if self._load() else "textblob_fallback"

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple[str, float, float]]:
        if not self._load():
            fallback = TextBlobSentimentProvider()
            return [fallback.analyze(text) for text in texts]
        import torch

        tokenizer = _TRANSFORMER_CACHE["tokenizer"]
        model = _TRANSFORMER_CACHE["model"]
        device = _TRANSFORMER_CACHE["device"]
        results: list[tuple[str, float, float]] = []
        id_to_label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
                for batch_index, probability in enumerate(probabilities):
                    index = int(torch.argmax(probability).item())
                    confidence = float(probability[index].item())
                    label = "Positive" if id_to_label[index] == "POSITIVE" else "Negative"
                    polarity = confidence if label == "Positive" else -confidence
                    margin = float(abs(probability[0].item() - probability[1].item()))
                    results.append(_contextual_sentiment(batch[batch_index], label, confidence, polarity, margin))
        return results

    def analyze(self, text: str) -> tuple[str, float, float]:
        return self.analyze_batch([text])[0]


class SemanticTopicGrouper:
    """Optional small sentence-transformer normalizer for extracted topic phrases."""

    def _load(self) -> Any | None:
        if "attempted" in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE.get("model")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(os.getenv("FEEDBACKIQ_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            _EMBEDDING_CACHE.update({"attempted": True, "model": model})
        except Exception as error:  # optional model/download failure must not break NLP
            _EMBEDDING_CACHE.update({"attempted": True, "model": None, "error": str(error)})
        return _EMBEDDING_CACHE.get("model")

    def normalize(self, topic_lists: list[list[str]]) -> list[list[str]]:
        phrases = list(dict.fromkeys(phrase for values in topic_lists for phrase in values))
        if len(phrases) < 2:
            return topic_lists
        model = self._load()
        if model is None:
            return topic_lists
        import numpy as np

        vectors = model.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
        canonical: dict[str, str] = {}
        for index, phrase in enumerate(phrases):
            if phrase in canonical:
                continue
            similarities = np.dot(vectors, vectors[index])
            for match_index in np.where(similarities >= 0.84)[0]:
                canonical[phrases[match_index]] = phrase
        return [[canonical.get(phrase, phrase) for phrase in values] for values in topic_lists]


@dataclass
class TextBlobSentimentProvider:
    """Offline baseline provider that combines polarity with clause contrast."""

    def analyze(self, text: str) -> tuple[str, float, float]:
        if not text.strip():
            return "Neutral", 0.0, 0.0
        clauses = [part.strip() for part in re.split(r"\b(?:but|however|although|though)\b|[;]", text, flags=re.I) if part.strip()]
        polarities = [TextBlob(clause).sentiment.polarity for clause in clauses] or [0.0]
        polarity = float(TextBlob(text).sentiment.polarity)
        problem_signal = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NEGATIVE_STRUCTURE_PATTERNS)
        if problem_signal:
            polarity = min(-0.35, polarity - 0.35)
        if any(mark in text for mark in ("😡", "😠", "😭", ":(", ":/")):
            polarity = min(-0.25, polarity - 0.15)
        if any(mark in text for mark in ("😊", "😄", "👍", ":)", ":D")):
            polarity = max(0.25, polarity + 0.15)
        has_positive = any(value > 0.12 for value in polarities)
        has_negative = any(value < -0.12 for value in polarities)
        if len(polarities) > 1 and has_positive and has_negative:
            label = "Mixed"
        elif polarity > 0.12:
            label = "Positive"
        elif polarity < -0.12:
            label = "Negative"
        else:
            label = "Neutral"
        confidence = min(1.0, 0.5 + abs(polarity) * 0.5 + (0.08 if label == "Mixed" else 0))
        return label, round(confidence, 4), polarity


def _emotion(label: str, polarity: float, subjectivity: float, text: str) -> str:
    if any(mark in text for mark in ("😡", "😠")):
        return "Anger"
    if any(mark in text for mark in ("😭", ":(", ":/")):
        return "Frustration"
    if label == "Positive":
        return "Satisfaction" if subjectivity >= 0.35 else "Joy"
    if label == "Negative":
        return "Frustration" if polarity <= -0.45 else "Disappointment"
    if label == "Mixed":
        return "Concern"
    return "Neutral"


def _content_tokens(text: str) -> list[str]:
    return [token for token in CONTENT_TOKEN.findall(text.casefold()) if token not in GENERIC_TERMS and len(token) > 1]


def _candidate_phrases(text: str) -> list[str]:
    candidates: list[str] = []
    segment: list[str] = []
    for token in CONTENT_TOKEN.findall(text.casefold()):
        if token in NON_TOPIC_TERMS and token not in ISSUE_TERMS:
            if segment:
                candidates.append(" ".join(segment))
                segment = []
            continue
        if len(token) > 1:
            segment.append(token)
    if segment:
        candidates.append(" ".join(segment))
    expanded: list[str] = []
    for candidate in candidates:
        words = candidate.split()
        for size in (3, 2, 1):
            for index in range(len(words) - size + 1):
                phrase = " ".join(words[index:index + size])
                if phrase not in expanded:
                    expanded.append(phrase)
    return expanded


def _discover_topics_and_keywords(texts: list[str]) -> tuple[list[str], list[str]]:
    if not texts:
        return [], []
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(GENERIC_TERMS),
        ngram_range=(1, 3),
        min_df=1,
        max_features=4000,
        token_pattern=r"(?u)\b[\w][\w'-]*\b",
    )
    matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    topics: list[str] = []
    keywords: list[str] = []
    topic_values_by_row: list[list[str]] = []
    for row_index, text in enumerate(texts):
        weights = matrix[row_index].toarray().ravel()
        weighted = {vocabulary[index]: weights[index] for index in weights.argsort()[::-1] if weights[index] > 0}
        candidates = [
            phrase for phrase in _candidate_phrases(text)
            if phrase in weighted or any(part in weighted for part in phrase.split())
        ]
        candidates.sort(key=lambda phrase: max(weighted.get(part, 0.0) for part in phrase.split()) * len(phrase.split()), reverse=True)
        selected: list[str] = []
        selected_keywords: list[str] = []
        for phrase in candidates:
            if any(phrase == item or phrase in item or item in phrase for item in selected_keywords):
                continue
            selected_keywords.append(phrase)
            if len(selected_keywords) == 5:
                break
        for phrase in candidates:
            phrase_terms = phrase.split()
            topic_valid = phrase_terms[-1] not in NON_TOPIC_TERMS and not all(term in NON_TOPIC_TERMS for term in phrase_terms)
            if not topic_valid or any(phrase == item or phrase in item or item in phrase for item in selected):
                continue
            selected.append(phrase)
            if len(selected) == 3:
                break
        topic_values_by_row.append(selected[:2])
        keywords.append(json.dumps(selected_keywords, ensure_ascii=False))
    normalized_topics = SemanticTopicGrouper().normalize(topic_values_by_row)
    topics = [json.dumps([value.title() for value in values], ensure_ascii=False) for values in normalized_topics]
    return topics, keywords


def _aspect_sentiments(text: str, provider: SentimentProvider) -> str:
    clauses = [part.strip() for part in re.split(r"\b(?:but|however|although|though)\b|[;]", text, flags=re.I) if part.strip()]
    values = []
    for clause in clauses:
        aspect = _extract_aspect(clause)
        if aspect:
            label, _, _ = provider.analyze(clause)
            values.append({"aspect": aspect.title(), "sentiment": label})
    return json.dumps(values, ensure_ascii=False)


def _extract_aspect(clause: str) -> str:
    """Extract the entity being evaluated, excluding its problem description."""
    value = clause.strip(" .,!?:;")
    patterns = (
        r"^(?:not enough|too many|too much|too few|too little)\s+(.+)$",
        r"^(.+?)\s+(?:doesn['’]?t|doesnt|don['’]?t|dont|wasn['’]?t|isn['’]?t|aren['’]?t)\s+(?:work|working|function).*$",
        r"^(.+?)\s+(?:barely|hardly)\s+works?.*$",
        r"^(.+?)\s+(?:dies|fails|closes)\b.*$",
        r"^(.+?)\s+(?:needs?|need)\s+(?:better\s+)?(?:maintenance|improvement|attention|repair).*$",
        r"^(.+?)\s+(?:is|are|was|were)\s+(?:too\s+)?(?:high|low|late|early|slow|expensive|outdated|confusing).*$",
    )
    for pattern in patterns:
        match = re.match(pattern, value, flags=re.IGNORECASE)
        if match:
            candidate = " ".join(_content_tokens(match.group(1)))
            if candidate:
                return candidate
    tokens = _content_tokens(value)
    return " ".join(tokens[:2])


def analyze_feedback(text: str, provider: SentimentProvider | None = None) -> dict[str, Any]:
    provider = provider or TextBlobSentimentProvider()
    label, confidence, polarity = provider.analyze(text)
    label = label if label in SENTIMENT_LABELS else "Neutral"
    subjectivity = float(TextBlob(text).sentiment.subjectivity) if text.strip() else 0.0
    return {
        "sentiment": label,
        "sentiment_score": round(max(0.0, min(1.0, confidence)), 4),
        "emotion": _emotion(label, polarity, subjectivity, text),
    }


def analyze_dataset(cleaned_data: pd.DataFrame, provider: SentimentProvider | None = None) -> pd.DataFrame:
    """Return cleaned data enriched with validated semantic NLP fields."""
    if "cleaned_text" not in cleaned_data.columns:
        raise ValueError("The cleaned dataset must contain a cleaned_text column.")
    provider = provider or TransformerSentimentProvider()
    analyzed = cleaned_data.copy()
    texts = analyzed["cleaned_text"].fillna("").astype(str).tolist()
    unique_texts = list(dict.fromkeys(texts))
    if hasattr(provider, "analyze_batch"):
        unique_results = dict(zip(unique_texts, provider.analyze_batch(unique_texts), strict=True))
        results = [
            analyze_feedback(text, _CachedSentimentProvider(unique_results))
            for text in texts
        ]
    else:
        results = [analyze_feedback(text, provider) for text in texts]
    topics, keywords = _discover_topics_and_keywords(texts)
    analyzed["sentiment"] = [result["sentiment"] for result in results]
    analyzed["sentiment_score"] = [result["sentiment_score"] for result in results]
    analyzed["sentiment_confidence"] = analyzed["sentiment_score"]
    analyzed["topics"] = topics
    analyzed["keywords"] = keywords
    analyzed["emotion"] = [result["emotion"] for result in results]
    analyzed["aspect_sentiments"] = [_aspect_sentiments(text, provider) for text in texts]
    return analyzed


@dataclass
class _CachedSentimentProvider:
    values: dict[str, tuple[str, float, float]]

    def analyze(self, text: str) -> tuple[str, float, float]:
        return self.values[text]


def export_analyzed_dataset(analyzed_data: pd.DataFrame, output_path: str) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    analyzed_data.to_csv(destination, index=False, encoding="utf-8")
    return destination.as_posix()
