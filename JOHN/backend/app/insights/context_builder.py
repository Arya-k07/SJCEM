from __future__ import annotations

import ast
import json
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

import pandas as pd


def _parse(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return [value]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return parsed if isinstance(parsed, list) else [parsed]


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.dropna().value_counts().items()}


def _percentages(values: dict[str, int], total: int) -> dict[str, float]:
    return {key: round(value * 100 / total, 2) for key, value in values.items()} if total else {}


class InsightContextBuilder:
    """Builds compact, deterministic evidence instead of sending raw rows to Gemini."""

    def build(self, records: Iterable[dict[str, Any]] | pd.DataFrame, domain: str | None = None) -> dict[str, Any]:
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        total = len(frame)
        fields = list(frame.columns)
        dimensions = [
            field for field in fields
            if field not in {"feedback", "cleaned_text", "topics", "keywords", "aspect_sentiments"}
            and frame[field].nunique(dropna=True) <= min(50, max(1, total))
        ]
        context: dict[str, Any] = {
            "dataset": {
                "row_count": total,
                "valid_rows": int((frame.get("validation_status", pd.Series(["VALID"] * total)) == "VALID").sum()) if total else 0,
                "date_range": self._date_range(frame),
                "available_dimensions": dimensions,
                "available_fields": fields,
                "domain": domain or "generic",
            },
            "sentiment": self._sentiment(frame),
            "severity": self._decision_distribution(frame, "severity"),
            "priority": self._decision_distribution(frame, "priority"),
            "impact": self._decision_distribution(frame, "impact_level"),
            "actions": self._decision_distribution(frame, "action"),
            "categories": self._categories(frame),
            "top_topics": self._top_list(frame, "topics"),
            "top_keywords": self._top_list(frame, "keywords"),
            "emotions": self._top_list(frame, "emotion"),
            "aspect_sentiments": self._aspect_summary(frame),
            "cross_metrics": self._cross_metrics(frame),
            "time_analysis": None,
            "hotspots": self._hotspots(frame),
            "representative_cases": self._representative_cases(frame),
        }
        return context

    def _date_range(self, frame: pd.DataFrame) -> dict[str, str] | None:
        for field in ("date", "created_at", "submitted_at", "timestamp"):
            if field in frame:
                parsed = pd.to_datetime(frame[field], errors="coerce").dropna()
                if not parsed.empty:
                    return {"start": parsed.min().isoformat(), "end": parsed.max().isoformat()}
        return None

    def _sentiment(self, frame: pd.DataFrame) -> dict[str, Any]:
        distribution = _distribution(frame["sentiment"]) if "sentiment" in frame else {}
        scores = pd.to_numeric(frame["sentiment_score"], errors="coerce").dropna() if "sentiment_score" in frame else pd.Series(dtype=float)
        return {"distribution": distribution, "average_score": round(float(scores.mean()), 4) if not scores.empty else None}

    def _decision_distribution(self, frame: pd.DataFrame, field: str) -> dict[str, Any]:
        values = _distribution(frame[field]) if field in frame else {}
        return {"distribution": values, "percentages": _percentages(values, len(frame))}

    def _categories(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        field = next((name for name in ("category", "department", "support_team", "product_category", "business_unit") if name in frame), None)
        if field is None:
            return []
        return [{"field": field, "value": str(key), "count": int(value)} for key, value in frame[field].value_counts().head(20).items()]

    def _top_list(self, frame: pd.DataFrame, field: str) -> list[dict[str, Any]]:
        if field not in frame:
            return []
        counter: Counter[str] = Counter()
        for value in frame[field]:
            counter.update(str(item) for item in _parse(value) if str(item).strip())
        return [{"value": key, "count": count} for key, count in counter.most_common(20)]

    def _aspect_summary(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if "aspect_sentiments" not in frame:
            return []
        counter: Counter[tuple[str, str]] = Counter()
        for value in frame["aspect_sentiments"]:
            for item in _parse(value):
                if isinstance(item, dict):
                    counter[(str(item.get("aspect", "")), str(item.get("sentiment", "")))] += 1
        return [{"aspect": aspect, "sentiment": sentiment, "count": count} for (aspect, sentiment), count in counter.most_common(30)]

    def _cross_metrics(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if not {"severity", "action"}.issubset(frame.columns):
            return []
        grouped = frame.groupby(["severity", "action"]).size().reset_index(name="count")
        return grouped.to_dict(orient="records")

    def _hotspots(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if not {"department", "severity"}.issubset(frame.columns):
            return []
        grouped = frame[frame["severity"].isin(["HIGH", "CRITICAL"])].groupby("department").size()
        return [{"dimension": "department", "value": str(key), "count": int(value)} for key, value in grouped.sort_values(ascending=False).head(10).items()]

    def _representative_cases(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        fields = [field for field in ("feedback_id", "feedback", "severity", "priority", "department", "action") if field in frame]
        if not fields:
            return []
        return frame[fields].head(5).fillna("").to_dict(orient="records")
