from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class FeedbackState(TypedDict, total=False):
    feedback_id: str
    date: str
    feedback: str
    source: str
    cleaned_text: str
    sentiment: str
    sentiment_score: float
    sentiment_confidence: float
    topics: list[str]
    keywords: list[str]
    emotion: str
    aspect_sentiments: list[dict[str, str]]
    context: dict[str, Any]
    severity: str
    severity_reason: str
    priority: str
    priority_score: float
    priority_reason: str
    department: str
    department_reason: str
    category: str
    category_reason: str
    impact_level: str
    impact_reason: str
    action: str
    action_reason: str
    recommendation: str
    recommendation_reason: str
    suggested_next_step: str
    recommendation_status: str
    validation_status: str
    validation_errors: list[str]
    final_decision: dict[str, Any]
    persistence_status: str


class AnalyzedFeedback(BaseModel):
    feedback_id: str | None = None
    date: str | None = None
    feedback: str
    source: str | None = None
    cleaned_text: str
    sentiment: str
    sentiment_score: float = Field(ge=0, le=1)
    sentiment_confidence: float = Field(ge=0, le=1)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    emotion: str
    aspect_sentiments: list[dict[str, str]] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    validation_status: str
    validation_errors: list[str] = Field(default_factory=list)
    final_decision: dict[str, Any] = Field(default_factory=dict)
    persistence_status: str = "SKIPPED_INVALID"
