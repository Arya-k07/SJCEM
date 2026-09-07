from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import gemini_model
from app.integrations.key_manager import GeminiKeyManager, get_gemini_key_manager
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationResult:
    recommendation: str
    reason: str
    suggested_next_step: str
    status: str


class RecommendationPayload(BaseModel):
    recommendation: str
    reason: str
    suggested_next_step: str


class RecommendationProvider(Protocol):
    def recommend(self, state: dict[str, Any]) -> RecommendationResult: ...


class GeminiRecommendationProvider:
    """Gemini recommendation adapter with deterministic failure handling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        key_manager: GeminiKeyManager | None = None,
    ) -> None:
        self.key_manager = key_manager or (
            GeminiKeyManager([api_key]) if api_key is not None else get_gemini_key_manager()
        )
        self.model = model or gemini_model()

    def recommend(self, state: dict[str, Any]) -> RecommendationResult:
        logger.info("[Gemini] Recommendation generation started")
        if not self.key_manager.count:
            logger.info("[Gemini] API unavailable; using deterministic fallback")
            return self.fallback(state)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.info("[Gemini] SDK unavailable; using deterministic fallback")
            return self.fallback(state)
        for attempt in range(self.key_manager.count):
            slot = self.key_manager.next_slot()
            if slot is None:
                break
            logger.info("[Gemini] Using key slot %s", slot.slot)
            try:
                model = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=slot.key,
                    temperature=0,
                ).with_structured_output(RecommendationPayload)
                payload = model.invoke(self._prompt(state))
                parsed = RecommendationPayload.model_validate(payload)
                logger.info("[Gemini] Recommendation generated successfully")
                return RecommendationResult(
                    parsed.recommendation.strip(),
                    parsed.reason.strip(),
                    parsed.suggested_next_step.strip(),
                    "GEMINI",
                )
            except Exception as error:
                logger.warning("[Gemini] Key slot %s failed: %s", slot.slot, type(error).__name__)
                if attempt + 1 < self.key_manager.count:
                    logger.info("[Gemini] Key slot %s failed, trying next key", slot.slot)
        logger.info("[Gemini] All configured keys failed, using deterministic fallback")
        return self.fallback(state)

    @staticmethod
    def _prompt(state: dict[str, Any]) -> str:
        return f"""You are a decision-support assistant for an educational institution's feedback management system.

The feedback has already been analyzed by an AI/NLP pipeline. The operational decision has already been determined by a LangGraph decision engine. Do not reclassify or change any supplied analysis or decision. Generate only a concise, practical recommendation for the responsible department.

Return JSON with exactly these string fields: recommendation, reason, suggested_next_step.
Base the response strictly on the supplied evidence. Do not invent facts or exaggerate.

Feedback: {state.get("feedback", "")}
Cleaned text: {state.get("cleaned_text", "")}
Sentiment: {state.get("sentiment", "")}
Sentiment score: {state.get("sentiment_score", "")}
Sentiment confidence: {state.get("sentiment_confidence", "")}
Topics: {state.get("topics", [])}
Keywords: {state.get("keywords", [])}
Emotion: {state.get("emotion", "")}
Aspect sentiments: {state.get("aspect_sentiments", [])}
Severity: {state.get("severity", "")}
Severity reason: {state.get("severity_reason", "")}
Priority: {state.get("priority", "")}
Priority reason: {state.get("priority_reason", "")}
Department: {state.get("department", "")}
Department reason: {state.get("department_reason", "")}
Impact: {state.get("impact_level", "")}
Impact reason: {state.get("impact_reason", "")}
Action: {state.get("action", "")}
Action reason: {state.get("action_reason", "")}"""

    @staticmethod
    def fallback(state: dict[str, Any]) -> RecommendationResult:
        action = str(state.get("action", "review")).replace("_", " ").lower()
        department = state.get("department", "responsible")
        severity = state.get("severity", "unclassified")
        priority = state.get("priority", "unassigned")
        return RecommendationResult(
            f"{action.title()} this {severity.lower()}-severity issue with the {department} department.",
            f"The LangGraph decision classified this issue as {severity} severity with {priority} priority.",
            f"Ask {department} to review the feedback and complete the {action} action.",
            "FALLBACK",
        )
