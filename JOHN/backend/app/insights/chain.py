from __future__ import annotations

import logging
from typing import Any

from app.config import gemini_model
from app.integrations.key_manager import GeminiKeyManager, get_gemini_key_manager
from app.insights.prompts import SYSTEM_PROMPT
from app.insights.schemas import DatasetInsights

logger = logging.getLogger(__name__)


class DatasetInsightChain:
    def __init__(self, key_manager: GeminiKeyManager | None = None) -> None:
        self.key_manager = key_manager or get_gemini_key_manager()

    def generate(self, context: dict[str, Any]) -> DatasetInsights:
        if not self.key_manager.count:
            return self.fallback(context)
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as error:
            logger.warning("[Gemini Insights] SDK unavailable: %s", type(error).__name__)
            return self.fallback(context)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Analyze this evidence JSON and return the required structured insight object:\n{context}"),
        ])
        for attempt in range(self.key_manager.count):
            slot = self.key_manager.next_slot()
            if slot is None:
                break
            logger.info("[Gemini Insights] Using key slot %s", slot.slot)
            try:
                model = ChatGoogleGenerativeAI(
                    model=gemini_model(),
                    google_api_key=slot.key,
                    temperature=0,
                ).with_structured_output(DatasetInsights)
                result = (prompt | model).invoke({"context": context})
                parsed = result if isinstance(result, DatasetInsights) else DatasetInsights.model_validate(result)
                parsed.generation_status = "GEMINI"
                return parsed
            except Exception as error:
                logger.warning("[Gemini Insights] Key slot %s failed: %s", slot.slot, type(error).__name__)
                if attempt + 1 < self.key_manager.count:
                    logger.info("[Gemini Insights] Trying next key")
        logger.info("[Gemini Insights] All configured keys failed; using deterministic fallback")
        return self.fallback(context)

    @staticmethod
    def fallback(context: dict[str, Any]) -> DatasetInsights:
        dataset = context["dataset"]
        severity = context["severity"]["distribution"]
        actions = context["actions"]["distribution"]
        top_topics = context["top_topics"][:5]
        summary = f"Processed {dataset['row_count']} records with deterministic decision evidence."
        if severity:
            summary += f" Severity distribution: {severity}."
        insights = [
            {
                "title": "Decision workload",
                "description": f"Recorded actions are distributed as {actions or 'unavailable'}.",
                "importance": "MEDIUM",
                "evidence": [str(actions or "Action data unavailable")],
            }
        ]
        themes = [
            {
                "theme": item["value"],
                "description": f"Recurring analyzed topic with {item['count']} records.",
                "sentiment": "UNKNOWN",
                "importance": "MEDIUM",
            }
            for item in top_topics
        ]
        return DatasetInsights(
            executive_summary=summary,
            key_insights=insights,
            themes=themes,
            data_limitations=["Gemini was unavailable; this result contains deterministic summaries only."],
            generation_status="FALLBACK",
        )
