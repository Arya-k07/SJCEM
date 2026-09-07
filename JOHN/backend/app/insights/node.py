from __future__ import annotations

from typing import Any

from app.insights.chain import DatasetInsightChain
from app.insights.context_builder import InsightContextBuilder
from app.insights.schemas import DatasetInsights


def generate_dataset_insights(
    records: list[dict[str, Any]],
    domain: str | None = None,
    chain: DatasetInsightChain | None = None,
) -> DatasetInsights:
    context = InsightContextBuilder().build(records, domain=domain)
    return (chain or DatasetInsightChain()).generate(context)
