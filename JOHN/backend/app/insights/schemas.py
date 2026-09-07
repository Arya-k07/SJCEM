from __future__ import annotations

from pydantic import BaseModel, Field


class KeyInsight(BaseModel):
    title: str
    description: str
    importance: str
    evidence: list[str] = Field(default_factory=list)


class Hotspot(BaseModel):
    dimension: str
    value: str
    reason: str
    evidence: str


class Trend(BaseModel):
    metric: str
    direction: str
    description: str
    evidence: str


class Theme(BaseModel):
    theme: str
    description: str
    sentiment: str
    importance: str


class PositiveSignal(BaseModel):
    area: str
    description: str
    evidence: str


class Anomaly(BaseModel):
    description: str
    evidence: str
    confidence: str


class StrategicRecommendation(BaseModel):
    priority: str
    action: str
    reason: str
    target: str


class DatasetInsights(BaseModel):
    executive_summary: str
    key_insights: list[KeyInsight] = Field(default_factory=list)
    hotspots: list[Hotspot] = Field(default_factory=list)
    trends: list[Trend] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    positive_signals: list[PositiveSignal] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    recommendations: list[StrategicRecommendation] = Field(default_factory=list)
    data_limitations: list[str] = Field(default_factory=list)
    generation_status: str = "FALLBACK"
