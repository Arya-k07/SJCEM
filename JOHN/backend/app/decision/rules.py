from __future__ import annotations

import re
from typing import Any

from app.decision.domain_profile import DomainProfile, college_profile

_CRITICAL = re.compile(r"\b(emergency|danger|unsafe|fire|medical|injur|security threat)\b", re.I)
_CAMPUS = re.compile(r"\b(campus|everyone|all students|whole college|entire)\b", re.I)
_URGENT = re.compile(r"\b(now|urgent|immediately|cannot access|down|outage)\b", re.I)


def text_context(state: dict[str, Any]) -> str:
    values = [state.get("feedback", ""), state.get("cleaned_text", "")]
    values += [str(value) for value in state.get("topics", [])]
    values += [str(value) for value in state.get("keywords", [])]
    return " ".join(values).casefold()


def assess_severity(state: dict[str, Any]) -> tuple[str, str]:
    text = text_context(state)
    sentiment = state.get("sentiment", "Neutral")
    if _CRITICAL.search(text):
        return "CRITICAL", "Safety or emergency language requires immediate attention."
    if _CAMPUS.search(text) and (_URGENT.search(text) or sentiment in {"Negative", "Mixed"}):
        return "CRITICAL", "The feedback indicates an urgent issue affecting the wider campus."
    if _URGENT.search(text):
        return "HIGH", "Urgency or strong negative emotion indicates a substantial operational issue."
    if sentiment == "Negative":
        return "MEDIUM", "Negative feedback describes an issue that should be addressed but has no verified emergency signal."
    if sentiment == "Mixed":
        return "MEDIUM", "Mixed feedback contains a concern that merits follow-up."
    return "LOW", "No strong operational or safety signal was found."


def decide_priority(state: dict[str, Any]) -> tuple[str, float, str]:
    scores = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}
    score = scores[state["severity"]]
    if state.get("impact_level") in {"DEPARTMENT_WIDE", "CAMPUS_WIDE"}:
        score = min(1.0, score + 0.1)
    priority = "P1" if score >= 0.9 else "P2" if score >= 0.65 else "P3" if score >= 0.4 else "P4"
    return priority, round(score, 2), f"Priority is based on severity ({state['severity']}) and assessed impact."


def route_department(state: dict[str, Any], profile: DomainProfile | None = None) -> tuple[str, str]:
    return (profile or college_profile()).category_for(state)


def assess_impact(state: dict[str, Any]) -> tuple[str, str]:
    text = text_context(state)
    if _CAMPUS.search(text) and state.get("sentiment") in {"Negative", "Mixed"}:
        return "CAMPUS_WIDE", "Campus-wide language indicates broad impact."
    if re.search(r"\b(several|many|multiple|students|users|department)\b", text, re.I):
        return "SMALL_GROUP", "Plural or group-oriented language indicates impact beyond one individual."
    if state.get("department") in {"IT", "Facilities"} and _URGENT.search(text):
        return "DEPARTMENT_WIDE", "An urgent service issue is likely to affect multiple users of the responsible department."
    return "INDIVIDUAL", "The feedback describes an individual or unspecified impact."


def decide_action(state: dict[str, Any]) -> tuple[str, str]:
    if state["severity"] == "CRITICAL" and state["impact_level"] == "CAMPUS_WIDE":
        return "IMMEDIATE_ESCALATION", "Critical campus-wide issues require immediate escalation."
    if state["severity"] == "HIGH":
        return "ESCALATE", "High-severity feedback should be escalated to the responsible department."
    if state["severity"] == "MEDIUM":
        return "CREATE_TASK", "Medium-severity feedback should create a trackable remediation task."
    if state.get("sentiment") in {"Negative", "Mixed"}:
        return "RESPOND", "A direct response is appropriate for non-critical concerns."
    return "MONITOR", "Low-risk feedback can be monitored."
