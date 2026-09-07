from __future__ import annotations

from typing import Any

from app.decision.rules import assess_impact, assess_severity, decide_action, decide_priority, route_department
from app.decision.domain_profile import DomainProfile
from app.decision.schemas import FeedbackState
from app.integrations.gemini import GeminiRecommendationProvider
from app.integrations.supabase import SupabaseDecisionStore

REQUIRED_FIELDS = (
    "feedback", "cleaned_text", "sentiment", "sentiment_score",
    "sentiment_confidence", "topics", "keywords", "emotion", "aspect_sentiments",
)


def input_validator(state: FeedbackState) -> FeedbackState:
    errors = [f"Missing required field: {field}" for field in REQUIRED_FIELDS if field not in state or state[field] is None]
    return {"validation_status": "INVALID" if errors else "VALID", "validation_errors": errors}


def context_builder(state: FeedbackState) -> FeedbackState:
    return {"context": {key: state.get(key) for key in REQUIRED_FIELDS}}


def severity_assessment(state: FeedbackState) -> FeedbackState:
    severity, reason = assess_severity(state)
    return {"severity": severity, "severity_reason": reason}


def priority_decision(state: FeedbackState) -> FeedbackState:
    priority, score, reason = decide_priority(state)
    return {"priority": priority, "priority_score": score, "priority_reason": reason}


def department_router(state: FeedbackState, profile: DomainProfile | None = None) -> FeedbackState:
    category, reason = route_department(state, profile)
    return {"department": category, "department_reason": reason, "category": category, "category_reason": reason}


def impact_assessment(state: FeedbackState) -> FeedbackState:
    impact, reason = assess_impact(state)
    return {"impact_level": impact, "impact_reason": reason}


def action_decision(state: FeedbackState) -> FeedbackState:
    action, reason = decide_action(state)
    return {"action": action, "action_reason": reason}


def recommendation_generator(
    state: FeedbackState,
    provider: GeminiRecommendationProvider | None = None,
) -> FeedbackState:
    result = (provider or GeminiRecommendationProvider()).recommend(state)
    return {
        "recommendation": result.recommendation,
        "recommendation_reason": result.reason,
        "suggested_next_step": result.suggested_next_step,
        "recommendation_status": result.status,
    }


def decision_validator(state: FeedbackState) -> FeedbackState:
    errors = list(state.get("validation_errors", []))
    if state.get("validation_status") != "VALID":
        return {"validation_status": "INVALID", "validation_errors": errors}
    for field in (
        "severity", "priority", "department", "impact_level", "action",
        "recommendation", "recommendation_reason", "suggested_next_step",
        "recommendation_status",
    ):
        if not state.get(field):
            errors.append(f"Decision field is empty: {field}")
    severity = state.get("severity")
    priority = state.get("priority")
    action = state.get("action")
    impact = state.get("impact_level")
    valid_priorities = {
        "CRITICAL": {"P1"},
        "HIGH": {"P1", "P2"},
        "MEDIUM": {"P2", "P3"},
        "LOW": {"P3", "P4"},
    }
    if severity in valid_priorities and priority not in valid_priorities[severity]:
        errors.append(f"Priority {priority} is inconsistent with severity {severity}")
    if severity == "CRITICAL" and action not in {"ESCALATE", "IMMEDIATE_ESCALATION"}:
        errors.append("Critical decisions cannot use a non-escalation action")
    if impact == "CAMPUS_WIDE" and action == "MONITOR":
        errors.append("Campus-wide decisions cannot use MONITOR")
    return {"validation_status": "INVALID" if errors else "VALID", "validation_errors": errors}


def persist_decision(
    state: FeedbackState,
    store: SupabaseDecisionStore | None = None,
) -> FeedbackState:
    if state.get("validation_status") != "VALID":
        return {"persistence_status": "SKIPPED_INVALID"}
    decision = {key: state.get(key) for key in (
        "feedback_id", "severity", "severity_reason", "priority", "priority_score",
        "priority_reason", "department", "department_reason", "impact_level",
        "impact_reason", "action", "action_reason", "recommendation",
        "recommendation_reason", "suggested_next_step", "recommendation_status",
    )}
    result = (store or SupabaseDecisionStore()).save_decision(decision)
    return {"persistence_status": result["status"], "final_decision": decision}
