from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.decision.nodes import (
    action_decision, context_builder, decision_validator, department_router,
    impact_assessment, input_validator, persist_decision, priority_decision,
    recommendation_generator, severity_assessment,
)
from app.decision.schemas import FeedbackState
from app.decision.domain_profile import DomainProfile, college_profile


def build_decision_graph(profile: DomainProfile | None = None) -> object:
    active_profile = profile or college_profile()
    graph = StateGraph(FeedbackState)
    graph.add_node("input_validator", input_validator)
    graph.add_node("context_builder", context_builder)
    graph.add_node("severity_assessment", severity_assessment)
    graph.add_node("priority_decision", priority_decision)
    graph.add_node("department_router", lambda state: department_router(state, active_profile))
    graph.add_node("impact_assessment", impact_assessment)
    graph.add_node("action_decision", action_decision)
    graph.add_node("recommendation_generator", recommendation_generator)
    graph.add_node("decision_validator", decision_validator)
    graph.add_node("persist_decision", persist_decision)
    graph.add_edge(START, "input_validator")
    graph.add_conditional_edges("input_validator", lambda state: "context_builder" if state["validation_status"] == "VALID" else END)
    graph.add_edge("context_builder", "severity_assessment")
    graph.add_edge("severity_assessment", "priority_decision")
    graph.add_edge("priority_decision", "department_router")
    graph.add_edge("department_router", "impact_assessment")
    graph.add_edge("impact_assessment", "action_decision")
    graph.add_edge("action_decision", "recommendation_generator")
    graph.add_edge("recommendation_generator", "decision_validator")
    graph.add_conditional_edges("decision_validator", lambda state: "persist_decision" if state["validation_status"] == "VALID" else END)
    graph.add_edge("persist_decision", END)
    return graph.compile()


def run_decision(record: dict, profile: DomainProfile | None = None) -> FeedbackState:
    return build_decision_graph(profile).invoke(record)
