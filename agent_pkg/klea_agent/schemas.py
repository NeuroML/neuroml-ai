#!/usr/bin/env python3
"""
Schemas used by the agent

File: klea_agent/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Literal

from fastmcp.client.client import CallToolResult
from klea_utils.graph.reducers import add_token_usage
from klea_utils.graph.schemas import TokenUsage
from klea_utils.mcp.schemas import ToolCallSchema
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Any


class CodeSchema(BaseModel):
    code: str = ""
    version: int = 0
    patches: list[str] = []


class StepSchema(BaseModel):
    step_number: int = 1
    description: str = ""
    suggested_tools: list[str] = Field(default_factory=list)
    depends_on: list[int] = []
    status: Literal["pending", "done", "failed"] = Field(
        default="pending", validate_default=True
    )


class PlanSchema(BaseModel):
    step_list: list[StepSchema] = Field(default_factory=list)
    status: Literal["not_started", "in_progress", "completed", "failed", "aborted"] = (
        Field(default="not_started", validate_default=True)
    )
    current_step_index: int = 0


class GoalSchema(BaseModel):
    goal: str = ""
    success_criteria: str = ""


class ArtefactSchema(BaseModel):
    id_: str = ""
    type_: str = ""
    content: Any
    # mtime!
    metadata: dict[str, Any] = {}


class Discovery(BaseModel):
    # when it was created
    timestamp: int = 0
    # TODO
    # general: files, scripts
    # NeuroML specific: files, semantic info (ions/parameters)
    pass


class KleaAgentState(BaseModel):
    """The state of the graph"""

    query: str = ""
    messages: list[AnyMessage] = Field(default_factory=list)
    guard_decision: str = "safe"
    usage_metrics: Annotated[TokenUsage, add_token_usage] = Field(
        default_factory=TokenUsage
    )
    # Operating mode: general is the default unverified agentic workflow;
    # scientific will enforce ADR-0029 evidence/verification invariants.
    # Surfaced at every node via NodeStreamData so the user is always aware
    # of the active mode; one-way downgrade (scientific -> general) allowed
    # with explicit permission, upgrade requires new session.
    mode: Literal["general", "scientific"] = Field(
        default="general", validate_default=True
    )

    # code string if any
    code: CodeSchema = CodeSchema()

    # planning related
    goal: GoalSchema = GoalSchema()
    plan: PlanSchema = PlanSchema()
    step_outputs: dict[int, list[CallToolResult]] = Field(default_factory=dict)
    # global project discovery information
    # only to be updated if files change
    discovery_persistent: Discovery = Discovery()
    # per step cache
    discovery_per_step: Discovery = Discovery()

    # { id -> artefact }
    artefacts: dict[str, ArtefactSchema] = Field(default_factory=dict)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""

    # selected tool calls and their results (one call per plan step, kept as
    # a list to share the tool caller/picker nodes with RAG)
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)
    tool_results: list[CallToolResult] = Field(default_factory=list)
