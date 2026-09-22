# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory data models
"""

from enum import Enum
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentica.model.message import Message
from agentica.run.response import RunResponse


class MemoryType(str, Enum):
    """Typed memory classification for workspace memory entries.

    USER: user role, preferences, communication style, technical background.
    FEEDBACK: corrections and validated approaches from prior interactions.
        Record both failures ("don't do X") and successes ("yes, do it this way").
    PROJECT: non-code-derivable project context (deadlines, decisions, rationale).
    REFERENCE: external system pointers (issue trackers, dashboards, API docs).
    """
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class MemoryEntry(BaseModel):
    """A single typed memory entry parsed from a memory file's frontmatter."""
    name: str = Field(description="Short display name for the memory")
    description: str = Field(
        description="One-line hook used for relevance scoring. Should contain searchable keywords."
    )
    memory_type: MemoryType = Field(description="Memory category")
    file_path: str = Field(description="Path to the memory file (relative to workspace)")
    content: str = Field(description="Full memory content (frontmatter stripped)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentRun(BaseModel):
    message: Optional[Message] = None
    messages: Optional[List[Message]] = None
    response: Optional[RunResponse] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionSummary(BaseModel):
    """Model for Session Summary."""

    summary: str = Field(
        ...,
        description="Summary of the session. Be concise and focus on only important information. Do not make anything up.",
    )
    topics: Optional[List[str]] = Field(None, description="Topics discussed in the session.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)
