# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory data models
"""

from enum import Enum
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentica.model.message import Message
from agentica.run_response import RunResponse


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


class Memory(BaseModel):
    """Model for Model memories"""

    memory: str
    input_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_str(self) -> str:
        attributes = self.model_dump(exclude_none=True)
        attributes_str = ";".join(f"{key}: {value}" for key, value in attributes.items())
        return attributes_str


class MemorySearchResponse(BaseModel):
    """Model for Memory Search Response used in agentic search."""

    memory_ids: List[str] = Field(
        ...,
        description="The IDs of the memories that are most semantically similar to the query.",
    )


class MemoryRetrieval(str, Enum):
    last_n = "last_n"
    first_n = "first_n"
    keyword = "keyword"
    agentic = "agentic"
