# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
from time import time
from enum import Enum
from typing import Optional, Any, Dict, List, Union, Iterable

from pydantic import BaseModel, ConfigDict, Field

from agentica.model.content import Video, Image, Audio
from agentica.reasoning import ReasoningStep
from agentica.utils.log import logger
from agentica.utils.timer import Timer
from agentica.model.message import Message, MessageReferences


class RunEvent(str, Enum):
    """Events that can be sent by the run() functions"""

    run_started = "RunStarted"
    run_response = "RunResponse"
    run_completed = "RunCompleted"
    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"
    reasoning_started = "ReasoningStarted"
    reasoning_step = "ReasoningStep"
    reasoning_completed = "ReasoningCompleted"
    updating_memory = "UpdatingMemory"
    workflow_started = "WorkflowStarted"
    workflow_completed = "WorkflowCompleted"
    # Multi-round events
    multi_round_turn = "MultiRoundTurn"
    multi_round_tool_call = "MultiRoundToolCall"
    multi_round_tool_result = "MultiRoundToolResult"
    multi_round_completed = "MultiRoundCompleted"


class RunResponseExtraData(BaseModel):
    references: Optional[List[MessageReferences]] = None
    add_messages: Optional[List[Message]] = None
    history: Optional[List[Message]] = None
    reasoning_steps: Optional[List[ReasoningStep]] = None
    reasoning_messages: Optional[List[Message]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class RunResponse(BaseModel):
    """Response returned by Agent.run() or Workflow.run() functions"""

    content: Optional[Any] = None
    content_type: str = "str"

    event: str = RunEvent.run_response.value
    messages: Optional[List[Message]] = None
    metrics: Optional[Dict[str, Any]] = None

    model: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None

    tools: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Image]] = None  # Images attached to the response
    videos: Optional[List[Video]] = None  # Videos attached to the response
    audio: Optional[List[Audio]] = None  # Audio attached to the response
    response_audio: Optional[Dict] = None  # Model audio response
    reasoning_content: Optional[str] = ""
    extra_data: Optional[RunResponseExtraData] = None
    created_at: int = Field(default_factory=lambda: int(time()))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_json(self) -> str:
        _dict = self.model_dump(
            exclude_none=True,
            exclude={"messages"},
        )
        if self.messages is not None:
            _dict["messages"] = [
                m.model_dump(
                    exclude_none=True,
                    exclude={"parts"},  # Exclude what Gemini adds
                )
                for m in self.messages
            ]
        return json.dumps(_dict, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(
            exclude_none=True,
            exclude={"messages"},
        )
        if self.messages is not None:
            _dict["messages"] = [m.to_dict() for m in self.messages]
        return _dict

    def get_content_as_string(self, **kwargs) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            return json.dumps(self.content, ensure_ascii=False, **kwargs)

    def __str__(self) -> str:
        """Return content as string for easy printing."""
        return self.get_content_as_string()

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"RunResponse(run_id={self.run_id!r}, event={self.event!r}, content={self.content!r})"


def pprint_run_response(run_response: Union[RunResponse, Iterable[RunResponse]]) -> None:
    """
    Pretty print run response without rich dependency.

    Args:
        run_response: Single RunResponse or iterable of RunResponse objects
    """

    # Handle single RunResponse
    if isinstance(run_response, RunResponse):
        print("=" * 80)
        print("🤖 RESPONSE")
        print("=" * 80)

        # Display reasoning content if available
        if hasattr(run_response, 'reasoning_content') and run_response.reasoning_content:
            print("💭 THINKING")
            print("-" * 40)
            print(run_response.reasoning_content)
            print()
            print("-" * 40)
            print("💬 ANSWER")
            print("-" * 40)

        # Display main content
        if isinstance(run_response.content, str):
            print(run_response.content)
        elif isinstance(run_response.content, BaseModel):
            try:
                content_json = run_response.content.model_dump_json(exclude_none=True, indent=2)
                print(content_json)
            except Exception as e:
                logger.warning(f"Failed to convert BaseModel response to JSON: {e}")
                print(str(run_response.content))
        else:
            try:
                content_json = json.dumps(run_response.content, indent=2, ensure_ascii=False)
                print(content_json)
            except Exception as e:
                logger.warning(f"Failed to convert response to JSON: {e}")
                print(str(run_response.content))

        print("=" * 80)
    else:
        # Handle streaming responses
        print("=" * 80)
        print("🤖 RESPONSE")
        print("=" * 80)

        streaming_content = ""
        reasoning_content = ""
        reasoning_displayed = False
        response_timer = Timer()
        response_timer.start()

        for resp in run_response:
            if isinstance(resp, RunResponse):
                # Handle reasoning content
                if (hasattr(resp, 'reasoning_content') and resp.reasoning_content
                        and resp.reasoning_content != reasoning_content):

                    if not reasoning_displayed:
                        print("💭 THINKING")
                        print("-" * 40)
                        reasoning_displayed = True

                    print(resp.reasoning_content, end='', flush=True)
                    reasoning_content = resp.reasoning_content

                # Handle main content
                if isinstance(resp.content, str) and resp.content != streaming_content:
                    # Add separator when transitioning from reasoning to answer
                    if reasoning_displayed and streaming_content == "":
                        print()
                        print("-" * 40)
                        print("💬 ANSWER")
                        print("-" * 40)

                    print(resp.content, end='', flush=True)
                    streaming_content = resp.content
        response_timer.stop()
