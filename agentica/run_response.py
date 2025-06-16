# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
from time import time
from enum import Enum
from typing import Optional, Any, Dict, List,Union, Iterable

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

    def __str__(self):
        return self.get_content_as_string()

    def __repr__(self):
        return f"RunResponse(run_id={self.run_id!r}, event={self.event!r}, content={self.content!r})"



def pprint_run_response(
    run_response: Union[RunResponse, Iterable[RunResponse]], markdown: bool = False, show_time: bool = False
) -> None:
    from rich.live import Live
    from rich.table import Table
    from rich.status import Status
    from rich.box import ROUNDED
    from rich.markdown import Markdown
    from rich.json import JSON
    from agentica.utils.console import console

    # If run_response is a single RunResponse, wrap it in a list to make it iterable
    if isinstance(run_response, RunResponse):
        single_response_content: Union[str, JSON, Markdown] = ""
        if isinstance(run_response.content, str):
            single_response_content = (
                Markdown(run_response.content) if markdown else run_response.get_content_as_string(indent=4)
            )
        elif isinstance(run_response.content, BaseModel):
            try:
                single_response_content = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
            except Exception as e:
                logger.warning(f"Failed to convert response to Markdown: {e}")
        else:
            try:
                single_response_content = JSON(json.dumps(run_response.content), indent=4)
            except Exception as e:
                logger.warning(f"Failed to convert response to string: {e}")

        table = Table(box=ROUNDED, border_style="blue", show_header=False)
        table.add_row(single_response_content)
        console.print(table)
    else:
        streaming_response_content: str = ""
        with Live(console=console) as live_log:
            status = Status("Working...", spinner="dots")
            live_log.update(status)
            response_timer = Timer()
            response_timer.start()
            for resp in run_response:
                if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                    streaming_response_content += resp.content

                formatted_response = Markdown(streaming_response_content) if markdown else streaming_response_content  # type: ignore
                table = Table(box=ROUNDED, border_style="blue", show_header=False)
                if show_time:
                    table.add_row(f"Response\n({response_timer.elapsed:.1f}s)", formatted_response)  # type: ignore
                else:
                    table.add_row(formatted_response)  # type: ignore
                live_log.update(table)
            response_timer.stop()