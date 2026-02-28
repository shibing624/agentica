# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Work memory for managing conversation history and session summaries (short-term memory)
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, ConfigDict

from agentica.db.base import BASE64_PLACEHOLDER, clean_media_placeholders
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.run_response import RunResponse
from agentica.utils.tokens import count_message_tokens
from agentica.memory.models import (
    AgentRun,
    SessionSummary,
)
from agentica.memory.summarizer import MemorySummarizer


def _clean_content_list(content_list: list) -> list:
    """Clean a list-type content by removing BASE64 placeholder items."""
    cleaned = []
    for item in content_list:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "image_url":
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else ""
                if BASE64_PLACEHOLDER in str(url):
                    continue
            elif item_type == "input_audio":
                audio_data = item.get("input_audio", {})
                data = audio_data.get("data", "") if isinstance(audio_data, dict) else ""
                if BASE64_PLACEHOLDER in str(data):
                    continue
            elif item_type == "text":
                text = item.get("text", "")
                if BASE64_PLACEHOLDER in str(text):
                    continue
            else:
                if BASE64_PLACEHOLDER in str(item):
                    continue
            cleaned.append(item)
        elif isinstance(item, str):
            if BASE64_PLACEHOLDER not in item:
                cleaned.append(item)
        else:
            cleaned.append(item)
    return cleaned


def _clean_media_list(media_list: list) -> Optional[list]:
    """Clean a list of media items (images/videos) by removing placeholders."""
    cleaned = []
    for item in media_list:
        if isinstance(item, str) and BASE64_PLACEHOLDER in item:
            continue
        elif isinstance(item, dict):
            result = clean_media_placeholders(item)
            if result is not None:
                cleaned.append(result)
        else:
            cleaned.append(item)
    return cleaned if cleaned else None


def _clean_message_for_history(msg: Message) -> Message:
    """Clean a message by removing filtered media placeholders."""
    cleaned_msg = msg.model_copy(deep=True)

    if cleaned_msg.content is not None:
        if isinstance(cleaned_msg.content, list):
            cleaned_content_list = _clean_content_list(cleaned_msg.content)
            if len(cleaned_content_list) == 1 and isinstance(cleaned_content_list[0], dict):
                if cleaned_content_list[0].get("type") == "text":
                    cleaned_msg.content = cleaned_content_list[0].get("text", "")
                else:
                    cleaned_msg.content = cleaned_content_list
            elif len(cleaned_content_list) == 0:
                cleaned_msg.content = ""
            else:
                cleaned_msg.content = cleaned_content_list
        elif isinstance(cleaned_msg.content, str):
            if BASE64_PLACEHOLDER in cleaned_msg.content:
                cleaned_msg.content = ""

    if cleaned_msg.images is not None:
        cleaned_msg.images = _clean_media_list(cleaned_msg.images)

    if cleaned_msg.audio is not None:
        if isinstance(cleaned_msg.audio, str) and BASE64_PLACEHOLDER in cleaned_msg.audio:
            cleaned_msg.audio = None
        elif isinstance(cleaned_msg.audio, dict):
            cleaned_msg.audio = clean_media_placeholders(cleaned_msg.audio)

    if cleaned_msg.videos is not None:
        cleaned_msg.videos = _clean_media_list(cleaned_msg.videos)

    return cleaned_msg


def _is_history_message(msg: Message) -> bool:
    """Check if a message should be included in history for multi-turn conversations.

    Includes tool-related messages (assistant tool_calls, tool responses) to preserve
    tool call context across turns, preventing the model from forgetting tool usage.
    """
    if msg.role == "system":
        return False
    if msg.role in ("user", "assistant", "tool"):
        return True
    return False


def _truncate_tool_content(msg: Message, max_chars: int = 500) -> Message:
    """Truncate long content in tool response messages to save token space.

    Only truncates tool role messages; user/assistant messages are kept intact.
    """
    if msg.role != "tool":
        return msg
    if not msg.content or not isinstance(msg.content, str):
        return msg
    if len(msg.content) <= max_chars:
        return msg

    truncated = msg.model_copy(deep=True)
    head_size = max_chars * 2 // 3
    tail_size = max_chars - head_size
    truncated.content = (
        msg.content[:head_size]
        + f"\n...[truncated {len(msg.content) - max_chars} chars]...\n"
        + msg.content[-tail_size:]
    )
    return truncated


class WorkingMemory(BaseModel):
    """Work memory for managing conversation history and session summaries (short-term memory).

    WorkingMemory focuses on runtime conversation management:
    - Session history (runs, messages)
    - Session summaries (optional)

    For persistent user memories and preferences, use Workspace instead:
    - Workspace stores memories in human-readable Markdown files
    - Supports version control (Git)
    - Easy to edit and share

    Example - Session management only (recommended):
        >>> memory = WorkingMemory(create_session_summary=True)
        >>> agent = Agent(memory=memory)

    Example - With Workspace for persistent memory:
        >>> from agentica.workspace import Workspace
        >>> workspace = Workspace("~/.agentica/workspace")
        >>> agent = Agent(workspace=workspace)
    """

    # ========== Session Management (Core Features) ==========
    runs: List[AgentRun] = []
    messages: List[Message] = []
    update_system_message_on_change: bool = False

    create_session_summary: bool = False
    update_session_summary_after_run: bool = True
    summary: Optional[SessionSummary] = None
    summarizer: Optional[MemorySummarizer] = None

    updating_memory: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def with_summary(cls, **kwargs) -> "WorkingMemory":
        """Factory method to create WorkingMemory with session summary enabled."""
        return cls(
            create_session_summary=True,
            update_session_summary_after_run=True,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(
            exclude_none=True,
            exclude={
                "summary",
                "summarizer",
                "updating_memory",
            },
        )
        if self.summary:
            _memory_dict["summary"] = self.summary.to_dict()
        return _memory_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        """Create an WorkingMemory instance from a dictionary."""
        if not data:
            return cls()

        data_copy = data.copy()

        if "runs" in data_copy and data_copy["runs"]:
            runs = []
            for run_data in data_copy["runs"]:
                if isinstance(run_data, dict):
                    if "message" in run_data and run_data["message"]:
                        run_data["message"] = Message(**run_data["message"])
                    if "messages" in run_data and run_data["messages"]:
                        run_data["messages"] = [Message(**m) for m in run_data["messages"]]
                    if "response" in run_data and run_data["response"]:
                        run_data["response"] = RunResponse(**run_data["response"])
                    runs.append(AgentRun(**run_data))
                elif isinstance(run_data, AgentRun):
                    runs.append(run_data)
            data_copy["runs"] = runs

        if "messages" in data_copy and data_copy["messages"]:
            data_copy["messages"] = [
                Message(**m) if isinstance(m, dict) else m
                for m in data_copy["messages"]
            ]

        if "summary" in data_copy and data_copy["summary"]:
            if isinstance(data_copy["summary"], dict):
                data_copy["summary"] = SessionSummary(**data_copy["summary"])

        # Remove deprecated fields from old serialized data
        for field_name in [
            "summarizer", "db", "classifier", "manager",
            "memories", "create_user_memories", "update_user_memories_after_run",
            "user_id", "retrieval", "num_memories",
        ]:
            data_copy.pop(field_name, None)

        return cls(**data_copy)

    def add_run(self, agent_run: AgentRun) -> None:
        """Adds an AgentRun to the runs list."""
        self.runs.append(agent_run)

    def add_system_message(self, message: Message, system_message_role: str = "system") -> None:
        """Add the system messages to the messages list"""
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            if system_message_index is not None:
                if (
                        self.messages[system_message_index].content != message.content
                        and self.update_system_message_on_change
                ):
                    logger.info("Updating system message in memory with new content")
                    self.messages[system_message_index] = message
            else:
                self.messages.insert(0, message)

    def add_message(self, message: Message) -> None:
        """Add a Message to the messages list."""
        self.messages.append(message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add a list of messages to the messages list."""
        self.messages.extend(messages)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the messages list as a list of dictionaries."""
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def get_messages_from_last_n_runs(
            self,
            last_n: Optional[int] = None,
            skip_role: Optional[str] = None,
            max_tokens: Optional[int] = None,
            truncate_tool_results: bool = True,
            tool_result_max_chars: int = 500,
    ) -> List[Message]:
        """Returns the messages from the last_n runs with token budget awareness."""
        if last_n is None:
            selected_runs = self.runs
        else:
            selected_runs = self.runs[-last_n:]

        if not selected_runs:
            return []

        runs_messages: List[List[Message]] = []
        for prev_run in selected_runs:
            if prev_run.response and prev_run.response.messages:
                run_msgs = [
                    m for m in prev_run.response.messages
                    if _is_history_message(m) and (not skip_role or m.role != skip_role)
                ]
                cleaned = [_clean_message_for_history(m) for m in run_msgs]
                if cleaned:
                    runs_messages.append(cleaned)

        if not runs_messages:
            return []

        if max_tokens is None:
            all_messages = []
            for i, run_msgs in enumerate(runs_messages):
                if truncate_tool_results and i < len(runs_messages) - 1:
                    run_msgs = [_truncate_tool_content(m, tool_result_max_chars) for m in run_msgs]
                all_messages.extend(run_msgs)
            logger.debug(f"History messages: {len(all_messages)} (no token limit)")
            return all_messages

        total_tokens = 0
        collected_runs: List[List[Message]] = []

        for i in range(len(runs_messages) - 1, -1, -1):
            run_msgs = runs_messages[i]
            is_older_run = i < len(runs_messages) - 1

            if truncate_tool_results and is_older_run:
                run_msgs = [_truncate_tool_content(m, tool_result_max_chars) for m in run_msgs]

            run_tokens = sum(count_message_tokens(m) for m in run_msgs)

            if total_tokens + run_tokens > max_tokens:
                if is_older_run:
                    truncated = [_truncate_tool_content(m, tool_result_max_chars // 2) for m in run_msgs]
                    run_tokens = sum(count_message_tokens(m) for m in truncated)
                    if total_tokens + run_tokens <= max_tokens:
                        collected_runs.insert(0, truncated)
                        total_tokens += run_tokens
                        continue
                if not collected_runs:
                    collected_runs.append(run_msgs)
                    total_tokens += run_tokens
                break

            collected_runs.insert(0, run_msgs)
            total_tokens += run_tokens

        result = []
        for run_msgs in collected_runs:
            result.extend(run_msgs)

        logger.debug(f"History messages: {len(result)}, estimated tokens: {total_tokens}")
        return result

    def get_message_pairs(
            self, user_role: str = "user", assistant_role: Optional[List[str]] = None
    ) -> List[Tuple[Message, Message]]:
        """Returns a list of tuples of (user message, assistant response)."""

        if assistant_role is None:
            assistant_role = ["assistant", "model", "CHATBOT"]

        runs_as_message_pairs: List[Tuple[Message, Message]] = []
        for run in self.runs:
            if run.response and run.response.messages:
                user_messages_from_run = None
                assistant_messages_from_run = None

                for message in run.response.messages:
                    if message.role == user_role:
                        user_messages_from_run = message
                        break

                for message in run.response.messages[::-1]:
                    if message.role in assistant_role:
                        assistant_messages_from_run = message
                        break

                if user_messages_from_run and assistant_messages_from_run:
                    runs_as_message_pairs.append((user_messages_from_run, assistant_messages_from_run))
        return runs_as_message_pairs

    def get_tool_calls(self, num_calls: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns a list of tool calls from the messages"""

        tool_calls = []
        for message in self.messages[::-1]:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(tool_call)
                    if num_calls and len(tool_calls) >= num_calls:
                        return tool_calls
        return tool_calls

    async def update_summary(self) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        self.updating_memory = True
        try:
            if self.summarizer is None:
                self.summarizer = MemorySummarizer()
            self.summary = await self.summarizer.run(self.get_message_pairs())
            return self.summary
        finally:
            self.updating_memory = False

    def clear(self) -> None:
        """Clear the WorkingMemory"""
        self.runs = []
        self.messages = []
        self.summary = None

