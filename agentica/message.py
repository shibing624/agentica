# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model for LLM messages

part of the code from https://github.com/phidatahq/phidata
"""

import json
from typing import Optional, Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict

from agentica.utils.log import logger


class Message(BaseModel):
    """Model for LLM messages"""

    # The role of the message author.
    # One of system, user, assistant, or tool.
    role: str
    # The contents of the message. content is required for all messages,
    # and may be null for assistant messages with function calls.
    content: Optional[Union[List[Dict], str]] = None
    # An optional name for the participant.
    # Provides the model information to differentiate between participants of the same role.
    name: Optional[str] = None
    # Tool call that this message is responding to.
    tool_call_id: Optional[str] = None
    # The name of the tool call
    tool_call_name: Optional[str] = None
    # The error of the tool call
    tool_call_error: bool = False
    # The tool calls generated by the model, such as function calls.
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Metrics for the message, tokes + the time it took to generate the response.
    metrics: Dict[str, Any] = {}
    # Internal identifier for the message.
    internal_id: Optional[str] = None

    # DEPRECATED: The name and arguments of a function that should be called, as generated by the model.
    function_call: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(
            exclude_none=True, exclude={"metrics", "tool_call_name", "internal_id", "tool_call_error"})
        # Manually add the content field if it is None
        if self.content is None:
            _dict["content"] = None
        return _dict

    def log(self):
        """Log the message to the console
        """
        logger.debug(f"============== {self.role} ==============")
        if self.name:
            logger.debug(f"Name: {self.name}")
        if self.tool_call_id:
            logger.debug(f"Call Id: {self.tool_call_id}")
        if self.content:
            logger.debug(self.content)
        if self.tool_calls:
            logger.debug(f"Tool Calls: {json.dumps(self.tool_calls, indent=2, ensure_ascii=False)}")
        if self.function_call:
            logger.debug(f"Function Call: {json.dumps(self.function_call, indent=2, ensure_ascii=False)}")

    def content_is_valid(self) -> bool:
        """Check if the message content is valid."""

        return self.content is not None and len(self.content) > 0


def get_text_from_message(message: Union[List, Dict, str]) -> str:
    """Return the user texts from the message"""

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_messages = []
        if len(message) == 0:
            return ""

        if "type" in message[0]:
            for m in message:
                m_type = m.get("type")
                if m_type is not None and isinstance(m_type, str):
                    m_value = m.get(m_type)
                    if m_value is not None and isinstance(m_value, str):
                        if m_type == "text":
                            text_messages.append(m_value)
                        # if m_type == "image_url":
                        #     logger.warning(f"Image: {m_value}, not supported yet.")
        elif "role" in message[0]:
            for m in message:
                m_role = m.get("role")
                if m_role is not None and isinstance(m_role, str):
                    m_content = m.get("content")
                    if m_content is not None and isinstance(m_content, str):
                        if m_role == "user":
                            text_messages.append(m_content)
        if len(text_messages) > 0:
            return "\n".join(text_messages)
    return ""
