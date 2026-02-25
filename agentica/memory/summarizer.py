# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory summarizer for generating session summaries
"""

import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Any, Optional, cast, Tuple
from pydantic import ValidationError

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.memory.models import SessionSummary


@dataclass
class MemorySummarizer:
    model: Optional[Model] = None
    use_structured_outputs: bool = False

    def update_model(self) -> None:
        if self.model is None:
            from agentica.model.openai import OpenAIChat
            self.model = OpenAIChat()

        if self.use_structured_outputs:
            self.model.response_format = SessionSummary
            self.model.structured_outputs = True
        else:
            self.model.response_format = {"type": "json_object"}

    def get_system_message(self, messages_for_summarization: List[Dict[str, str]]) -> Message:
        system_prompt = dedent("""\
        Analyze the following conversation between a user and an assistant, and extract the following details:
          - Summary (str): Provide a concise summary of the session, focusing on important information that would be helpful for future interactions.
          - Topics (Optional[List[str]]): List the topics discussed in the session.
        Please ignore any frivolous information.

        Conversation:
        """)
        conversation = []
        for message_pair in messages_for_summarization:
            conversation.append(f"User: {message_pair['user']}")
            if "assistant" in message_pair:
                conversation.append(f"Assistant: {message_pair['assistant']}")
            elif "model" in message_pair:
                conversation.append(f"Assistant: {message_pair['model']}")

        system_prompt += "\n".join(conversation)

        if not self.use_structured_outputs:
            system_prompt += "\n\nRespond with a JSON object containing the following fields:"
            json_schema = SessionSummary.model_json_schema()
            response_model_properties = {}
            json_schema_properties = json_schema.get("properties")
            if json_schema_properties is not None:
                for field_name, field_properties in json_schema_properties.items():
                    formatted_field_properties = {
                        prop_name: prop_value
                        for prop_name, prop_value in field_properties.items()
                        if prop_name != "title"
                    }
                    response_model_properties[field_name] = formatted_field_properties

            if len(response_model_properties) > 0:
                field_names = [key for key in response_model_properties.keys() if key != '$defs']
                system_prompt += f"\n{json.dumps(field_names, ensure_ascii=False)}\n"
                system_prompt += "\nField schemas:\n"
                system_prompt += f"{json.dumps(response_model_properties, indent=2, ensure_ascii=False)}\n"

            system_prompt += (
                "\nIMPORTANT JSON formatting rules:\n"
                "- Output ONLY the JSON object, nothing else.\n"
                "- Start with { and end with }.\n"
                "- Use the EXACT field names shown above (snake_case).\n"
                "- Do NOT wrap the JSON in markdown code blocks.\n"
                "- Your output will be parsed by json.loads(), so it must be valid JSON."
            )
        return Message(role="system", content=system_prompt)

    def _prepare_messages(
            self,
            message_pairs: List[Tuple[Message, Message]],
    ) -> Optional[List[Message]]:
        """Prepare messages for summarization. Returns None if no valid input."""
        if message_pairs is None or len(message_pairs) == 0:
            logger.info("No message pairs provided for summarization.")
            return None

        self.update_model()

        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append(
                {
                    user_message.role: user_message.get_content_string(),
                    assistant_message.role: assistant_message.get_content_string(),
                }
            )

        return [self.get_system_message(messages_for_summarization)]

    def _parse_summary_response(self, response) -> Optional[SessionSummary]:
        """Parse summary from model response (shared logic for sync/async)."""
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed

        if isinstance(response.content, str):
            try:
                return self._parse_json_summary(response.content)
            except Exception as e:
                logger.warning(f"Failed to convert response to session_summary: {e}")
        return None

    @staticmethod
    def _parse_json_summary(content: str) -> Optional[SessionSummary]:
        """Parse JSON summary from response content, handling markdown code blocks."""
        try:
            return SessionSummary.model_validate_json(content)
        except ValidationError:
            # Handle various markdown code block formats
            cleaned = re.sub(r'^```(?:json|JSON)?\s*\n?', '', content.strip())
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            if cleaned != content.strip():
                try:
                    return SessionSummary.model_validate_json(cleaned)
                except ValidationError as exc:
                    logger.warning(f"Failed to validate session_summary response: {exc}")
            return None

    async def run(
            self,
            message_pairs: List[Tuple[Message, Message]],
            **kwargs: Any,
    ) -> Optional[SessionSummary]:
        messages_for_model = self._prepare_messages(message_pairs)
        if messages_for_model is None:
            return None

        self.model = cast(Model, self.model)
        response = await self.model.response(messages=messages_for_model)
        return self._parse_summary_response(response)
