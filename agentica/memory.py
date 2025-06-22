# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from __future__ import annotations

from enum import Enum
from textwrap import dedent
import json
from typing import Dict, List, Any, Optional, cast, Tuple, Literal
from copy import deepcopy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentica.memorydb import MemoryDb, MemoryRow
from agentica.model.openai import OpenAIChat
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.run_response import RunResponse

__all__ = [
    "AgentRun", "SessionSummary", "Memory", "MemoryManager", "MemoryClassifier", "MemoryRetrieval",
    "MemorySummarizer", "AgentMemory", "WorkflowMemory", "WorkflowRun"
]


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


class MemoryManager(BaseModel):
    """
    MemoryManager class to manage the memories for the user,
        including adding, updating, deleting, and clearing memories.
    """

    mode: Literal["model", "rule"] = "rule"
    model: Optional[Model] = None
    user_id: Optional[str] = None

    # Provide the system prompt for the manager as a string
    system_prompt: Optional[str] = None
    # Memory Database
    db: Optional[MemoryDb] = None

    # Do not set the input message here, it will be set by the run method
    input_message: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_model(self) -> None:
        if self.model is None:
            self.model = OpenAIChat()
            logger.debug(f"Using Model: {self.model}")
        self.model.add_tool(self.add_memory)
        self.model.add_tool(self.update_memory)
        self.model.add_tool(self.delete_memory)
        self.model.add_tool(self.clear_memory)

    def get_existing_memories(self) -> Optional[List[MemoryRow]]:
        if self.db is None:
            return None

        return self.db.read_memories(user_id=self.user_id)

    def add_memory(self, memory: str) -> str:
        """Use this function to add a memory to the database.
        Args:
            memory (str): The memory to be stored.
        Returns:
            str: A message indicating if the memory was added successfully or not.
        """
        try:
            if self.db:
                self.db.upsert_memory(
                    MemoryRow(
                        user_id=self.user_id,
                        memory=Memory(memory=memory, input_text=self.input_message).to_dict()
                    )
                )
            return "Memory added successfully"
        except Exception as e:
            logger.warning(f"Error storing memory in db: {e}")
            return f"Error adding memory: {e}"

    def delete_memory(self, id: str) -> str:
        """Use this function to delete a memory from the database.
        Args:
            id (str): The id of the memory to be deleted.
        Returns:
            str: A message indicating if the memory was deleted successfully or not.
        """
        try:
            if self.db:
                self.db.delete_memory(id=id)
            return "Memory deleted successfully"
        except Exception as e:
            logger.warning(f"Error deleting memory in db: {e}")
            return f"Error deleting memory: {e}"

    def update_memory(self, id: str, memory: str) -> str:
        """Use this function to update a memory in the database.
        Args:
            id (str): The id of the memory to be updated.
            memory (str): The updated memory.
        Returns:
            str: A message indicating if the memory was updated successfully or not.
        """
        try:
            if self.db:
                self.db.upsert_memory(
                    MemoryRow(
                        id=id,
                        user_id=self.user_id,
                        memory=Memory(memory=memory, input_text=self.input_message).to_dict()
                    )
                )
            return "Memory updated successfully"
        except Exception as e:
            logger.warning(f"Error updating memory in db: {e}")
            return f"Error updating memory: {e}"

    def clear_memory(self) -> str:
        """Use this function to clear all memories from the database.

        Returns:
            str: A message indicating if the memory was cleared successfully or not.
        """
        try:
            if self.db:
                self.db.clear()
            return "Memory cleared successfully"
        except Exception as e:
            logger.warning(f"Error clearing memory in db: {e}")
            return f"Error clearing memory: {e}"

    def get_system_prompt(self) -> Optional[str]:
        # If the system_prompt is provided, use it
        if self.system_prompt is not None:
            return self.system_prompt

        # -*- Build a default system prompt for classification
        system_prompt_lines = [
            "Your task is to generate a concise memory for the user's message. "
            "Create a memory that captures the key information provided by the user, as if you were storing it for future reference. "
            "The memory should be a brief, third-person statement that encapsulates the most important aspect of the user's input, without adding any extraneous details. "
            "This memory will be used to enhance the user's experience in subsequent conversations.",
            "You will also be provided with a list of existing memories. You may:",
            "  1. Add a new memory using the `add_memory` tool.",
            "  2. Update a memory using the `update_memory` tool.",
            "  3. Delete a memory using the `delete_memory` tool.",
            "  4. Clear all memories using the `clear_memory` tool. Use this with extreme caution, as it will remove all memories from the database.",
        ]
        existing_memories = self.get_existing_memories()
        if existing_memories and len(existing_memories) > 0:
            system_prompt_lines.extend(
                [
                    "\nExisting memories:",
                    "<existing_memories>\n"
                    + "\n".join([f"  - id: {m.id} | memory: {m.memory}" for m in existing_memories])
                    + "\n</existing_memories>",
                ]
            )
        return "\n".join(system_prompt_lines)

    def run(
            self,
            message: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        # Set input message added with the memory
        self.input_message = message
        if self.mode == "rule":
            exist_id = None
            existing_memories = self.get_existing_memories()
            if existing_memories and len(existing_memories) > 0:
                for m in existing_memories:
                    if message in m.memory:
                        exist_id = m.id
                        break
            if exist_id:
                self.update_memory(id=exist_id, memory=message)
                response = f"Memory updated successfully, id: {exist_id}, memory: {message}"
            else:
                self.add_memory(memory=message)
                response = f"Memory added successfully, memory: {message}"
            logger.debug(f"MemoryManager mode: {self.mode}, response: {response}")
        else:
            # Update the Model (set defaults, add logit etc.)
            self.update_model()

            # -*- Prepare the List of messages sent to the Model
            llm_messages: List[Message] = []

            # Create the system prompt message
            system_prompt_message = Message(role="system", content=self.get_system_prompt())
            llm_messages.append(system_prompt_message)
            # Create the user prompt message
            user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
            if user_prompt_message is not None:
                llm_messages += [user_prompt_message]

            # -*- Generate a response from the llm (includes running function calls)
            self.model = cast(Model, self.model)
            res = self.model.response(messages=llm_messages)
            response = res.content
        return response

    async def arun(self, message: Optional[str] = None, **kwargs: Any):
        # Set input message added with the memory
        self.input_message = message
        if self.mode == "rule":
            exist_id = None
            existing_memories = self.get_existing_memories()
            if existing_memories and len(existing_memories) > 0:
                for m in existing_memories:
                    if message in m.memory:
                        exist_id = m.id
                        break
            if exist_id:
                self.update_memory(id=exist_id, memory=message)
                response = f"Memory updated successfully, id: {exist_id}, memory: {message}"
            else:
                self.add_memory(memory=message)
                response = f"Memory added successfully, memory: {message}"
            logger.debug(f"MemoryManager mode: {self.mode}, response: {response}")
        else:
            self.update_model()
            llm_messages: List[Message] = []

            system_prompt_message = Message(role="system", content=self.get_system_prompt())
            llm_messages.append(system_prompt_message)
            # Create the user prompt message
            user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
            if user_prompt_message is not None:
                llm_messages += [user_prompt_message]
            self.model = cast(Model, self.model)
            res = await self.model.aresponse(messages=llm_messages)
            response = res.content
        return response


class MemoryClassifier(BaseModel):
    model: Optional[Model] = None

    # Provide the system prompt for the classifier as a string
    system_prompt: Optional[str] = None
    # Existing Memories
    existing_memories: Optional[List[Memory]] = None

    def update_model(self) -> None:
        if self.model is None:
            self.model = OpenAIChat()
        logger.debug(f"MemoryClassifier, using model: {self.model}")


    def get_system_prompt(self) -> Optional[str]:
        # If the system_prompt is provided, use it
        if self.system_prompt is not None:
            return self.system_prompt

        # -*- Build a default system prompt for classification
        system_prompt_lines = [
            "Your task is to identify if the user's message contains information that is worth remembering for future conversations.",
            "This includes details that could personalize ongoing interactions with the user, such as:\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - User shared code snippets\n"
            "  - Valuable questions asked by the user, and the reply answer.\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - Significant life events or experiences shared by the user\n"
            "  - Important context about the user's current situation, challenges or goals\n"
            "  - What the user likes or dislikes, their opinions, beliefs, values, etc.\n"
            "  - Any other details that provide valuable insights into the user's personality, perspective or needs",
            "Your task is to decide whether the user input contains any of the above information worth remembering.",
            "If the user input contains any information worth remembering for future conversations, respond with 'yes'.",
            "If the input does not contain any important details worth saving, respond with 'no' to disregard it.",
            "You will also be provided with a list of existing memories to help you decide if the input is new or already known.",
            "If the memory already exists that matches the input, respond with 'no' to keep it as is.",
            "If a memory exists that needs to be updated or deleted, respond with 'yes' to update/delete it.",
            "You must only respond with 'yes' or 'no'. Nothing else will be considered as a valid response.",
        ]
        if self.existing_memories and len(self.existing_memories) > 0:
            system_prompt_lines.extend(
                [
                    "\nExisting memories:",
                    "<existing_memories>\n"
                    + "\n".join([f"  - {m.memory}" for m in self.existing_memories])
                    + "\n</existing_memories>",
                ]
            )
        return "\n".join(system_prompt_lines)

    def run(
            self,
            message: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # -*- Prepare the List of messages sent to the Model
        llm_messages: List[Message] = []

        # Get the System prompt
        system_prompt = self.get_system_prompt()
        # Create system prompt message
        system_prompt_message = Message(role="system", content=system_prompt)
        # Add system prompt message to the messages list
        if system_prompt_message.content_is_valid():
            llm_messages.append(system_prompt_message)

        # Build the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            llm_messages += [user_prompt_message]

        # -*- generate_a_response_from_the_llm (includes_running_function_calls)
        self.model = cast(Model, self.model)
        classification_response = self.model.response(messages=llm_messages)
        return classification_response.content

    async def arun(self, message: Optional[str] = None, **kwargs: Any):
        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # -*- Prepare the List of messages sent to the Model
        llm_messages: List[Message] = []

        # Get the System prompt
        system_prompt = self.get_system_prompt()
        # Create system prompt message
        system_prompt_message = Message(role="system", content=system_prompt)
        # Add system prompt message to the messages list
        if system_prompt_message.content_is_valid():
            llm_messages.append(system_prompt_message)

        # Build the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            llm_messages += [user_prompt_message]

        # -*- generate_a_response_from_the_llm (includes_running_function_calls)
        self.model = cast(Model, self.model)
        classification_response = await self.model.aresponse(messages=llm_messages)
        return classification_response.content


class MemoryRetrieval(str, Enum):
    last_n = "last_n"
    first_n = "first_n"
    only_user = "only_user"


class MemorySummarizer(BaseModel):
    model: Optional[Model] = None
    use_structured_outputs: bool = False

    def update_model(self) -> None:
        if self.model is None:
            self.model = OpenAIChat()

        # Set response_format if it is not set on the Model
        if self.use_structured_outputs:
            self.model.response_format = SessionSummary
            self.model.structured_outputs = True
        else:
            self.model.response_format = {"type": "json_object"}

    def get_system_message(self, messages_for_summarization: List[Dict[str, str]]) -> Message:
        # -*- Return a system message for summarization
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
            system_prompt += "\n\nProvide your output as a JSON containing the following fields:"
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
                system_prompt += "\n<json_fields>"
                system_prompt += f"\n{json.dumps([key for key in response_model_properties.keys() if key != '$defs'])}"
                system_prompt += "\n</json_fields>"
                system_prompt += "\nHere are the properties for each field:"
                system_prompt += "\n<json_field_properties>"
                system_prompt += f"\n{json.dumps(response_model_properties, indent=2, ensure_ascii=False)}"
                system_prompt += "\n</json_field_properties>"

            system_prompt += "\nStart your response with `{` and end it with `}`."
            system_prompt += "\nYour output will be passed to json.loads() to convert it to a Python object."
            system_prompt += "\nMake sure it only contains valid JSON."
        return Message(role="system", content=system_prompt)

    def run(
            self,
            message_pairs: List[Tuple[Message, Message]],
            **kwargs: Any,
    ) -> Optional[SessionSummary]:
        # logger.debug("*********** MemorySummarizer Start ***********")
        if message_pairs is None or len(message_pairs) == 0:
            logger.info("No message pairs provided for summarization.")
            return None

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Convert the message pairs to a list of dictionaries
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append(
                {
                    user_message.role: user_message.get_content_string(),
                    assistant_message.role: assistant_message.get_content_string(),
                }
            )

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message(messages_for_summarization)]
        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = self.model.response(messages=messages_for_model)
        # logger.debug("*********** MemorySummarizer End ***********")

        # If the model natively supports structured outputs, the parsed value is already in the structured format
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed

        # Otherwise convert the response to the structured format
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    # Check if response starts with ```json
                    if response.content.startswith("```json"):
                        response.content = response.content.replace("```json\n", "").replace("\n```", "")
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            logger.warning(f"Failed to validate session_summary response: {exc}")
                return session_summary
            except Exception as e:
                logger.warning(f"Failed to convert response to session_summary: {e}")
        return None

    async def arun(
            self,
            message_pairs: List[Tuple[Message, Message]],
            **kwargs: Any,
    ) -> Optional[SessionSummary]:
        # logger.debug("*********** Async MemorySummarizer Start ***********")
        if message_pairs is None or len(message_pairs) == 0:
            logger.info("No message pairs provided for summarization.")
            return None

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Convert the message pairs to a list of dictionaries
        messages_for_summarization: List[Dict[str, str]] = []
        for message_pair in message_pairs:
            user_message, assistant_message = message_pair
            messages_for_summarization.append(
                {
                    user_message.role: user_message.get_content_string(),
                    assistant_message.role: assistant_message.get_content_string(),
                }
            )

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message(messages_for_summarization)]
        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        # logger.debug("*********** Async MemorySummarizer End ***********")

        # If the model natively supports structured outputs, the parsed value is already in the structured format
        if self.use_structured_outputs and response.parsed is not None and isinstance(response.parsed, SessionSummary):
            return response.parsed

        # Otherwise convert the response to the structured format
        if isinstance(response.content, str):
            try:
                session_summary = None
                try:
                    session_summary = SessionSummary.model_validate_json(response.content)
                except ValidationError:
                    # Check if response starts with ```json
                    if response.content.startswith("```json"):
                        response.content = response.content.replace("```json\n", "").replace("\n```", "")
                        try:
                            session_summary = SessionSummary.model_validate_json(response.content)
                        except ValidationError as exc:
                            logger.warning(f"Failed to validate session_summary response: {exc}")
                return session_summary
            except Exception as e:
                logger.warning(f"Failed to convert response to session_summary: {e}")
        return None


class AgentMemory(BaseModel):
    # Runs between the user and agent
    runs: List[AgentRun] = []
    # List of messages sent to the model
    messages: List[Message] = []
    update_system_message_on_change: bool = False

    # Create and store session summaries
    create_session_summary: bool = False
    # Update session summaries after each run, effect when create_session_summary is True
    update_session_summary_after_run: bool = True
    # Summary of the session
    summary: Optional[SessionSummary] = None
    # Summarizer to generate session summaries
    summarizer: Optional[MemorySummarizer] = None

    # Create and store personalized memories for this user
    create_user_memories: bool = False
    # Update memories for the user after each run, effect when create_user_memories is True
    update_user_memories_after_run: bool = True

    # MemoryDb to store personalized memories
    db: Optional[MemoryDb] = None
    # User ID for the personalized memories
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    num_memories: Optional[int] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None

    # True when memory is being updated, auto record for memory status
    updating_memory: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(
            exclude_none=True,
            exclude={
                "summary",
                "summarizer",
                "db",
                "updating_memory",
                "memories",
                "classifier",
                "manager",
                "retrieval",
            },
        )
        if self.summary:
            _memory_dict["summary"] = self.summary.to_dict()
        if self.memories:
            _memory_dict["memories"] = [memory.to_dict() for memory in self.memories]
        return _memory_dict

    def add_run(self, agent_run: AgentRun) -> None:
        """Adds an AgentRun to the runs list."""
        self.runs.append(agent_run)

    def add_system_message(self, message: Message, system_message_role: str = "system") -> None:
        """Add the system messages to the messages list"""
        # If this is the first run in the session, add the system message to the messages list
        if len(self.messages) == 0:
            if message is not None:
                self.messages.append(message)
        # If there are messages in the memory, check if the system message is already in the memory
        # If it is not, add the system message to the messages list
        # If it is, update the system message if content has changed and update_system_message_on_change is True
        else:
            system_message_index = next((i for i, m in enumerate(self.messages) if m.role == system_message_role), None)
            # Update the system message in memory if content has changed
            if system_message_index is not None:
                if (
                        self.messages[system_message_index].content != message.content
                        and self.update_system_message_on_change
                ):
                    logger.info("Updating system message in memory with new content")
                    self.messages[system_message_index] = message
            else:
                # Add the system message to the messages list
                self.messages.insert(0, message)

    def add_message(self, message: Message) -> None:
        """Add a Message to the messages list."""
        self.messages.append(message)
        # logger.debug("Added Message to AgentMemory")

    def add_messages(self, messages: List[Message]) -> None:
        """Add a list of messages to the messages list."""
        self.messages.extend(messages)
        # logger.debug(f"Added {len(messages)} Messages to AgentMemory")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the messages list as a list of dictionaries."""
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def get_messages_from_last_n_runs(
            self, last_n: Optional[int] = None, skip_role: Optional[str] = None
    ) -> List[Message]:
        """Returns the messages from the last_n runs

        Args:
            last_n: The number of runs to return from the end of the conversation.
            skip_role: Skip messages with this role.

        Returns:
            A list of Messages in the last_n runs.
        """
        if last_n is None:
            logger.debug("Getting messages from all previous runs")
            messages_from_all_history = []
            for prev_run in self.runs:
                if prev_run.response and prev_run.response.messages:
                    if skip_role:
                        prev_run_messages = [m for m in prev_run.response.messages if m.role != skip_role]
                    else:
                        prev_run_messages = prev_run.response.messages
                    messages_from_all_history.extend(prev_run_messages)
            logger.debug(f"Messages from previous runs: {len(messages_from_all_history)}")
            return messages_from_all_history

        logger.debug(f"Getting messages from last {last_n} runs")
        messages_from_last_n_history = []
        for prev_run in self.runs[-last_n:]:
            if prev_run.response and prev_run.response.messages:
                if skip_role:
                    prev_run_messages = [m for m in prev_run.response.messages if m.role != skip_role]
                else:
                    prev_run_messages = prev_run.response.messages
                messages_from_last_n_history.extend(prev_run_messages)
        logger.debug(f"Messages from last {last_n} runs: {len(messages_from_last_n_history)}")
        return messages_from_last_n_history

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

                # Start from the beginning to look for the user message
                for message in run.response.messages:
                    if message.role == user_role:
                        user_messages_from_run = message
                        break

                # Start from the end to look for the assistant response
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

    def load_user_memories(self) -> None:
        """Load memories from memory db for this user."""
        if self.db is None:
            return

        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="asc" if self.retrieval == MemoryRetrieval.first_n else "desc",
                )
            else:
                raise NotImplementedError("Semantic retrieval not yet supported.")
        except Exception as e:
            logger.debug(f"Error reading memory: {e}")
            return

        # Clear the existing memories
        self.memories = []

        # No memories to load
        if memory_rows is None or len(memory_rows) == 0:
            return

        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                logger.warning(f"Error loading memory: {e}")
                continue

    def should_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input)
        if classifier_response == "yes":
            return True
        return False

    async def ashould_update_memory(self, input: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = await self.classifier.arun(input)
        if classifier_response == "yes":
            return True
        return False

    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            logger.warning("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or self.should_update_memory(input=input)
        logger.debug(f"Update memory: {should_update_memory}")

        if not should_update_memory:
            logger.debug("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = self.manager.run(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    async def aupdate_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            logger.warning("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or await self.ashould_update_memory(input=input)
        logger.debug(f"Async update memory: {should_update_memory}")

        if not should_update_memory:
            logger.debug("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)
        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = await self.manager.arun(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    def update_summary(self) -> Optional[SessionSummary]:
        """Creates a summary of the session"""

        self.updating_memory = True

        if self.summarizer is None:
            self.summarizer = MemorySummarizer()

        self.summary = self.summarizer.run(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    async def aupdate_summary(self) -> Optional[SessionSummary]:
        """Creates a summary of the session"""

        self.updating_memory = True

        if self.summarizer is None:
            self.summarizer = MemorySummarizer()

        self.summary = await self.summarizer.arun(self.get_message_pairs())
        self.updating_memory = False
        return self.summary

    def clear(self) -> None:
        """Clear the AgentMemory"""

        self.runs = []
        self.messages = []
        self.summary = None
        self.memories = None

    def deep_copy(self):
        # Create a shallow copy of the object
        copied_obj = self.__class__(**self.model_dump())

        # Manually deepcopy fields that are known to be safe
        for field_name, field_value in self.__dict__.items():
            if field_name not in ["db", "classifier", "manager", "summarizer"]:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    logger.warning(f"Failed to deepcopy field: {field_name} - {e}")
                    setattr(copied_obj, field_name, field_value)

        copied_obj.db = self.db
        copied_obj.classifier = self.classifier
        copied_obj.manager = self.manager
        copied_obj.summarizer = self.summarizer

        return copied_obj


class WorkflowRun(BaseModel):
    input: Optional[Dict[str, Any]] = None
    response: Optional[RunResponse] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowMemory(BaseModel):
    runs: List[WorkflowRun] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def add_run(self, workflow_run: WorkflowRun) -> None:
        """Adds a WorkflowRun to the runs list."""
        self.runs.append(workflow_run)
        logger.debug("Added WorkflowRun to WorkflowMemory")

    def clear(self) -> None:
        """Clear the WorkflowMemory"""

        self.runs = []

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> "WorkflowMemory":
        new_memory = self.model_copy(deep=True, update=update)
        # clear the new memory to remove any references to the old memory
        new_memory.clear()
        return new_memory
