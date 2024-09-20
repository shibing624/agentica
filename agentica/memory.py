# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from hashlib import md5
import uuid
from typing import Dict, List, Any, Optional, cast, Tuple, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from agentica.config import AGENTICA_HOME
from agentica.llm.base import LLM
from agentica.llm.openai_llm import OpenAILLM
from agentica.message import Message
from agentica.references import References
from agentica.utils.log import logger


class Memory(BaseModel):
    """Model for LLM memories"""

    memory: str
    id: Optional[str] = None
    topic: Optional[str] = None
    input_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_str(self) -> str:
        attributes = self.model_dump(exclude_none=True)
        attributes_str = ";".join(f"{key}: {value}" for key, value in attributes.items())
        return attributes_str


class MemoryRow(BaseModel):
    """Memory Row that is stored in the database"""

    memory: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # id for this memory, auto-generated from the memory
    id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def serializable_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_dict(self) -> Dict[str, Any]:
        return self.serializable_dict()

    @model_validator(mode="after")
    def generate_id(self) -> "MemoryRow":
        if self.id is None:
            memory_str = self.memory
            cleaned_memory = memory_str.replace(" ", "").replace("\n", "").replace("\t", "")
            self.id = md5(cleaned_memory.encode()).hexdigest()
        return self

    @model_validator(mode="after")
    def generate_timestamps(self) -> "MemoryRow":
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        return self


class MemoryDb(ABC):
    """Base class for the Memory Database."""

    @abstractmethod
    def create_table(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_table(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def table_exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear_table(self) -> bool:
        raise NotImplementedError


class CsvMemoryDb(MemoryDb):
    def __init__(self, file_path: str = None):
        self.file_path = file_path if file_path else os.path.join(AGENTICA_HOME, f"memory_{uuid.uuid4()}.csv")
        self.memories = []

    def create_table(self) -> None:
        # In the context of a CSV file, creating a table means creating a new file
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
        self.memories = []

    def memory_exists(self, memory: MemoryRow) -> bool:
        with open(self.file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == memory.id:
                    return True
        return False

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories = []
        if not os.path.exists(self.file_path):
            return memories
        with open(self.file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                if user_id is not None and row[1] != user_id:
                    continue
                memory = MemoryRow(
                    id=row[0],
                    user_id=row[1],
                    memory=row[2],
                    created_at=row[3],
                    updated_at=row[4]
                )
                memories.append(memory)
                if limit is not None and len(memories) >= limit:
                    break
        self.memories = memories
        return memories

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        memories = self.read_memories()
        if memories:
            for i, m in enumerate(memories):
                if m and m.id == memory.id:
                    memories[i] = memory
                    break
        else:
            memories.append(memory)
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
            for memory in memories:
                writer.writerow([
                    memory.id, memory.user_id, memory.memory, memory.created_at, memory.updated_at
                ])
            logger.debug(f"Memory {memory.id} upserted, memories size: {len(memories)}, saved: {self.file_path}")
        self.memories = memories
        return memory

    def delete_memory(self, id: str) -> None:
        memories = self.read_memories()
        memories = [m for m in memories if m.id != id]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
            for memory in memories:
                writer.writerow([
                    memory.id, memory.user_id, memory.memory, memory.created_at, memory.updated_at
                ])
            logger.debug(f"Memory {id} deleted, memories size: {len(memories)}, saved: {self.file_path}")
        self.memories = memories

    def delete_table(self) -> None:
        # In the context of a CSV file, deleting a table means deleting the file
        os.remove(self.file_path)
        self.memories = []

    def table_exists(self) -> bool:
        return os.path.exists(self.file_path)

    def clear_table(self) -> bool:
        # In the context of a CSV file, clearing a table means deleting all rows except the header
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
        self.memories = []
        return True


class InMemoryDb(MemoryDb):
    def __init__(self):
        self.memories = []

    def create_table(self) -> None:
        # Memory initialization
        self.memories = []

    def memory_exists(self, memory: MemoryRow) -> bool:
        for m in self.memories:
            if m.id == memory.id:
                return True
        return False

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        results = []
        for memory in self.memories:
            if user_id and memory.user_id == user_id:
                results.append(memory)

        # Sort results if needed
        if sort == "asc":
            results = sorted(results, key=lambda x: x.created_at)
        elif sort == "desc":
            results = sorted(results, key=lambda x: x.created_at, reverse=True)

        if limit:
            results = results[:limit]

        return results

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        if self.memory_exists(memory):
            self.delete_memory(memory.id)
        self.memories.append(memory)
        return memory

    def delete_memory(self, id: str) -> None:
        for i, memory in enumerate(self.memories):
            if memory.id == id:
                del self.memories[i]
                break

    def delete_table(self) -> None:
        self.create_table()

    def table_exists(self) -> bool:
        return True

    def clear_table(self) -> bool:
        self.create_table()
        return True


class MemoryManager(BaseModel):
    """
    MemoryManager class to manage the memories for the user,
        including adding, updating, deleting, and clearing memories.
    """

    mode: Literal["llm", "rule"] = "rule"
    llm: Optional[LLM] = None
    user_id: Optional[str] = None

    # Provide the system prompt for the manager as a string
    system_prompt: Optional[str] = None
    # Memory Database
    db: Optional[MemoryDb] = None

    # Do not set the input message here, it will be set by the run method
    input_message: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAILLM()
            logger.debug(f"Using LLM: {self.llm}")
        self.llm.add_tool(self.add_memory)
        self.llm.add_tool(self.update_memory)
        self.llm.add_tool(self.delete_memory)
        self.llm.add_tool(self.clear_memory)

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
                        memory=Memory(memory=memory, input_text=self.input_message).to_str()
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
                        memory=Memory(memory=memory, input_text=self.input_message).to_str()
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
                self.db.clear_table()
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
        logger.debug("*********** MemoryManager Start ***********")

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
            # Update the LLM (set defaults, add logit etc.)
            self.update_llm()

            # -*- Prepare the List of messages sent to the LLM
            llm_messages: List[Message] = []

            # Create the system prompt message
            system_prompt_message = Message(role="system", content=self.get_system_prompt())
            llm_messages.append(system_prompt_message)
            # Set input message added with the memory
            self.input_message = message
            # Create the user prompt message
            user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
            if user_prompt_message is not None:
                llm_messages += [user_prompt_message]

            # -*- Generate a response from the llm (includes running function calls)
            self.llm = cast(LLM, self.llm)
            response = self.llm.response(messages=llm_messages)
        logger.debug("*********** MemoryManager End ***********")
        return response


class MemoryClassifier(BaseModel):
    llm: Optional[LLM] = None

    # Provide the system prompt for the classifier as a string
    system_prompt: Optional[str] = None
    # Existing Memories
    existing_memories: Optional[List[Memory]] = None

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAILLM()
            logger.debug(f"Using LLM: {self.llm}")

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
        logger.debug("*********** MemoryClassifier Start ***********")

        # Update the LLM (set defaults, add logit etc.)
        self.update_llm()

        # -*- Prepare the List of messages sent to the LLM
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
        self.llm = cast(LLM, self.llm)
        classification_response = self.llm.response(messages=llm_messages)
        logger.debug("*********** MemoryClassifier End ***********")
        return classification_response


class MemoryRetrieval(str, Enum):
    last_n = "last_n"
    first_n = "first_n"
    only_user = "only_user"


class AssistantMemory(BaseModel):
    # Messages between the user and the Assistant.
    # Note: the llm prompts are stored in the llm_messages
    chat_history: List[Message] = []
    # Prompts sent to the LLM and the LLM responses.
    llm_messages: List[Message] = []
    # References from the vector database.
    references: List[References] = []

    # Create personalized memories for this user
    db: Optional[MemoryDb] = None
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    num_memories: Optional[int] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None
    updating: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(
            exclude_none=True, exclude={"db", "updating", "memories", "classifier", "manager"}
        )
        if self.memories:
            _memory_dict["memories"] = [memory.to_dict() for memory in self.memories]
        return _memory_dict

    def add_chat_message(self, message: Message) -> None:
        """Adds a Message to the chat_history."""
        self.chat_history.append(message)

    def add_llm_message(self, message: Message) -> None:
        """Adds a Message to the llm_messages."""
        self.llm_messages.append(message)

    def add_chat_messages(self, messages: List[Message]) -> None:
        """Adds a list of messages to the chat_history."""
        self.chat_history.extend(messages)

    def add_llm_messages(self, messages: List[Message]) -> None:
        """Adds a list of messages to the llm_messages."""
        self.llm_messages.extend(messages)

    def add_references(self, references: References) -> None:
        """Adds references to the references list."""
        self.references.append(references)

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Returns the chat_history as a list of dictionaries.

        :return: A list of dictionaries representing the chat_history.
        """
        return [message.model_dump(exclude_none=True) for message in self.chat_history]

    def get_last_n_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Returns the last n messages in the chat_history.

        :param last_n: The number of messages to return from the end of the conversation.
            If None, returns all messages.
        :return: A list of Messages in the chat_history.
        """
        return self.chat_history[-last_n:] if last_n else self.chat_history

    def get_llm_messages(self) -> List[Dict[str, Any]]:
        """Returns the llm_messages as a list of dictionaries."""
        return [message.model_dump(exclude_none=True) for message in self.llm_messages]

    def get_formatted_chat_history(self, num_messages: Optional[int] = None) -> str:
        """Returns the chat_history as a formatted string."""

        messages = self.get_last_n_messages(num_messages)
        if len(messages) == 0:
            return ""

        history = ""
        for message in self.get_last_n_messages(num_messages):
            if message.role == "user":
                history += "\n---\n"
            history += f"{message.role.upper()}: {message.content}\n"
        return history

    def get_chats(self) -> List[Tuple[Message, Message]]:
        """Returns a list of tuples of user messages and LLM responses."""

        all_chats: List[Tuple[Message, Message]] = []
        current_chat: List[Message] = []

        # Make a copy of the chat_history and remove all system messages from the beginning.
        chat_history = self.chat_history.copy()
        while len(chat_history) > 0 and chat_history[0].role in ("system", "assistant"):
            chat_history = chat_history[1:]

        for m in chat_history:
            if m.role == "system":
                continue
            if m.role == "user":
                # This is a new chat record
                if len(current_chat) == 2:
                    all_chats.append((current_chat[0], current_chat[1]))
                    current_chat = []
                current_chat.append(m)
            if m.role == "assistant":
                current_chat.append(m)

        if len(current_chat) >= 1:
            all_chats.append((current_chat[0], current_chat[1]))
        return all_chats

    def get_tool_calls(self, num_calls: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns a list of tool calls from the llm_messages.

        Example:
            - To get the last tool call, use num_calls=1.
            - To get all tool calls, use num_calls=-1.
        """

        tool_calls = []
        for llm_message in self.llm_messages[::-1]:
            if llm_message.tool_calls:
                for tool_call in llm_message.tool_calls:
                    tool_calls.append(tool_call)

        if num_calls and num_calls > 0:
            return tool_calls[:num_calls]
        return tool_calls

    def load_memory(self) -> None:
        """Load the memory from memory db for this user."""
        if self.db is None:
            return

        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="asc" if self.retrieval == MemoryRetrieval.first_n else "desc",
                )
            elif self.retrieval == MemoryRetrieval.only_user:
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="desc",
                )
                memory_rows = [row for row in memory_rows if row.user_id == self.user_id]
            else:
                raise NotImplementedError(f"{self.retrieval} retrieval method is not supported.")
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
                id = getattr(row, 'id', None)
                topic = getattr(row, 'topic', None)
                input_text = getattr(row, 'input_text', None)
                self.memories.append(Memory(memory=row.memory, id=id, topic=topic, input_text=input_text))
            except Exception as e:
                logger.warning(f"Error loading memory: {e}")
                continue

    def should_update_memory(self, input_text: str) -> bool:
        """Determines if a message should be added to the memory db."""

        if self.classifier is None:
            self.classifier = MemoryClassifier()

        self.classifier.existing_memories = self.memories
        classifier_response = self.classifier.run(input_text)
        classifier_response = classifier_response.strip().lower()
        if classifier_response == "yes" or classifier_response == "是":
            return True
        return False

    def update_memory(self, input_text: str, force: bool = False) -> str:
        """Creates a memory from a message and adds it to the memory db."""

        if input_text is None or not isinstance(input_text, str):
            return "Invalid message content"
        if self.db is None:
            logger.warning("MemoryDb not provided.")
            return "Please provide a db to store memories"
        self.updating = True
        # Check if this user message should be added to long term memory
        should_update_memory = force or self.should_update_memory(input_text=input_text)
        logger.debug(f"Update memory: {should_update_memory}")

        if not should_update_memory:
            logger.debug("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        response = self.manager.run(input_text)
        self.load_memory()

        return response

    def get_memories_for_system_prompt(self) -> Optional[str]:
        if self.memories is None or len(self.memories) == 0:
            return None
        memory_str = "<memory_from_previous_interactions>\n"
        memory_str += "\n".join([f"- {memory.memory}" for memory in self.memories])
        memory_str += "\n</memory_from_previous_interactions>"

        return memory_str
