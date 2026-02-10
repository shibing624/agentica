# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory manager and classifier
"""

import json
from typing import Dict, List, Any, Optional, cast, Literal
from copy import deepcopy
from pydantic import BaseModel, ConfigDict

from agentica.db.base import BaseDb, MemoryRow
from agentica.model.openai import OpenAIChat
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.memory.models import Memory


class MemoryManager(BaseModel):
    """
    MemoryManager class to manage the memories for the user,
        including adding, updating, deleting, clearing, and searching memories.
    
    Features:
        - CRUD operations for memories
        - Intelligent memory search with multiple retrieval methods:
            - last_n: Return the most recent memories
            - first_n: Return the oldest memories
            - keyword: Search memories by keyword matching
            - agentic: Use Agent for semantic similarity search
    
    Example:
        >>> from agentica.memory import MemoryManager
        >>> from agentica.db.sqlite import SqliteDb
        >>> 
        >>> manager = MemoryManager(
        ...     model=OpenAIChat(id="gpt-4o"),
        ...     db=SqliteDb(db_file="agent.db"),
        ... )
        >>> 
        >>> # Add memory
        >>> manager.add_memory("User likes Python programming")
        >>> 
        >>> # Search memories
        >>> memories = manager.search_user_memories(
        ...     query="programming language",
        ...     retrieval_method="agentic",
        ...     user_id="user123"
        ... )
    """

    mode: Literal["model", "rule"] = "rule"
    model: Optional[Model] = None
    user_id: Optional[str] = None

    # Provide the system prompt for the manager as a string
    system_prompt: Optional[str] = None
    # Memory Database
    db: Optional[BaseDb] = None

    # Memory operation switches
    delete_memories: bool = True
    clear_memories: bool = True
    update_memories: bool = True
    add_memories: bool = True

    # Do not set the input message here, it will be set by the run method
    input_message: Optional[str] = None

    # Track whether tools have been added to the model
    _tools_registered: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _ensure_model(self) -> Model:
        """Ensure model is initialized, return it."""
        if self.model is None:
            self.model = OpenAIChat()
            logger.debug(f"Using Model: {self.model}")
        return self.model

    def update_model(self) -> None:
        self._ensure_model()
        if not self._tools_registered:
            if self.add_memories:
                self.model.add_tool(self.add_memory)
            if self.update_memories:
                self.model.add_tool(self.update_memory)
            if self.delete_memories:
                self.model.add_tool(self.delete_memory)
            if self.clear_memories:
                self.model.add_tool(self.clear_memory)
            self._tools_registered = True

    def get_existing_memories(self) -> Optional[List[MemoryRow]]:
        if self.db is None:
            return None

        return self.db.read_memories(user_id=self.user_id)

    def get_user_memories(self, user_id: Optional[str] = None) -> List[MemoryRow]:
        """Get the user memories for a given user id."""
        if self.db is None:
            logger.warning("Memory DB not provided.")
            return []
        
        _user_id = user_id or self.user_id
        memories = self.db.read_memories(user_id=_user_id)
        return memories if memories else []

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

    def add_user_memory(self, memory: Memory, user_id: Optional[str] = None) -> str:
        """Add a user memory for a given user id."""
        try:
            if self.db:
                _user_id = user_id or self.user_id
                row = MemoryRow(
                    user_id=_user_id,
                    memory=memory.to_dict()
                )
                self.db.upsert_memory(row)
                return row.id or "Memory added successfully"
            return "Database not provided"
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
                self.db.delete_memory(memory_id=id)
            return "Memory deleted successfully"
        except Exception as e:
            logger.warning(f"Error deleting memory in db: {e}")
            return f"Error deleting memory: {e}"

    def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> str:
        """Delete a user memory for a given user id."""
        return self.delete_memory(id=memory_id)

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

    def replace_user_memory(
        self,
        memory_id: str,
        memory: Memory,
        user_id: Optional[str] = None
    ) -> str:
        """Replace a user memory for a given user id."""
        try:
            if self.db:
                _user_id = user_id or self.user_id
                row = MemoryRow(
                    id=memory_id,
                    user_id=_user_id,
                    memory=memory.to_dict()
                )
                self.db.upsert_memory(row)
                return memory_id
            return "Database not provided"
        except Exception as e:
            logger.warning(f"Error replacing memory in db: {e}")
            return f"Error replacing memory: {e}"

    def clear_memory(self) -> str:
        """Use this function to clear all memories from the database.

        Returns:
            str: A message indicating if the memory was cleared successfully or not.
        """
        try:
            if self.db:
                self.db.clear_memories(user_id=self.user_id)
            return "Memory cleared successfully"
        except Exception as e:
            logger.warning(f"Error clearing memory in db: {e}")
            return f"Error clearing memory: {e}"

    def clear_user_memories(self, user_id: Optional[str] = None) -> str:
        """Clear all memories for a specific user."""
        try:
            if self.db:
                _user_id = user_id or self.user_id
                self.db.clear_memories(user_id=_user_id)
            return "Memories cleared successfully"
        except Exception as e:
            logger.warning(f"Error clearing memories in db: {e}")
            return f"Error clearing memories: {e}"

    # ==================== Memory Search Methods ====================

    def search_user_memories(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        retrieval_method: Literal["last_n", "first_n", "keyword", "agentic"] = "last_n",
        user_id: Optional[str] = None,
    ) -> List[MemoryRow]:
        """Search through user memories using the specified retrieval method."""
        _user_id = user_id or self.user_id

        if retrieval_method == "agentic":
            if not query:
                raise ValueError("Query is required for agentic search")
            return self._search_memories_agentic(user_id=_user_id, query=query, limit=limit)

        elif retrieval_method == "keyword":
            if not query:
                raise ValueError("Query is required for keyword search")
            return self._search_memories_keyword(user_id=_user_id, query=query, limit=limit)

        elif retrieval_method == "first_n":
            return self._get_first_n_memories(user_id=_user_id, limit=limit)

        else:  # Default to last_n
            return self._get_last_n_memories(user_id=_user_id, limit=limit)

    async def asearch_user_memories(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        retrieval_method: Literal["last_n", "first_n", "keyword", "agentic"] = "last_n",
        user_id: Optional[str] = None,
    ) -> List[MemoryRow]:
        """Async version of search_user_memories."""
        _user_id = user_id or self.user_id

        if retrieval_method == "agentic":
            if not query:
                raise ValueError("Query is required for agentic search")
            return await self._asearch_memories_agentic(user_id=_user_id, query=query, limit=limit)

        elif retrieval_method == "keyword":
            if not query:
                raise ValueError("Query is required for keyword search")
            return self._search_memories_keyword(user_id=_user_id, query=query, limit=limit)

        elif retrieval_method == "first_n":
            return self._get_first_n_memories(user_id=_user_id, limit=limit)

        else:  # Default to last_n
            return self._get_last_n_memories(user_id=_user_id, limit=limit)

    def _get_last_n_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """Get the most recent user memories."""
        if self.db is None:
            return []
        memories = self.db.read_memories(user_id=user_id, limit=limit, sort="desc")
        return memories if memories else []

    def _get_first_n_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """Get the oldest user memories."""
        if self.db is None:
            return []
        memories = self.db.read_memories(user_id=user_id, limit=limit, sort="asc")
        return memories if memories else []

    def _search_memories_keyword(
        self,
        user_id: Optional[str] = None,
        query: str = "",
        limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """Search memories by keyword matching."""
        if self.db is None:
            return []

        all_memories = self.db.read_memories(user_id=user_id)
        if not all_memories:
            return []

        keywords = query.lower().split()
        
        matched_memories = []
        for memory in all_memories:
            memory_text = json.dumps(memory.memory, ensure_ascii=False).lower()
            if any(keyword in memory_text for keyword in keywords):
                matched_memories.append(memory)

        if limit and limit > 0:
            matched_memories = matched_memories[:limit]

        return matched_memories

    def _build_agentic_search_messages(
        self,
        all_memories: List[MemoryRow],
        query: str,
    ) -> List[Message]:
        """Build messages for agentic memory search (shared between sync/async)."""
        system_message_lines = [
            "Your task is to search through user memories and return the IDs of the memories that are related to the query.",
            "",
            "<user_memories>"
        ]
        for memory in all_memories:
            memory_content = memory.memory.get("memory", "") if isinstance(memory.memory, dict) else str(memory.memory)
            system_message_lines.append(f"ID: {memory.id}")
            system_message_lines.append(f"Memory: {memory_content}")
            system_message_lines.append("")
        system_message_lines.append("</user_memories>")
        system_message_lines.append("")
        system_message_lines.append("IMPORTANT: Only return the IDs of the memories that are semantically related to the query.")
        system_message_lines.append("Return your response as a JSON object with a 'memory_ids' field containing a list of memory IDs.")
        system_message_lines.append('Example: {"memory_ids": ["id1", "id2"]}')

        return [
            Message(role="system", content="\n".join(system_message_lines)),
            Message(
                role="user",
                content=f"Return the IDs of the memories related to the following query: {query}",
            ),
        ]

    def _parse_agentic_response(self, response_content: Optional[str]) -> List[str]:
        """Parse memory IDs from agentic search response."""
        if not response_content:
            return []
        try:
            result = json.loads(response_content)
            return result.get("memory_ids", [])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse agentic search response: {response_content}")
            return []

    def _filter_memories_by_ids(
        self,
        all_memories: List[MemoryRow],
        memory_ids: List[str],
        limit: Optional[int] = None,
    ) -> List[MemoryRow]:
        """Filter memories by IDs and apply limit."""
        matched = [m for m in all_memories if m.id in memory_ids]
        if limit and limit > 0:
            matched = matched[:limit]
        return matched

    def _search_memories_agentic(
        self,
        user_id: Optional[str] = None,
        query: str = "",
        limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """Search through user memories using Agent for semantic similarity."""
        if self.db is None:
            return []

        all_memories = self.db.read_memories(user_id=user_id)
        if not all_memories:
            return []

        self._ensure_model()
        logger.debug("Searching for memories using agentic method")

        messages_for_model = self._build_agentic_search_messages(all_memories, query)

        model_copy = deepcopy(self.model)
        model_copy.response_format = {"type": "json_object"}

        response = model_copy.response(messages=messages_for_model)
        logger.debug("Agentic memory search complete")

        memory_ids = self._parse_agentic_response(response.content)
        return self._filter_memories_by_ids(all_memories, memory_ids, limit)

    async def _asearch_memories_agentic(
        self,
        user_id: Optional[str] = None,
        query: str = "",
        limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """Async version of _search_memories_agentic."""
        if self.db is None:
            return []

        all_memories = self.db.read_memories(user_id=user_id)
        if not all_memories:
            return []

        self._ensure_model()
        logger.debug("Searching for memories using agentic method (async)")

        messages_for_model = self._build_agentic_search_messages(all_memories, query)

        model_copy = deepcopy(self.model)
        model_copy.response_format = {"type": "json_object"}

        response = await model_copy.aresponse(messages=messages_for_model)
        logger.debug("Agentic memory search complete (async)")

        memory_ids = self._parse_agentic_response(response.content)
        return self._filter_memories_by_ids(all_memories, memory_ids, limit)

    # ==================== System Prompt and Run Methods ====================

    def get_system_prompt(self) -> Optional[str]:
        if self.system_prompt is not None:
            return self.system_prompt

        system_prompt_lines = [
            "Your task is to generate a concise memory for the user's message. "
            "Create a memory that captures the key information provided by the user, as if you were storing it for future reference. "
            "The memory should be a brief, third-person statement that encapsulates the most important aspect of the user's input, without adding any extraneous details. "
            "This memory will be used to enhance the user's experience in subsequent conversations.",
            "You will also be provided with a list of existing memories. You may:",
        ]
        if self.add_memories:
            system_prompt_lines.append("  - Add a new memory using the `add_memory` tool.")
        if self.update_memories:
            system_prompt_lines.append("  - Update a memory using the `update_memory` tool.")
        if self.delete_memories:
            system_prompt_lines.append("  - Delete a memory using the `delete_memory` tool.")
        if self.clear_memories:
            system_prompt_lines.append("  - Clear all memories using the `clear_memory` tool. Use this with extreme caution.")

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

    def _run_rule_mode(self, message: Optional[str]) -> str:
        """Run memory update in rule mode (shared logic for sync/async)."""
        exist_id = None
        existing_memories = self.get_existing_memories()
        if existing_memories and len(existing_memories) > 0:
            for m in existing_memories:
                memory_text = str(m.memory)
                # Use exact equality check instead of substring match to avoid false positives
                if message and message == memory_text:
                    exist_id = m.id
                    break
        if exist_id:
            self.update_memory(id=exist_id, memory=message)
            response = f"Memory updated successfully, id: {exist_id}, memory: {message}"
        else:
            self.add_memory(memory=message)
            response = f"Memory added successfully, memory: {message}"
        logger.debug(f"MemoryManager mode: {self.mode}, response: {response}")
        return response

    def _build_llm_messages(self, message: Optional[str], **kwargs: Any) -> List[Message]:
        """Build messages for LLM model mode (shared logic for sync/async)."""
        llm_messages: List[Message] = []
        system_prompt_message = Message(role="system", content=self.get_system_prompt())
        llm_messages.append(system_prompt_message)
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            llm_messages.append(user_prompt_message)
        return llm_messages

    def run(
            self,
            message: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        self.input_message = message
        if self.mode == "rule":
            return self._run_rule_mode(message)
        else:
            self.update_model()
            llm_messages = self._build_llm_messages(message, **kwargs)
            self.model = cast(Model, self.model)
            res = self.model.response(messages=llm_messages)
            return res.content

    async def arun(self, message: Optional[str] = None, **kwargs: Any):
        self.input_message = message
        if self.mode == "rule":
            return self._run_rule_mode(message)
        else:
            self.update_model()
            llm_messages = self._build_llm_messages(message, **kwargs)
            self.model = cast(Model, self.model)
            res = await self.model.aresponse(messages=llm_messages)
            return res.content


class MemoryClassifier(BaseModel):
    model: Optional[Model] = None

    # Provide the system prompt for the classifier as a string
    system_prompt: Optional[str] = None
    # Existing Memories
    existing_memories: Optional[List[Memory]] = None

    def _ensure_model(self) -> Model:
        """Ensure model is initialized, return it."""
        if self.model is None:
            self.model = OpenAIChat()
        logger.debug(f"MemoryClassifier, using model: {self.model}")
        return self.model

    def get_system_prompt(self) -> Optional[str]:
        if self.system_prompt is not None:
            return self.system_prompt

        system_prompt_lines = [
            "Your task is to identify if the user's message contains information that is worth remembering for future conversations.",
            "This includes details that could personalize ongoing interactions with the user, such as:\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - User shared code snippets\n"
            "  - Valuable questions asked by the user, and the reply answer.\n"
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

    def _build_messages(self, message: Optional[str], **kwargs: Any) -> List[Message]:
        """Build messages for classification (shared logic for sync/async)."""
        llm_messages: List[Message] = []
        system_prompt = self.get_system_prompt()
        system_prompt_message = Message(role="system", content=system_prompt)
        if system_prompt_message.content_is_valid():
            llm_messages.append(system_prompt_message)
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            llm_messages.append(user_prompt_message)
        return llm_messages

    def run(
            self,
            message: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        self._ensure_model()
        llm_messages = self._build_messages(message, **kwargs)
        self.model = cast(Model, self.model)
        classification_response = self.model.response(messages=llm_messages)
        return classification_response.content

    async def arun(self, message: Optional[str] = None, **kwargs: Any):
        self._ensure_model()
        llm_messages = self._build_messages(message, **kwargs)
        self.model = cast(Model, self.model)
        classification_response = await self.model.aresponse(messages=llm_messages)
        return classification_response.content
