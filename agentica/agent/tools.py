# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Default tools for Agent

This module contains the default tool implementations that can be
enabled on the Agent (chat history, knowledge base, memory, etc.)
"""

import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from agentica.utils.log import logger
from agentica.utils.timer import Timer
from agentica.document import Document
from agentica.model.message import MessageReferences
from agentica.run_response import RunResponseExtraData
from agentica.tools.base import Function


class ToolsMixin:
    """Mixin class containing default tool implementations for Agent."""

    def get_chat_history(self, num_chats: Optional[int] = None) -> str:
        """Use this function to get the chat history between the user and agent.

        Args:
            num_chats: The number of chats to return.
                Each chat contains 2 messages. One from the user and one from the agent.
                Default: None, means get all chats.

        Returns:
            str: A JSON of a list of dictionaries representing the chat history.

        Example:
            - To get the last chat, use num_chats=1.
            - To get the last 5 chats, use num_chats=5.
            - To get all chats, use num_chats=None.
            - To get the first chat, use num_chats=None and pick the first message.
        """
        history: List[Dict[str, Any]] = []
        all_chats = self.memory.get_message_pairs()
        if len(all_chats) == 0:
            return ""

        chats_added = 0
        for chat in all_chats[::-1]:
            history.insert(0, chat[1].to_dict())
            history.insert(0, chat[0].to_dict())
            chats_added += 1
            if num_chats is not None and chats_added >= num_chats:
                break
        return json.dumps(history, ensure_ascii=False)

    def get_tool_call_history(self, num_calls: int = 3) -> str:
        """Use this function to get the tools called by the agent in reverse chronological order.

        Args:
            num_calls: The number of tool calls to return.
                Default: 3

        Returns:
            str: A JSON of a list of dictionaries representing the tool call history.

        Example:
            - To get the last tool call, use num_calls=1.
            - To get all tool calls, use num_calls=None.
        """
        tool_calls = self.memory.get_tool_calls(num_calls)
        if len(tool_calls) == 0:
            return ""
        logger.debug(f"tool_calls: {tool_calls}")
        return json.dumps(tool_calls, ensure_ascii=False)

    def search_knowledge_base(self, query: str) -> str:
        """Use this function to search the knowledge base for information about a query.

        Args:
            query: The query to search for.

        Returns:
            str: A string containing the response from the knowledge base.
        """
        # Get the relevant documents from the knowledge base
        retrieval_timer = Timer()
        retrieval_timer.start()
        docs_from_knowledge = self.get_relevant_docs_from_knowledge(query=query)
        if docs_from_knowledge is not None:
            references = MessageReferences(
                query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4)
            )
            # Add the references to the run_response
            if self.run_response.extra_data is None:
                self.run_response.extra_data = RunResponseExtraData()
            if self.run_response.extra_data.references is None:
                self.run_response.extra_data.references = []
            self.run_response.extra_data.references.append(references)
        retrieval_timer.stop()
        logger.debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")

        if docs_from_knowledge is None:
            return "No documents found"
        return self.convert_documents_to_string(docs_from_knowledge)

    def add_to_knowledge(self, query: str, result: str) -> str:
        """Use this function to add information to the knowledge base for future use.

        Args:
            query: The query to add.
            result: The result of the query.

        Returns:
            str: A string indicating the status of the addition.
        """
        if self.knowledge is None:
            return "Knowledge base not available"
        document_name = self.name
        if document_name is None:
            document_name = query.replace(" ", "_").replace("?", "").replace("!", "").replace(".", "")
        document_content = json.dumps({"query": query, "result": result}, ensure_ascii=False)
        logger.info(f"Adding document to knowledge base: {document_name}: {document_content}")
        self.knowledge.load_document(
            document=Document(
                name=document_name,
                content=document_content,
            )
        )
        return "Successfully added to knowledge base"

    async def update_memory(self, task: str) -> str:
        """Use this function to update the Agent's memory. Describe the task in detail.

        Args:
            task: The task to update the memory with.

        Returns:
            str: A string indicating the status of the task.
        """
        try:
            return await self.memory.update_memory(input=task, force=True) or "Memory updated successfully"
        except Exception as e:
            return f"Failed to update memory: {e}"

    def _create_run_data(self) -> Dict[str, Any]:
        """Create and return the run data dictionary."""
        run_response_format = "text"
        if self.response_model is not None:
            run_response_format = "json"
        elif self.markdown:
            run_response_format = "markdown"

        functions = {}
        if self.model is not None and self.model.functions is not None:
            functions = {
                f_name: func.to_dict() for f_name, func in self.model.functions.items() if isinstance(func, Function)
            }

        run_data: Dict[str, Any] = {
            "functions": functions,
            "metrics": self.run_response.metrics if self.run_response is not None else None,
        }

        if self.monitoring:
            run_data.update(
                {
                    "run_input": self.run_input,
                    "run_response": self.run_response.to_dict(),
                    "run_response_format": run_response_format,
                }
            )

        return run_data
