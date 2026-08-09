# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Default tools for Agent.

This module contains the default tool implementations that can be
enabled on the Agent (knowledge base search, memory, etc.)
"""

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from agentica.utils.log import logger
from agentica.utils.timer import Timer
from agentica.document import Document
from agentica.model.message import MessageReferences
from agentica.run_response import RunResponseExtraData
from agentica.tools.base import Function, ModelTool, Tool


class ToolsMixin:
    """Mixin class containing default tool implementations for Agent."""

    def get_tools(self) -> Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]]:
        """Get all tools available to this agent.

        Includes user-provided tools plus knowledge-base tools when configured.
        Multi-agent composition is handled separately by ``Agent.as_tool()`` and
        the explicit ``Swarm`` / ``BuiltinTaskTool`` runtimes; this method does
        not inject any implicit delegation tools.
        """
        tools: List[Union[ModelTool, Tool, Callable, Dict, Function]] = []

        if self.tools is not None:
            tools.extend(self.tools)

        if self.knowledge is not None:
            if self.tool_config.search_knowledge:
                tools.append(self.search_knowledge_base)
            if self.tool_config.update_knowledge:
                tools.append(self.add_to_knowledge)

        return tools if len(tools) > 0 else []

    async def search_knowledge_base(self, query: str) -> str:
        """Use this function to search the knowledge base for information about a query.

        Args:
            query: The query to search for.

        Returns:
            str: A string containing the response from the knowledge base.
        """
        # Get the relevant documents from the knowledge base
        retrieval_timer = Timer()
        retrieval_timer.start()
        docs_from_knowledge = await self.get_relevant_docs_from_knowledge(query=query)
        if docs_from_knowledge is not None:
            # Truncate each document's content to prevent context overflow
            _max_doc_chars = 2000
            for doc in docs_from_knowledge:
                if isinstance(doc, dict) and isinstance(doc.get("content"), str):
                    if len(doc["content"]) > _max_doc_chars:
                        doc["content"] = doc["content"][:_max_doc_chars] + "..."
                elif hasattr(doc, "content") and isinstance(doc.content, str):
                    if len(doc.content) > _max_doc_chars:
                        doc.content = doc.content[:_max_doc_chars] + "..."

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

    def _create_run_data(self) -> Dict[str, Any]:
        """Create and return the run data dictionary."""
        run_response_format = "text"
        if self.response_model is not None:
            run_response_format = "json"
        elif self.prompt_config.markdown:
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

        if self.enable_tracing:
            run_data.update(
                {
                    "run_input": self.run_input,
                    "run_response": self.run_response.to_dict(),
                    "run_response_format": run_response_format,
                }
            )

        return run_data
