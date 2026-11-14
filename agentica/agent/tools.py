# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Default tools for Agent.

This module contains the default tool implementations that can be
enabled on the Agent.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from agentica.tools.base import Function, ModelTool, Tool


class ToolsMixin:
    """Mixin class containing default tool implementations for Agent."""

    def get_tools(self) -> Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]]:
        """Get all tools available to this agent.

        Multi-agent composition is handled separately by ``Agent.as_tool()``
        and ``BuiltinTaskTool``; this method does not inject any implicit
        delegation tools.
        """
        tools: List[Union[ModelTool, Tool, Callable, Dict, Function]] = []

        if self.tools is not None:
            tools.extend(self.tools)

        return tools if len(tools) > 0 else []

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
