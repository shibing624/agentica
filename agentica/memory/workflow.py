# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Workflow memory models
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict

from agentica.utils.log import logger
from agentica.run_response import RunResponse


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

    def create_empty_copy(self, *, update: Optional[Dict[str, Any]] = None) -> "WorkflowMemory":
        """Create a new empty WorkflowMemory with same config but no runs."""
        new_memory = self.model_copy(deep=True, update=update)
        new_memory.clear()
        return new_memory

    # Keep backward compat alias
    deep_copy = create_empty_copy
