# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for Workflow async pipeline.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.workflow import Workflow
from agentica.run_response import RunResponse


# ===========================================================================
# TestWorkflowBasic
# ===========================================================================


class TestWorkflowBasic:
    """Tests for Workflow base class."""

    def test_workflow_initialization(self):
        wf = Workflow(name="TestWorkflow")
        assert wf.name == "TestWorkflow"

    def test_workflow_run_is_async(self):
        assert asyncio.iscoroutinefunction(Workflow.run)

    def test_workflow_has_run_sync(self):
        wf = Workflow(name="W")
        assert hasattr(wf, "run_sync")
        assert not asyncio.iscoroutinefunction(wf.run_sync)

    def test_workflow_session_id(self):
        wf = Workflow(name="W", session_id="wf-sess-1")
        assert wf.session_id == "wf-sess-1"

    def test_workflow_memory(self):
        wf = Workflow(name="W")
        assert wf.memory is not None


# ===========================================================================
# TestWorkflowMultiStep
# ===========================================================================


class TestWorkflowMultiStep:
    """Tests for multi-step Workflow execution."""

    @pytest.mark.asyncio
    async def test_subclass_workflow_runs(self):
        """A subclass overriding run() should work."""
        class MyWorkflow(Workflow):
            async def run(self, message=None, **kwargs):
                return RunResponse(content="Step completed")

        wf = MyWorkflow(name="Multi")
        resp = await wf.run("Start")
        assert resp.content == "Step completed"

    def test_subclass_workflow_run_sync(self):
        """run_sync should work for subclassed workflow."""
        class MyWorkflow(Workflow):
            async def run(self, message=None, **kwargs):
                return RunResponse(content="Sync step done")

        wf = MyWorkflow(name="Multi")
        resp = wf.run_sync("Start")
        assert resp.content == "Sync step done"

    @pytest.mark.asyncio
    async def test_workflow_sequential_steps(self):
        """Simulate sequential agent steps."""
        results = []

        class PipelineWorkflow(Workflow):
            async def run(self, message=None, **kwargs):
                results.append("step1")
                results.append("step2")
                results.append("step3")
                return RunResponse(content=f"Completed {len(results)} steps")

        wf = PipelineWorkflow(name="Pipeline")
        resp = await wf.run("Start")
        assert len(results) == 3
        assert "3 steps" in resp.content


# ===========================================================================
# TestWorkflowSession
# ===========================================================================


class TestWorkflowSession:
    """Tests for Workflow session management."""

    @pytest.mark.asyncio
    async def test_workflow_read_from_storage_no_db(self):
        wf = Workflow(name="W")
        result = await wf.read_from_storage()
        assert result is None

    @pytest.mark.asyncio
    async def test_workflow_write_to_storage_no_db(self):
        wf = Workflow(name="W")
        result = await wf.write_to_storage()
        assert result is None

    @pytest.mark.asyncio
    async def test_workflow_read_from_storage_is_async(self):
        assert asyncio.iscoroutinefunction(Workflow.read_from_storage)

    @pytest.mark.asyncio
    async def test_workflow_write_to_storage_is_async(self):
        assert asyncio.iscoroutinefunction(Workflow.write_to_storage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
