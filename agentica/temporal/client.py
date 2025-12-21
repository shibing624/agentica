# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Client wrapper for Agentica.

Provides a simplified interface for starting and managing Temporal workflows.
"""
from dataclasses import dataclass, field
from typing import Optional, Type, Any, List
import uuid
try:
    from temporalio.client import Client, WorkflowHandle
except ImportError:
    raise ImportError("Temporal SDK not installed. Please install via: pip install temporalio")

from agentica.utils.log import logger


@dataclass
class WorkflowResult:
    """Result from a workflow execution.
    
    Attributes:
        workflow_id: The unique ID of the workflow.
        content: The final output content.
        steps: List of step outputs (for multi-agent workflows).
    """
    workflow_id: str
    content: str
    steps: List[Any] = field(default_factory=list)


class TemporalClient:
    """
    Wrapper for Temporal Client operations.
    
    Provides a simplified interface for common Temporal operations
    like starting workflows, getting results, and checking status.
    
    Usage:
        client = TemporalClient()
        await client.connect()
        
        # Start workflow
        workflow_id = await client.start_workflow(
            AgentWorkflow,
            WorkflowInput(message="Hello"),
        )
        
        # Get result (blocks until complete)
        result = await client.get_result(workflow_id)
        print(result.content)
        
    Attributes:
        host: Temporal server address (default: localhost:7233)
        namespace: Temporal namespace (default: default)
        task_queue: Task queue name (default: agentica-task-queue)
    """
    
    def __init__(
        self,
        host: str = "localhost:7233",
        namespace: str = "default",
        task_queue: str = "agentica-task-queue",
    ):
        """Initialize TemporalClient.
        
        Args:
            host: Temporal server address.
            namespace: Temporal namespace.
            task_queue: Default task queue for workflows.
        """
        self.host = host
        self.namespace = namespace
        self.task_queue = task_queue
        self._client: Optional[Client] = None
    
    async def connect(self) -> "TemporalClient":
        """Connect to Temporal server.
        
        Returns:
            Self for method chaining.
            
        Raises:
            Exception: If connection fails.
        """
        self._client = await Client.connect(
            self.host,
            namespace=self.namespace,
        )
        logger.info(f"Connected to Temporal server: {self.host}")
        return self
    
    @property
    def client(self) -> Client:
        """Get the underlying Temporal client.
        
        Returns:
            The Temporal Client instance.
            
        Raises:
            RuntimeError: If not connected.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client
    
    async def start_workflow(
        self,
        workflow_class: Type,
        input: Any,
        workflow_id: Optional[str] = None,
        task_queue: Optional[str] = None,
    ) -> str:
        """Start a new workflow and return workflow_id.
        
        Args:
            workflow_class: The workflow class to run.
            input: Input data for the workflow.
            workflow_id: Optional custom workflow ID.
            task_queue: Optional task queue (uses default if not provided).
            
        Returns:
            The workflow ID.
        """
        workflow_id = workflow_id or f"agentica-{uuid.uuid4().hex[:8]}"
        task_queue = task_queue or self.task_queue
        
        await self.client.start_workflow(
            workflow_class.run,
            input,
            id=workflow_id,
            task_queue=task_queue,
        )
        
        logger.info(f"Started workflow: {workflow_id}")
        return workflow_id
    
    async def get_result(self, workflow_id: str) -> WorkflowResult:
        """Get the result of a workflow (blocks until complete).
        
        Args:
            workflow_id: The workflow ID to get results for.
            
        Returns:
            WorkflowResult with content and steps.
        """
        handle = self.client.get_workflow_handle(workflow_id)
        result = await handle.result()
        
        # Handle different result types
        if isinstance(result, str):
            # Direct string result (e.g., ParallelTranslationWorkflow)
            return WorkflowResult(
                workflow_id=workflow_id,
                content=result,
                steps=[],
            )
        elif hasattr(result, 'content'):
            # WorkflowOutput with content and steps
            return WorkflowResult(
                workflow_id=workflow_id,
                content=result.content,
                steps=result.steps if hasattr(result, 'steps') else [],
            )
        else:
            # Unknown type, convert to string
            return WorkflowResult(
                workflow_id=workflow_id,
                content=str(result),
                steps=[],
            )
    
    async def get_status(self, workflow_id: str) -> str:
        """Get the current status of a workflow.
        
        Args:
            workflow_id: The workflow ID to check.
            
        Returns:
            Status string (e.g., "RUNNING", "COMPLETED", "FAILED").
        """
        handle = self.client.get_workflow_handle(workflow_id)
        desc = await handle.describe()
        return desc.status.name
    
    def get_handle(self, workflow_id: str) -> WorkflowHandle:
        """Get a workflow handle for advanced operations.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            WorkflowHandle for direct Temporal operations.
        """
        return self.client.get_workflow_handle(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow.
        
        Args:
            workflow_id: The workflow ID to cancel.
        """
        handle = self.client.get_workflow_handle(workflow_id)
        await handle.cancel()
        logger.info(f"Cancelled workflow: {workflow_id}")
    
    async def terminate_workflow(
        self, 
        workflow_id: str, 
        reason: str = "Terminated by user"
    ) -> None:
        """Terminate a workflow immediately.
        
        Args:
            workflow_id: The workflow ID to terminate.
            reason: Reason for termination.
        """
        handle = self.client.get_workflow_handle(workflow_id)
        await handle.terminate(reason=reason)
        logger.info(f"Terminated workflow: {workflow_id}")
