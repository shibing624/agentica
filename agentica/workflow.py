# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Workflow - Multi-agent orchestration engine for deterministic pipelines.

Async-first public API:
- async: run()
- sync adapter: run_sync()

Workflow provides programmatic control over multi-agent execution:
- Deterministic step ordering (A -> B -> C, no LLM improvisation)
- Cross-agent data flow with type safety
- Session state persistence and recovery
- Per-step agent isolation (different models/tools/prompts)
"""
import asyncio
import collections.abc
import inspect

from os import getenv
from uuid import uuid4
from types import GeneratorType
from typing import Any, AsyncIterator, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict, field_validator

from agentica.utils.log import logger, set_log_level_to_debug
from agentica.utils.async_utils import run_sync
from agentica.run_response import RunResponse
from agentica.memory import WorkflowMemory, WorkflowRun
from agentica.db.base import BaseDb, SessionRow
from agentica.utils.misc import merge_dictionaries
from agentica.workflow_session import WorkflowSession


class Workflow(BaseModel):
    """Multi-agent orchestration engine for deterministic pipelines."""

    name: Optional[str] = None
    description: Optional[str] = None
    workflow_id: Optional[str] = Field(None, validate_default=True)
    workflow_data: Optional[Dict[str, Any]] = None

    user_id: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None

    session_id: Optional[str] = Field(None, validate_default=True)
    session_name: Optional[str] = None
    session_state: Dict[str, Any] = Field(default_factory=dict)

    memory: WorkflowMemory = WorkflowMemory()

    db: Optional[BaseDb] = None
    _workflow_session: Optional[WorkflowSession] = None

    debug_mode: bool = Field(False, validate_default=True)

    # Internal run state
    run_id: Optional[str] = None
    run_input: Optional[Dict[str, Any]] = None
    run_response: RunResponse = Field(default_factory=RunResponse)
    session_data: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("workflow_id", mode="before")
    def set_workflow_id(cls, v: Optional[str]) -> str:
        return v or str(uuid4())

    @field_validator("session_id", mode="before")
    def set_session_id(cls, v: Optional[str]) -> str:
        return v or str(uuid4())

    @field_validator("debug_mode", mode="before")
    def set_log_level(cls, v: bool) -> bool:
        if v or getenv("AGENTICA_DEBUG", "false").lower() == "true":
            set_log_level_to_debug()
        return v

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, **data):
        super().__init__(**data)
        self.name = self.name or self.__class__.__name__

        if self._has_custom_run():
            self._setup_run_wrapper()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        from agentica.agent import Agent
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Agent):
                value.session_id = self.session_id

    def _has_custom_run(self) -> bool:
        """Check if subclass has overridden run()."""
        return self.__class__.run is not Workflow.run

    def _setup_run_wrapper(self):
        """Replace self.run with a wrapper that adds lifecycle management."""
        user_run = self.__class__.run
        object.__setattr__(self, '_user_run', user_run)
        object.__setattr__(self, 'run', self._wrap_user_run)

    # ------------------------------------------------------------------
    # Public API: Subclass should override this
    # ------------------------------------------------------------------

    async def run(self, *args: Any, **kwargs: Any):
        """Execute workflow. Override in subclass to define your pipeline."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    def run_sync(self, *args: Any, **kwargs: Any):
        """Synchronous adapter for run()."""
        return run_sync(self.run(*args, **kwargs))

    # ------------------------------------------------------------------
    # Lifecycle wrapper
    # ------------------------------------------------------------------

    async def _wrap_user_run(self, *args, **kwargs):
        """Wrapper: lifecycle management -> user run -> result wrapping."""
        self._prepare_run(args, kwargs)
        result = self._user_run(self, *args, **kwargs)
        # Await if the subclass run() is async
        if inspect.isawaitable(result):
            result = await result
        return self._process_result(result)

    def _prepare_run(self, args, kwargs):
        """Initialize run state: ID, storage, etc."""
        self.run_id = str(uuid4())
        self.run_input = {"args": args, "kwargs": kwargs}
        self.run_response = RunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.workflow_id,
        )
        self.read_from_storage()

    def _process_result(self, result):
        """Wrap result based on type (generator or single)."""
        if isinstance(result, (GeneratorType, collections.abc.Iterator)):
            return self._wrap_generator(result)
        elif isinstance(result, RunResponse):
            return self._wrap_single(result)
        else:
            logger.warning(f"Workflow.run() should return RunResponse, got: {type(result)}")
            return result

    def _wrap_generator(self, gen):
        """Wrap generator result."""
        self.run_response.content = ""

        def result_generator():
            for item in gen:
                if isinstance(item, RunResponse):
                    self._annotate_response(item)
                yield item
            self._finalize_run()

        return result_generator()

    def _wrap_single(self, result: RunResponse):
        """Wrap single RunResponse."""
        self._annotate_response(result)
        self._finalize_run()
        return result

    def _annotate_response(self, response: RunResponse):
        """Inject run metadata into response."""
        response.run_id = self.run_id
        response.session_id = self.session_id
        response.workflow_id = self.workflow_id
        if response.content is not None and isinstance(response.content, str):
            self.run_response.content = response.content

    def _finalize_run(self) -> None:
        """Persist run to memory and storage."""
        self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
        self.write_to_storage()

    # ------------------------------------------------------------------
    # Session / Storage
    # ------------------------------------------------------------------

    def get_workflow_data(self) -> Dict[str, Any]:
        data = self.workflow_data or {}
        if self.name is not None:
            data["name"] = self.name
        return data

    def get_session_data(self) -> Dict[str, Any]:
        data = self.session_data or {}
        if self.session_name is not None:
            data["session_name"] = self.session_name
        if self.session_state:
            data["session_state"] = self.session_state
        return data

    def get_workflow_session(self) -> WorkflowSession:
        return WorkflowSession(
            session_id=self.session_id,
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            memory=self.memory.to_dict(),
            workflow_data=self.get_workflow_data(),
            user_data=self.user_data,
            session_data=self.get_session_data(),
        )

    def from_workflow_session(self, session: WorkflowSession):
        """Load Workflow from database session."""
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if self.workflow_id is None and session.workflow_id is not None:
            self.workflow_id = session.workflow_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id

        if session.workflow_data is not None:
            if self.name is None and "name" in session.workflow_data:
                self.name = session.workflow_data.get("name")
            if self.workflow_data is not None:
                merge_dictionaries(session.workflow_data, self.workflow_data)
            self.workflow_data = session.workflow_data

        if session.user_data is not None:
            if self.user_data is not None:
                merge_dictionaries(session.user_data, self.user_data)
            self.user_data = session.user_data

        if session.session_data is not None:
            if self.session_name is None and "session_name" in session.session_data:
                self.session_name = session.session_data.get("session_name")
            if "session_state" in session.session_data:
                session_state_from_db = session.session_data.get("session_state")
                if session_state_from_db and isinstance(session_state_from_db, dict) and len(session_state_from_db) > 0:
                    if len(self.session_state) > 0:
                        merge_dictionaries(session_state_from_db, self.session_state)
                    self.session_state = session_state_from_db
            if self.session_data is not None:
                merge_dictionaries(session.session_data, self.session_data)
            self.session_data = session.session_data

        if session.memory is not None:
            try:
                if "runs" in session.memory:
                    self.memory.runs = [WorkflowRun(**m) for m in session.memory["runs"]]
            except Exception as e:
                logger.warning(f"Failed to load WorkflowMemory: {e}")
        logger.debug(f"-*- WorkflowSession loaded: {session.session_id}")

    def read_from_storage(self) -> Optional[WorkflowSession]:
        """Load from database."""
        if self.db is not None and self.session_id is not None:
            session_row = self.db.read_session(session_id=self.session_id, user_id=self.user_id)
            if session_row is not None:
                self._workflow_session = WorkflowSession(
                    session_id=session_row.session_id,
                    workflow_id=session_row.agent_id,
                    user_id=session_row.user_id,
                    memory=session_row.memory,
                    workflow_data=session_row.agent_data,
                    user_data=session_row.user_data,
                    session_data=session_row.session_data,
                    created_at=session_row.created_at,
                    updated_at=session_row.updated_at,
                )
                self.from_workflow_session(session=self._workflow_session)
        return self._workflow_session

    def write_to_storage(self) -> Optional[WorkflowSession]:
        """Save to database."""
        if self.db is not None:
            workflow_session = self.get_workflow_session()
            session_row = SessionRow(
                session_id=workflow_session.session_id,
                agent_id=workflow_session.workflow_id,
                user_id=workflow_session.user_id,
                memory=workflow_session.memory,
                agent_data=workflow_session.workflow_data,
                user_data=workflow_session.user_data,
                session_data=workflow_session.session_data,
            )
            self.db.upsert_session(session_row)
            self._workflow_session = workflow_session
        return self._workflow_session

    def load_session(self, force: bool = False) -> Optional[str]:
        """Load or create session."""
        if self._workflow_session is not None and not force:
            if self.session_id is not None and self._workflow_session.session_id == self.session_id:
                return self._workflow_session.session_id

        if self.db is not None:
            self.read_from_storage()
            if self._workflow_session is None:
                self.write_to_storage()
                if self._workflow_session is None:
                    raise RuntimeError("Failed to create WorkflowSession in storage")
        return self.session_id

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def rename_session(self, session_id: str, name: str):
        if self.db is None:
            raise ValueError("Database is not set")
        session_row = self.db.read_session(session_id=session_id)
        if session_row is None:
            raise ValueError(f"WorkflowSession not found: {session_id}")
        if session_row.session_data is not None:
            session_row.session_data["session_name"] = name
        else:
            session_row.session_data = {"session_name": name}
        self.db.upsert_session(session_row)

    def delete_session(self, session_id: str):
        if self.db is None:
            raise ValueError("Database is not set")
        self.db.delete_session(session_id=session_id)

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> "Workflow":
        """Create a deep copy."""
        from copy import copy, deepcopy
        from agentica.agent import Agent

        fields_for_new = {}
        for field_name in self.model_fields_set:
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, Agent):
                fields_for_new[field_name] = value.deep_copy()
            elif field_name == "memory":
                fields_for_new[field_name] = value.create_empty_copy()
            elif isinstance(value, (list, dict, set, BaseDb, BaseModel)):
                try:
                    fields_for_new[field_name] = deepcopy(value)
                except Exception:
                    try:
                        fields_for_new[field_name] = copy(value)
                    except Exception:
                        fields_for_new[field_name] = value
            else:
                fields_for_new[field_name] = value

        if update:
            fields_for_new.update(update)

        return self.__class__(**fields_for_new)
