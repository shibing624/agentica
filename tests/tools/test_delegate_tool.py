# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the delegate tool (agentica/tools/builtin/delegate_tool.py).

Nothing here spawns a real session: the registry is a recorder, so the tests are
about what command line a delegation produces and what it refuses to start.
"""
import asyncio
import shlex
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from agentica.tools.builtin.delegate_tool import (
    DEPTH_ENV_VAR,
    MAX_CONCURRENT_DELEGATES,
    BuiltinDelegateTool,
    agentica_command,
    delegation_depth,
)


@dataclass
class _FakeProcess:
    id: str
    num: int
    pid: int
    label: str
    kind: str
    command: str
    cwd: Optional[str]
    env: dict
    log_path: str = "/tmp/delegate.log"
    running: bool = True


@dataclass
class _FakeRegistry:
    """Stands in for BackgroundProcessRegistry: records instead of spawning."""

    started: List[_FakeProcess] = field(default_factory=list)

    def start(self, command, *, cwd=None, env=None, kind="command", label=""):
        item = _FakeProcess(
            id=f"term_{len(self.started) + 1}",
            num=len(self.started) + 1,
            pid=4200 + len(self.started),
            label=label,
            kind=kind,
            command=command,
            cwd=cwd,
            env=env or {},
        )
        self.started.append(item)
        return item

    def list(self, *, include_finished=False, kind=None):
        items = [p for p in self.started if kind is None or p.kind == kind]
        if include_finished:
            return items
        return [p for p in items if p.running]


def _tool(registry, *, mode="allow-all", provider="deepseek", model="deepseek-chat", work_dir="/tmp/proj"):
    return BuiltinDelegateTool(
        background_process_registry=registry,
        permission_mode=lambda: mode,
        work_dir=work_dir,
        model_provider=provider,
        model_name=model,
    )


def _delegate(tool, **kwargs) -> str:
    return asyncio.run(tool.delegate(**kwargs))


def _argv(item: _FakeProcess) -> List[str]:
    return shlex.split(item.command)


class TestWhatItStarts:
    def test_the_worker_is_a_headless_one_shot_agentica_run(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser to v2", label="parser port")

        argv = _argv(registry.started[0])
        assert argv[: len(agentica_command())] == agentica_command()
        assert "--query" in argv
        # --print: the caller reads the worker's stdout as its report, so a
        # banner or a log line in there would be noise it has to parse around.
        assert "--print" in argv

    def test_it_starts_the_same_installation_this_session_runs(self):
        command = agentica_command()

        # `-m agentica.cli.main` would make runpy warn on stderr, and that
        # warning lands in the report the caller reads.
        assert "-m" not in command
        assert command[0] == sys.executable or command[0].endswith("agentica")

    def test_the_task_reaches_the_worker_with_the_headless_framing(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser to v2")

        argv = _argv(registry.started[0])
        query = argv[argv.index("--query") + 1]
        assert query.endswith("port the parser to v2")
        assert "no user at this terminal" in query

    def test_the_workers_permission_tier_is_the_callers_own(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry, mode="auto"), task="run the migration")

        argv = _argv(registry.started[0])
        assert argv[argv.index("--permissions") + 1] == "auto"

    def test_the_tier_is_read_when_delegating_not_when_the_agent_was_built(self):
        registry = _FakeRegistry()
        mode = {"value": "allow-all"}
        tool = BuiltinDelegateTool(
            background_process_registry=registry,
            permission_mode=lambda: mode["value"],
            work_dir="/tmp/proj",
        )

        # /permissions switches the tier in place, without rebuilding the agent.
        mode["value"] = "auto"
        _delegate(tool, task="run the migration")

        argv = _argv(registry.started[0])
        assert argv[argv.index("--permissions") + 1] == "auto"

    def test_the_worker_inherits_the_callers_model(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser")

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "deepseek"
        assert argv[argv.index("--model_name") + 1] == "deepseek-chat"

    def test_a_provider_qualified_model_overrides_both(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser", model="zhipuai/glm-4.7-flash")

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "zhipuai"
        assert argv[argv.index("--model_name") + 1] == "glm-4.7-flash"

    def test_a_bare_model_name_keeps_the_callers_provider(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser", model="deepseek-reasoner")

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "deepseek"
        assert argv[argv.index("--model_name") + 1] == "deepseek-reasoner"

    def test_it_runs_where_the_caller_runs_unless_told_otherwise(self, tmp_path):
        registry = _FakeRegistry()
        tool = _tool(registry)

        _delegate(tool, task="build it")
        _delegate(tool, task="build the other one", work_dir=str(tmp_path))

        assert registry.started[0].cwd == "/tmp/proj"
        assert registry.started[1].cwd == str(tmp_path)

    def test_it_is_recorded_as_a_delegation_under_a_readable_label(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser to v2", label="parser port")

        assert registry.started[0].kind == "delegate"
        assert registry.started[0].label == "parser port"

    def test_an_unlabelled_task_names_itself(self):
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser\nto v2, then run the suite")

        # The label is what the user sees in /ps and in the completion notice,
        # so it has to be one line.
        assert registry.started[0].label == "port the parser to v2, then run the suite"

    def test_the_handle_tells_the_caller_how_to_collect_the_result(self):
        registry = _FakeRegistry()
        result = _delegate(_tool(registry), task="port the parser", label="parser port")

        assert "term_1" in result
        assert 'wait(id="term_1")' in result


class TestDepth:
    def test_a_fresh_session_is_at_the_top_of_the_chain(self, monkeypatch):
        monkeypatch.delenv(DEPTH_ENV_VAR, raising=False)
        assert delegation_depth() == 0

    def test_the_worker_is_told_how_deep_it_is(self, monkeypatch):
        monkeypatch.delenv(DEPTH_ENV_VAR, raising=False)
        registry = _FakeRegistry()
        _delegate(_tool(registry), task="port the parser")

        assert registry.started[0].env[DEPTH_ENV_VAR] == "1"

    def test_a_junk_depth_reads_as_the_top(self, monkeypatch):
        monkeypatch.setenv(DEPTH_ENV_VAR, "not-a-number")
        assert delegation_depth() == 0


class TestRefusals:
    def test_three_at_once_is_the_limit(self):
        registry = _FakeRegistry()
        tool = _tool(registry)
        for i in range(MAX_CONCURRENT_DELEGATES):
            _delegate(tool, task=f"task {i}", label=f"task {i}")

        result = _delegate(tool, task="one more", label="one more")

        assert result.startswith("Nothing delegated")
        assert 'wait(id="term_1")' in result
        assert len(registry.started) == MAX_CONCURRENT_DELEGATES

    def test_a_finished_worker_frees_its_slot(self):
        registry = _FakeRegistry()
        tool = _tool(registry)
        for i in range(MAX_CONCURRENT_DELEGATES):
            _delegate(tool, task=f"task {i}")
        registry.started[0].running = False

        _delegate(tool, task="one more")

        assert len(registry.started) == MAX_CONCURRENT_DELEGATES + 1

    def test_plain_background_commands_do_not_use_up_the_slots(self):
        registry = _FakeRegistry()
        for i in range(5):
            registry.start(f"sleep {i}")

        _delegate(_tool(registry), task="port the parser")

        assert registry.started[-1].kind == "delegate"

    def test_an_empty_task_starts_nothing(self):
        registry = _FakeRegistry()

        assert _delegate(_tool(registry), task="   ").startswith("Nothing delegated")
        assert registry.started == []

    def test_a_work_dir_that_does_not_exist_starts_nothing(self):
        registry = _FakeRegistry()

        result = _delegate(_tool(registry), task="build it", work_dir="/no/such/place")

        assert result.startswith("Nothing delegated")
        assert registry.started == []


class TestToolSurface:
    def test_the_agent_sees_one_function_called_delegate(self):
        tool = _tool(_FakeRegistry())
        assert list(tool.functions) == ["delegate"]

    def test_the_description_steers_away_from_the_cheaper_options(self):
        tool = _tool(_FakeRegistry())
        description = tool.functions["delegate"].description or ""
        assert "`task` tool" in description
        assert "cannot ask anyone anything" in description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
