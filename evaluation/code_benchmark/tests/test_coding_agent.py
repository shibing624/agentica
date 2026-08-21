# -*- coding: utf-8 -*-
"""Unit tests for the coding-eval agent factory (no LLM)."""
from evaluation.code_benchmark.common import (
    EVAL_DROP_TOOLS,
    INSTRUCTIONS_ADDENDUM,
    build_coding_agent,
    build_model,
    drop_eval_tools,
)


def test_addendum_skips_listing_and_stops_when_green():
    assert "Do not list, glob, or grep the directory" in INSTRUCTIONS_ADDENDUM
    assert "stop immediately" in INSTRUCTIONS_ADDENDUM
    assert "{file_list}" in INSTRUCTIONS_ADDENDUM
    assert "{test_list}" in INSTRUCTIONS_ADDENDUM


def test_drop_eval_tools_removes_glob():
    class Model:
        functions = {"glob": object(), "read_file": object()}

    class FileTool:
        functions = {"glob": object(), "read_file": object()}

    class Agent:
        model = Model()
        tools = [FileTool()]

    agent = Agent()
    drop_eval_tools(agent)
    assert "glob" not in agent.model.functions
    assert "read_file" in agent.model.functions
    assert "glob" not in agent.tools[0].functions


def test_eval_agent_schema_has_no_todo_or_glob(tmp_path):
    model = build_model("test-model", api_key="test-key", wire_api="responses")
    agent = build_coding_agent(model, tmp_path, tmp_path / "home")
    names = set()
    for tool in agent.tools or []:
        names.update((tool.functions or {}).keys())
    for name in EVAL_DROP_TOOLS:
        assert name not in names
    assert "todo" not in names
    assert "write_todos" not in names
    assert "read_file" in names
    assert "apply_patch" in names
    assert "write_file" in names
    assert "ls" not in names
    assert "edit_file" not in names
    assert "grep" not in names
    assert "execute" in names
