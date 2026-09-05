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
from types import SimpleNamespace
from typing import List, Optional

import pytest

from agentica.tools.builtin.delegate_tool import (
    DEPTH_ENV_VAR,
    MAX_CONCURRENT_DELEGATES,
    BuiltinDelegateTool,
    agentica_command,
    delegation_depth,
    profile_for_model,
    provider_for_model,
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


def _tool(
    registry,
    *,
    mode="allow-all",
    provider: Optional[str] = "deepseek",
    model: Optional[str] = "deepseek-chat",
    work_dir="/tmp/proj",
    profile_lookup=None,
    sdk_model=None,
    session_profile: Optional[str] = None,
):
    return BuiltinDelegateTool(
        background_process_registry=registry,
        permission_mode=lambda: mode,
        work_dir=work_dir,
        model_provider=provider,
        model_name=model,
        model=sdk_model,
        session_profile=session_profile,
        # Default "no profile matches" so argv tests never depend on the
        # machine's real config.yaml.
        profile_lookup=profile_lookup or (lambda name, provider=None, base_url=None: None),
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

    def test_the_callers_own_model_runs_on_its_profile_when_one_matches(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(registry, profile_lookup=lambda name, **_k: {"deepseek-chat": "deepseek-main"}.get(name)),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        # A profile carries base_url/api_key/tuning along; bare provider/name
        # flags cannot leave the child's active endpoint.
        assert argv[argv.index("--profile") + 1] == "deepseek-main"
        assert "--model_provider" not in argv

    def test_a_model_that_matches_a_profile_runs_on_it(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(registry, profile_lookup=lambda name, **_k: {"claude-opus-5": "opus-5-anthropic"}.get(name)),
            task="port the parser",
            model="anthropic/claude-opus-5",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--profile") + 1] == "opus-5-anthropic"
        assert "--model_provider" not in argv

    def test_an_sdk_model_sends_its_endpoint_as_a_flag_and_its_key_in_the_env(self):
        registry = _FakeRegistry()
        sdk_model = SimpleNamespace(
            id="internal-only-model", api_key="sk-sdk-key", base_url="http://llm.internal/v1"
        )
        _delegate(
            _tool(
                registry,
                provider="openai",
                model=sdk_model.id,
                sdk_model=sdk_model,
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "openai"
        assert argv[argv.index("--model_name") + 1] == "internal-only-model"
        # base_url is not a secret: it rides as a flag, so ps shows the truth
        # about where the worker sends traffic.
        assert argv[argv.index("--base_url") + 1] == "http://llm.internal/v1"
        # The key never appears in the command line; the child's model client
        # reads it from its environment.
        assert "sk-sdk-key" not in registry.started[0].command
        assert registry.started[0].env["OPENAI_API_KEY"] == "sk-sdk-key"

    def test_an_sdk_claude_model_uses_the_anthropic_env_var(self):
        registry = _FakeRegistry()
        sdk_model = SimpleNamespace(id="claude-opus-5", api_key="sk-ant", base_url=None)
        _delegate(
            _tool(
                registry,
                provider="anthropic",
                model=sdk_model.id,
                sdk_model=sdk_model,
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "anthropic"
        assert registry.started[0].env["ANTHROPIC_API_KEY"] == "sk-ant"
        assert "--base_url" not in argv

    def test_an_sdk_model_without_a_key_env_var_is_refused_not_launched(self):
        registry = _FakeRegistry()
        sdk_model = SimpleNamespace(id="gpt-4o", api_key="azure-secret", base_url=None)
        _delegate(
            _tool(
                registry,
                provider="azure",
                model=sdk_model.id,
                sdk_model=sdk_model,
            ),
            task="port the parser",
        )

        # Azure (and any provider without an env var the child's model client
        # reads) cannot hand a key to the child at all.
        assert registry.started == []

    def test_an_sdk_model_with_a_profile_match_still_prefers_the_profile(self):
        registry = _FakeRegistry()
        sdk_model = SimpleNamespace(id="glm-5.3-external", api_key="sk", base_url="http://x/v1")
        _delegate(
            _tool(
                registry,
                provider="openai",
                model=sdk_model.id,
                sdk_model=sdk_model,
                profile_lookup=lambda name, **_k: {"glm-5.3-external": "glm-5.3"}.get(name),
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--profile") + 1] == "glm-5.3"

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

    def test_inherit_uses_the_session_profile_not_the_first_name_match(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(
                registry,
                provider="openai",
                model="glm-5",
                session_profile="main",
                # Old behaviour: first config.yaml profile whose model_name
                # matches. That is how a worker landed on the cheap clone.
                profile_lookup=lambda name, **_k: "cheap",
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--profile") + 1] == "main"
        assert argv[argv.index("--model_name") + 1] == "glm-5"
        assert "--model_provider" not in argv

    def test_inherit_picks_the_profile_that_shares_the_callers_endpoint(self):
        profiles = {
            "cheap": {
                "model_name": "glm-5",
                "model_provider": "openai",
                "base_url": "http://cheap/",
            },
            "main": {
                "model_name": "glm-5",
                "model_provider": "openai",
                "base_url": "http://main/",
            },
        }
        registry = _FakeRegistry()
        sdk = SimpleNamespace(id="glm-5", api_key="sk", base_url="http://main/")
        _delegate(
            _tool(
                registry,
                provider="openai",
                model="glm-5",
                sdk_model=sdk,
                profile_lookup=lambda name, **k: profile_for_model(name, profiles=profiles, **k),
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--profile") + 1] == "main"

    def test_inherit_does_not_guess_a_profile_on_a_different_endpoint(self):
        profiles = {
            "cheap": {
                "model_name": "glm-5",
                "model_provider": "openai",
                "base_url": "http://cheap/",
            },
            "other": {
                "model_name": "glm-5",
                "model_provider": "openai",
                "base_url": "http://other/",
            },
        }
        registry = _FakeRegistry()
        sdk = SimpleNamespace(id="glm-5", api_key="sk", base_url="http://main/")
        _delegate(
            _tool(
                registry,
                provider="openai",
                model="glm-5",
                sdk_model=sdk,
                profile_lookup=lambda name, **k: profile_for_model(name, profiles=profiles, **k),
            ),
            task="port the parser",
        )

        argv = _argv(registry.started[0])
        assert "--profile" not in argv
        assert argv[argv.index("--base_url") + 1] == "http://main/"

    def test_passing_the_callers_slashed_id_does_not_split_it(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(registry, provider="openai", model="openai/glm-5"),
            task="port the parser",
            model="openai/glm-5",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_provider") + 1] == "openai"
        assert argv[argv.index("--model_name") + 1] == "openai/glm-5"

    def test_environment_style_provider_slash_id_inherits(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(registry, provider="openai", model="openai/glm-5"),
            task="port the parser",
            model="openai/openai/glm-5",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--model_name") + 1] == "openai/glm-5"

    def test_a_slashed_id_that_a_profile_runs_is_kept_whole(self):
        registry = _FakeRegistry()
        _delegate(
            _tool(
                registry,
                provider="deepseek",
                model="deepseek-chat",
                profile_lookup=lambda name, **_k: "openai-glm" if name == "openai/glm-5" else None,
            ),
            task="port the parser",
            model="openai/glm-5",
        )

        argv = _argv(registry.started[0])
        assert argv[argv.index("--profile") + 1] == "openai-glm"
        assert "--model_provider" not in argv


class TestProfileForModel:
    def test_a_single_match_is_returned(self):
        profiles = {
            "deepseek-main": {"model_name": "deepseek-chat", "model_provider": "deepseek"},
            "other": {"model_name": "glm-5", "model_provider": "openai"},
        }
        assert profile_for_model("deepseek-chat", profiles=profiles) == "deepseek-main"

    def test_two_profiles_sharing_a_name_are_not_guessed(self):
        profiles = {
            "cheap": {"model_name": "glm-5", "model_provider": "openai", "base_url": "http://a/"},
            "main": {"model_name": "glm-5", "model_provider": "openai", "base_url": "http://b/"},
        }
        assert profile_for_model("glm-5", profiles=profiles) is None

    def test_base_url_breaks_a_name_tie(self):
        profiles = {
            "cheap": {"model_name": "glm-5", "model_provider": "openai", "base_url": "http://a/"},
            "main": {"model_name": "glm-5", "model_provider": "openai", "base_url": "http://b/"},
        }
        assert profile_for_model("glm-5", base_url="http://b/", profiles=profiles) == "main"

    def test_provider_breaks_a_name_tie(self):
        profiles = {
            "zhipu": {"model_name": "glm-5", "model_provider": "zhipuai"},
            "openai-glm": {"model_name": "glm-5", "model_provider": "openai"},
        }
        assert profile_for_model("glm-5", provider="zhipuai", profiles=profiles) == "zhipu"


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

    def test_a_cross_provider_model_with_no_profile_or_key_starts_nothing(self):
        registry = _FakeRegistry()

        result = _delegate(
            _tool(registry),
            task="port the parser",
            model="anthropic/claude-opus-5",
        )

        # Credentials never travel the command line, so a provider nothing on
        # this machine is configured for can only fail authentication (the
        # child falls back to the provider's public endpoint + env key).
        assert result.startswith("Nothing delegated")
        assert "claude-opus-5" in result
        assert registry.started == []


class TestToolSurface:
    def test_the_agent_sees_one_function_called_delegate(self):
        tool = _tool(_FakeRegistry())
        assert list(tool.functions) == ["delegate"]

    def test_the_description_warns_the_worker_cannot_reach_the_user(self):
        tool = _tool(_FakeRegistry())
        description = tool.functions["delegate"].description or ""
        assert "cannot ask anyone anything" in description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
