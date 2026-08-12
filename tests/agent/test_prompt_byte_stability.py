# -*- coding: utf-8 -*-
"""Byte-stability guards for the system prompt (Reasonix prompt_stability_test.go port).

The system prompt is the provider-cached prefix of every request: any byte of
nondeterminism between builds (set/dict iteration, environment probes, ordering
changes) cold-starts the prompt cache for the whole session. These tests pin
byte equality; the existing tests/agent/test_prompts.py freeze tests only pin
stability across mid-session mutations, not across independent builds.

The cross-process case exists because CPython fixes PYTHONHASHSEED per process:
two builds inside one process share hash randomisation, so set-iteration drift
is invisible there. Only comparing two subprocesses with different seeds sees it.
"""
import asyncio
import hashlib
import os
import subprocess
import sys

import pytest

from agentica.agent import Agent
from agentica.agent.config import PromptConfig
from agentica.model.message import VOLATILE_SYSTEM_MARKER
from agentica.model.openai import OpenAIChat
from agentica.tools.base import Tool


class _FixedTool(Tool):
    """Deterministic stand-in tool so the tools guidance section is exercised."""

    def run(self, x: str) -> str:
        return x


def _build_agent() -> Agent:
    """One fixed configuration; any drift in its assembly is what we pin."""
    return Agent(
        name="ByteStability",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        description="A deterministic test assistant",
        instructions=["Be concise", "Be accurate", "Prefer simple solutions"],
        tools=[_FixedTool()],
        prompt_config=PromptConfig(markdown=True),
    )


async def _system_text() -> str:
    msg = await _build_agent().get_system_message()
    assert msg is not None and isinstance(msg.content, str)
    return msg.content


def _first_divergence(a: str, b: str) -> str:
    """Small window around the first differing byte (Reasonix firstDivergence)."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    start = max(i - 40, 0)
    return f"...{a[start:i + 40]!r}... vs ...{b[start:i + 40]!r}... (len {len(a)} vs {len(b)})"


# Script executed by the subprocess cases; self-contained (only imports agentica,
# which is importable from the inherited cwd) and prints one DIGEST line.
_CHILD_SCRIPT = r"""
import asyncio, hashlib
from agentica.agent import Agent
from agentica.agent.config import PromptConfig
from agentica.model.openai import OpenAIChat
from agentica.tools.base import Tool

class _FixedTool(Tool):
    def run(self, x: str) -> str:
        return x

async def _system_text():
    agent = Agent(
        name="ByteStability",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        description="A deterministic test assistant",
        instructions=["Be concise", "Be accurate", "Prefer simple solutions"],
        tools=[_FixedTool()],
        prompt_config=PromptConfig(markdown=True),
    )
    msg = await agent.get_system_message()
    return msg.content

content = asyncio.run(_system_text())
digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
print(f"DIGEST={digest}:{len(content)}")
"""


def _run_child(pythonhashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=pythonhashseed)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"child failed (seed={pythonhashseed}):\n{proc.stderr[-2000:]}"
    for line in proc.stdout.splitlines():
        if line.startswith("DIGEST="):
            return line[len("DIGEST="):]
    raise AssertionError(f"no DIGEST line in child output:\n{proc.stdout[-2000:]}")


class TestSystemPromptByteStability:
    @pytest.mark.asyncio
    async def test_two_builds_byte_identical(self):
        first, second = await _system_text(), await _system_text()
        assert first == second, f"system prompt drifted between two builds\n" \
                                f"first diff: {_first_divergence(first, second)}"

    @pytest.mark.asyncio
    async def test_stable_prefix_byte_identical(self):
        """The cache-relevant part is everything before VOLATILE_SYSTEM_MARKER."""
        first, second = await _system_text(), await _system_text()
        assert VOLATILE_SYSTEM_MARKER in first, "marker missing — cache split not applied?"
        head_a = first.split(VOLATILE_SYSTEM_MARKER, 1)[0]
        head_b = second.split(VOLATILE_SYSTEM_MARKER, 1)[0]
        assert head_a == head_b, f"stable prefix drifted\nfirst diff: {_first_divergence(head_a, head_b)}"

    def test_cross_process_byte_identical(self):
        """Two subprocesses with different PYTHONHASHSEED must agree byte-for-byte."""
        digest_a = _run_child("1")
        digest_b = _run_child("2")
        assert digest_a == digest_b, (
            f"system prompt differs across hash seeds: {digest_a} vs {digest_b} "
            "(set-iteration or async-registration ordering leaked into the prompt)"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
