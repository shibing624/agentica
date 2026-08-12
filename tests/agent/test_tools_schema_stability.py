# -*- coding: utf-8 -*-
"""Byte-stability guards for the tools array sent to providers (Reasonix
TestRegistrySchemasStableAndCanonical / TestWorkspaceToolSchemasStableAcrossRoots).

Tools sit between system and messages in the request, so any schema drift —
ordering, docstring parsing, cwd leakage — cold-starts the prompt cache from
that point on. MCP tools are deliberately out of scope: their schema bytes are
server-supplied, and our serialisation cannot stabilise what the server sends;
the fake is not worth the fidelity loss.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

from agentica.agent import Agent
from agentica.model.openai import OpenAIChat


def lookup_city(name: str, country: str = "CN") -> str:
    """Look up a city's timezone.

    Args:
        name: City name, e.g. "Beijing".
        country: ISO country code.
    """
    return f"{name}/{country}"


def add_numbers(a: int, b: float = 0.5) -> float:
    """Add two numbers.

    Args:
        a: First operand.
        b: Second operand, defaults to 0.5.
    """
    return a + b


def _tools_json() -> str:
    """Serialise the request-bound tools array through the real assembly path."""
    agent = Agent(
        name="SchemaStability",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tools=[lookup_city, add_numbers],
    )
    agent.update_model()
    tools_for_api = agent.model.get_tools_for_api()
    assert tools_for_api, "no tools reached the wire format"
    return json.dumps(tools_for_api, ensure_ascii=False)


def _digest(text: str) -> str:
    return f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}:{len(text)}"


# Self-contained child script (only imports agentica) printed as one DIGEST line.
_CHILD_SCRIPT = r"""
import hashlib, json
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat

def lookup_city(name: str, country: str = "CN") -> str:
    '''Look up a city's timezone.

    Args:
        name: City name, e.g. "Beijing".
        country: ISO country code.
    '''
    return f"{name}/{country}"

def add_numbers(a: int, b: float = 0.5) -> float:
    '''Add two numbers.

    Args:
        a: First operand.
        b: Second operand, defaults to 0.5.
    '''
    return a + b

agent = Agent(
    name="SchemaStability",
    model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    tools=[lookup_city, add_numbers],
)
agent.update_model()
tools_for_api = agent.model.get_tools_for_api()
text = json.dumps(tools_for_api, ensure_ascii=False)
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
print(f"DIGEST={digest}:{len(text)}")
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


class TestToolsSchemaByteStability:
    def test_two_builds_byte_identical(self):
        first, second = _tools_json(), _tools_json()
        assert first == second, (
            "tools array drifted between two builds "
            f"({len(first)} vs {len(second)} bytes)"
        )

    def test_stable_across_cwd(self):
        """Schemas must not embed the process cwd (absolute-path drift)."""
        cwd = os.getcwd()
        baseline = _tools_json()
        with tempfile.TemporaryDirectory() as other:
            try:
                os.chdir(other)
                assert os.getcwd() != cwd
                moved = _tools_json()
            finally:
                os.chdir(cwd)
        assert baseline == moved, "tools array changed when built from another cwd"

    def test_cross_process_byte_identical(self):
        digest_a = _run_child("1")
        digest_b = _run_child("2")
        assert digest_a == digest_b, (
            f"tools array differs across hash seeds: {digest_a} vs {digest_b}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
