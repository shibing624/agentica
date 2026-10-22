import asyncio
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from agentica.agent.hooks import RunHooks
from agentica.utils.hook_recorder import HookRecorder


class _MemoryHook(RunHooks):
    async def on_agent_end(self, agent, output, **kwargs):
        return None


class _Observation:
    def set_output(self, _output):
        pass


class TestHookRecorderLangfuse(unittest.TestCase):
    def test_only_overridden_hook_method_creates_child_span(self):
        recorder = HookRecorder()
        default_hook = RunHooks()
        memory_hook = _MemoryHook()
        exported = []

        @contextmanager
        def capture(**kwargs):
            exported.append(kwargs)
            yield _Observation()

        with patch(
            "agentica.utils.hook_recorder.langfuse_observation_context",
            new=capture,
        ):
            asyncio.run(
                recorder.run(
                    default_hook,
                    "run",
                    "on_tool_end",
                    default_hook.on_tool_end(None, tool_name="execute"),
                    base_class=RunHooks,
                    metadata={"tool_name": "execute"},
                )
            )
            asyncio.run(
                recorder.run(
                    memory_hook,
                    "run",
                    "on_agent_end",
                    memory_hook.on_agent_end(None, "done"),
                    base_class=RunHooks,
                )
            )

        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["as_type"], "span")
        self.assertEqual(exported[0]["name"], "hook.run._MemoryHook.on_agent_end")
