import asyncio
import unittest

import agentica.utils.hook_recorder as hook_recorder_module
from agentica.agent.hooks import RunHooks
from agentica.utils.hook_recorder import HookRecorder


class _MemoryHook(RunHooks):
    async def on_agent_end(self, agent, output, **kwargs):
        return None


class TestHookRecorderLangfuse(unittest.TestCase):
    def test_hooks_stay_internal_and_do_not_reach_langfuse(self):
        self.assertFalse(hasattr(hook_recorder_module, "langfuse_observation_context"))
        recorder = HookRecorder()
        default_hook = RunHooks()
        memory_hook = _MemoryHook()

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

        self.assertEqual(
            [(item["hook_class"], item["method"]) for item in recorder.export()],
            [("_MemoryHook", "on_agent_end")],
        )
