# -*- coding: utf-8 -*-
"""After /compact, session log must carry the post-compact transcript so
/fork and /resume keep the same stack the live process holds in memory.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.cli.commands.context import CommandContext
from agentica.hooks import MemoryExtractHooks
from agentica.cli.commands.session import _cmd_compact
from agentica.memory.models import AgentRun
from agentica.memory.session_log import SessionLog
from agentica.memory.working import WorkingMemory
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.run_response import RunResponse


def _run(turn: int) -> AgentRun:
    user = Message(role="user", content=f"question {turn} about the API design")
    assistant = Message(role="assistant", content=f"answer {turn} with details")
    return AgentRun(
        message=user,
        messages=[user],
        response=RunResponse(messages=[user, assistant]),
    )


class TestCompactPersistsForForkResume(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _agent_with_log(self, n_runs: int = 3):
        from agentica import Agent

        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
            add_history_to_context=True,
        )
        wm = WorkingMemory()
        for i in range(n_runs):
            run = _run(i)
            wm.add_run(run)
            wm.add_messages(run.response.messages)
        agent.working_memory = wm
        slog = SessionLog("compact-persist-test", base_dir=self.base)
        for msg in wm.messages:
            if msg.role == "system":
                continue
            slog.append(msg.role, msg.content or "")
        agent._session_log = slog
        agent.model._agent_ref = lambda: agent
        return agent, slog

    def test_compact_writes_boundary_and_post_compact_messages(self):
        agent, slog = self._agent_with_log()
        cm = agent.tool_config.compression_manager
        pre_compact = MagicMock()

        class Hooks:
            async def on_pre_compact(self, agent, messages=None, **kwargs):
                pre_compact()

            async def on_post_compact(self, agent, messages=None, **kwargs):
                pass

        agent._run_hooks = Hooks()
        ctx = CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=agent,
            tui_state={"context_tokens": 50_000, "context_window": 128_000},
        )

        with patch.object(
            cm, "_summarise_conversation", new_callable=AsyncMock, return_value="SUMMARY",
        ):
            _cmd_compact(ctx)

        pre_compact.assert_called_once()
        lines = Path(slog.path).read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        types = [e["type"] for e in entries]
        self.assertIn("compact_boundary", types)
        boundary_idx = types.index("compact_boundary")
        after = types[boundary_idx + 1:]
        # The preserved trailing turn, and only it: load() rebuilds the summary
        # turn from the boundary, so persisting it too would send it twice.
        self.assertEqual(after, ["user", "assistant"])
        self.assertNotIn(
            "SUMMARY",
            " ".join(str(e.get("content") or "") for e in entries[boundary_idx + 1:]),
        )

        resumed = slog.load()
        contents = " ".join(str(m.get("content") or "") for m in resumed)
        self.assertEqual(contents.count("SUMMARY"), 1)
        self.assertIn("question", contents)

    def test_fork_after_compact_keeps_summary_and_tail(self):
        agent, slog = self._agent_with_log()
        cm = agent.tool_config.compression_manager
        ctx = CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=agent,
            tui_state={"context_tokens": 50_000, "context_window": 128_000},
        )
        with patch.object(
            cm, "_summarise_conversation", new_callable=AsyncMock, return_value="FORK SUMMARY",
        ):
            _cmd_compact(ctx)

        forked = slog.fork("forked-after-compact")
        resumed = forked.load()
        contents = " ".join(str(m.get("content") or "") for m in resumed)
        self.assertIn("FORK SUMMARY", contents)
        self.assertIn("question", contents)


class TestMemoryExtractFitsAuxWindow(unittest.TestCase):
    def test_fit_text_keeps_tail_within_aux_window(self):
        model = MagicMock()
        model.context_window = 1_000  # tiny aux → 4*0.4*1000 = 1600 chars
        huge = ("OLD-TURN\n" * 500) + "RECENT-PREF keep this"
        fitted = MemoryExtractHooks._fit_text_to_model(model, huge)
        self.assertLessEqual(len(fitted), 1600)
        self.assertIn("RECENT-PREF", fitted)
        self.assertLess(len(fitted), len(huge))


if __name__ == "__main__":
    unittest.main()
