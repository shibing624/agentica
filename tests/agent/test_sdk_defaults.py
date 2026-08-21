# -*- coding: utf-8 -*-
"""
@description: The SDK's view of features built for the interactive CLI.

A backend service embeds Agent in a long-lived process, keeps its own
conversation store, and has no terminal for anyone to type /resume into. These
tests pin the seams where a CLI convenience would otherwise become that
service's problem.
"""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent import Agent
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.runner.compress import CompressMixin


def _agent(**kwargs) -> Agent:
    return Agent(model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"), **kwargs)


class TestSessionLogIsOptional(unittest.TestCase):
    def test_a_session_id_writes_a_transcript_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _agent(session_id="s1", session_base_dir=tmpdir)
        self.assertIsNotNone(agent._session_log)

    def test_a_service_with_its_own_store_can_turn_it_off(self):
        """The JSONL exists for /resume, /fork and /export. A service that has
        none of those gets a second copy of every turn under the process's home
        directory that nothing ever reads back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _agent(
                session_id="s1", session_base_dir=tmpdir, enable_session_log=False,
            )
            self.assertIsNone(agent._session_log)
            self.assertEqual(os.listdir(tmpdir), [])

    def test_the_session_id_itself_is_unaffected(self):
        agent = _agent(session_id="s1", enable_session_log=False)
        self.assertEqual(agent.session_id, "s1")


class TestCompactionIsReportedOnTheRun(unittest.TestCase):
    """Layer 2 is irreversible and costs an LLM call. Without a counter on the
    response, the only witness is the CLI event callback — so an SDK caller
    sees a slow, more expensive turn that quietly lost the early transcript,
    with nothing to attribute it to."""

    def test_a_run_that_never_compacts_reports_zero(self):
        self.assertEqual(_agent().run_response.context_compactions, 0)

    def test_a_reactive_compaction_is_counted(self):
        agent = _agent()
        cm = MagicMock()
        cm.auto_compact = AsyncMock(return_value=True)
        agent.tool_config.compression_manager = cm
        messages = [Message(role="user", content="hi")]

        compacted = asyncio.run(
            CompressMixin._try_reactive_compact(messages, agent, agent.model)
        )

        self.assertTrue(compacted)
        self.assertEqual(agent.run_response.context_compactions, 1)

    def test_a_refused_compaction_is_not_counted(self):
        agent = _agent()
        cm = MagicMock()
        cm.auto_compact = AsyncMock(return_value=False)
        agent.tool_config.compression_manager = cm

        asyncio.run(
            CompressMixin._try_reactive_compact(
                [Message(role="user", content="hi")], agent, agent.model,
            )
        )

        self.assertEqual(agent.run_response.context_compactions, 0)

    def test_layer_2_summarisation_is_counted(self):
        agent = _agent()
        agent.model.context_window = 1_000
        cm = MagicMock()
        cm.should_native_compact = MagicMock(return_value=False)
        cm.should_auto_compact = MagicMock(return_value=True)
        cm.auto_compact = AsyncMock(return_value=True)
        agent.tool_config.compression_manager = cm
        state = LoopState()

        asyncio.run(
            CompressMixin._maybe_compress_messages(
                [Message(role="user", content="hi")], agent, agent.model, state,
            )
        )

        self.assertEqual(agent.run_response.context_compactions, 1)
        self.assertTrue(state.context_collapsed)


class TestCompressionSwitches(unittest.TestCase):
    """Automatic Layer 1 / Layer 2 can be turned off; /compact stays available."""

    def test_defaults_are_on(self):
        from agentica.agent.config import ToolConfig

        tc = ToolConfig()
        self.assertTrue(tc.enable_evict)
        self.assertTrue(tc.enable_auto_compact)
        agent = _agent()
        self.assertTrue(agent.tool_config.enable_evict)
        self.assertTrue(agent.tool_config.enable_auto_compact)
        self.assertIsNone(agent.tool_config.compact_token_limit)
        self.assertIsNone(agent.tool_config.compression_manager.compact_token_limit)
        self.assertIsNotNone(agent.tool_config.compression_manager)

    def test_compact_token_limit_wires_to_manager(self):
        from agentica.agent.config import ToolConfig

        agent = _agent(tool_config=ToolConfig(compact_token_limit=300_000))
        self.assertEqual(agent.tool_config.compact_token_limit, 300_000)
        self.assertEqual(agent.tool_config.compression_manager.compact_token_limit, 300_000)

    def test_compact_token_limit_is_layer1_working_window(self):
        from agentica.agent.config import ToolConfig
        from agentica.compression import EvictionResult

        agent = _agent(tool_config=ToolConfig(compact_token_limit=8_000))
        agent.model.context_window = 32_768
        with patch("agentica.runner.compress.evict_context", return_value=EvictionResult()) as evict, \
             patch("agentica.runner.compress.count_tokens", return_value=7_000):
            asyncio.run(
                CompressMixin._maybe_compress_messages(
                    [Message(role="user", content="hi")], agent, agent.model, LoopState(),
                )
            )
        evict.assert_called_once()
        self.assertEqual(evict.call_args.kwargs["context_window"], 8_000)

    def test_layer1_off_skips_evict(self):
        from agentica.agent.config import ToolConfig
        from agentica.compression import EvictionResult

        agent = _agent(tool_config=ToolConfig(enable_evict=False))
        agent.model.context_window = 8_000
        with patch("agentica.runner.compress.evict_context", return_value=EvictionResult()) as evict:
            asyncio.run(
                CompressMixin._maybe_compress_messages(
                    [Message(role="user", content="hi")], agent, agent.model, LoopState(),
                )
            )
        evict.assert_not_called()

    def test_layer2_off_skips_auto_and_reactive_but_keeps_the_manager(self):
        from agentica.agent.config import ToolConfig

        agent = _agent(tool_config=ToolConfig(enable_auto_compact=False))
        agent.model.context_window = 1_000
        cm = MagicMock()
        cm.should_native_compact = MagicMock(return_value=False)
        cm.should_auto_compact = MagicMock(return_value=True)
        cm.auto_compact = AsyncMock(return_value=True)
        agent.tool_config.compression_manager = cm

        asyncio.run(
            CompressMixin._maybe_compress_messages(
                [Message(role="user", content="hi")], agent, agent.model, LoopState(),
            )
        )
        cm.auto_compact.assert_not_awaited()
        cm.should_auto_compact.assert_not_called()

        compacted = asyncio.run(
            CompressMixin._try_reactive_compact(
                [Message(role="user", content="hi")], agent, agent.model,
            )
        )
        self.assertFalse(compacted)
        cm.auto_compact.assert_not_awaited()
        self.assertIsNotNone(agent.tool_config.compression_manager)


if __name__ == "__main__":
    unittest.main()
