# -*- coding: utf-8 -*-
"""Per-turn long-term memory recall and memory-level secret redaction."""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent.config import WorkspaceMemoryConfig
from agentica.memory.recall import (
    MEMORY_RECALL_OPEN,
    build_memory_recall,
    parse_selection,
    render_recall_block,
    surfaced_memory_files,
)
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run.response import RunResponse
from agentica.security.redact import redact_sensitive_text
from agentica.workspace import Workspace


def _model() -> OpenAIChat:
    return OpenAIChat(id="gpt-4o", api_key="fake_openai_key")


async def _seed(ws: Workspace) -> None:
    await ws.write_memory_entry(
        title="prefer uv", content="Use uv, never pip, for installs.",
        memory_type="user", description="package manager choice uv pip",
    )
    await ws.write_memory_entry(
        title="gateway port", content="The dev gateway listens on 9091.",
        memory_type="reference", description="dev gateway port",
    )


class TestRedactMemoryLevel(unittest.TestCase):
    def test_masks_credential_assignment_but_keeps_numeric_tuning(self):
        text = "SERVICE_API_KEY=7804#apm-log-9d96db8ad9126d4a set; compact_token_limit=300000; max_tokens: 50"
        out = redact_sensitive_text(text, level="memory")
        self.assertNotIn("9d96db8ad9126d4a", out)
        self.assertIn("SERVICE_API_KEY=***REDACTED***", out)
        self.assertIn("compact_token_limit=300000", out)
        self.assertIn("max_tokens: 50", out)

    def test_masks_bare_high_entropy_token(self):
        token = "Rsjg81xRgfUtnxi2MNmSZAjVfyoKt9dYJsfQ9CcFQ76wsF"
        out = redact_sensitive_text(f"auth with {token} please", level="memory")
        self.assertNotIn(token, out)
        self.assertIn("***REDACTED_SECRET***", out)

    def test_keeps_numeric_user_facts(self):
        for text in (
            "账户 id 887656489",
            "广告主账户ID: 887656489",
            "account_id=887656489",
            "AUTH_ID=887656489",
            "advertiser_id=1708765432109876543",
        ):
            self.assertEqual(redact_sensitive_text(text, level="memory"), text)

    def test_keeps_paths_hex_digests_and_identifiers(self):
        text = (
            "see /Users/somebody/Documents/Codes/agentica/agentica/compression/evict.py, "
            "commit 093c23d2a8b1c4e5f6a7b8c9d0e1f2a3b4c5d6e7, "
            "fn test_handle_survives_a_compact_that_shadows_the_original_row"
        )
        self.assertEqual(redact_sensitive_text(text, level="memory"), text)

    def test_default_level_is_unchanged(self):
        text = "API_KEY=abcdefghijklmnop"
        self.assertEqual(redact_sensitive_text(text), text)


class TestWorkspaceMemoryRedaction(unittest.TestCase):
    def test_write_memory_entry_does_not_persist_a_pasted_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            path = asyncio.run(ws.write_memory_entry(
                title="agentlens setup",
                content="Set AI_SALES_AGENTLENS_API_KEY='7804#apm-log-9d96db8ad9126d4a' then restart.",
                memory_type="reference",
            ))
            with open(path, encoding="utf-8") as f:
                stored = f.read()
            index = ws.memory_index_path().read_text(encoding="utf-8")
        self.assertNotIn("9d96db8ad9126d4a", stored)
        self.assertNotIn("9d96db8ad9126d4a", index)
        self.assertIn("restart", stored)


class TestRecallHelpers(unittest.TestCase):
    def test_parse_selection_fails_closed(self):
        allowed = {"a.md", "b.md"}
        self.assertEqual(parse_selection('{"files": ["b.md", "x.md", "a.md"]}', allowed, 5), ["b.md", "a.md"])
        self.assertEqual(parse_selection('```json\n{"files": ["a.md"]}\n```', allowed, 5), ["a.md"])
        self.assertEqual(parse_selection("a.md", allowed, 5), [])
        self.assertEqual(parse_selection('{"files": "a.md"}', allowed, 5), [])
        self.assertEqual(parse_selection('{"files": ["a.md", "b.md"]}', allowed, 1), ["a.md"])

    def test_surfaced_files_read_back_from_the_rendered_block(self):
        block = render_recall_block([("prefer uv", "user_prefer_uv.md", "Use uv.")])
        msgs = [Message(role="user", content=f"hi\n\n{block}"), Message(role="assistant", content="ok")]
        self.assertEqual(surfaced_memory_files(msgs), {"user_prefer_uv.md"})

    def test_render_respects_the_total_budget(self):
        items = [(f"t{i}", f"f{i}.md", "x" * 1900) for i in range(10)]
        block = render_recall_block(items)
        self.assertLess(len(block), 6_000 + 2_500)
        self.assertNotIn("f9.md", block)


class TestBuildMemoryRecall(unittest.TestCase):
    def _run(self, ws, picked_json, context=()):
        model = _model()
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=picked_json))) as m:
            block = asyncio.run(build_memory_recall(ws, model, "how do I install deps?", list(context), 5, 5.0))
        return block, m

    def test_selected_body_is_recalled_and_selector_sees_no_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            block, mock = self._run(ws, '{"files": ["user_prefer_uv.md"]}')
        self.assertIn(MEMORY_RECALL_OPEN, block)
        self.assertIn("Use uv, never pip", block)
        self.assertNotIn("9091", block)
        request = mock.await_args.kwargs["messages"]
        self.assertIn("user_prefer_uv.md: prefer uv", request[1].content)
        self.assertIn("how do I install deps?", request[1].content)

    def test_entries_already_in_context_are_not_offered_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            prior = render_recall_block([("prefer uv", "user_prefer_uv.md", "Use uv.")])
            block, mock = self._run(
                ws, '{"files": ["user_prefer_uv.md"]}',
                context=[Message(role="user", content=f"earlier\n\n{prior}")],
            )
        self.assertIsNone(block)
        self.assertNotIn("user_prefer_uv.md", mock.await_args.kwargs["messages"][1].content)

    def test_selector_failure_degrades_to_no_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            model = _model()
            with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=ConnectionError("down"))):
                block = asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0))
        self.assertIsNone(block)

    def test_empty_store_makes_no_side_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            block, mock = self._run(Workspace(tmp), '{"files": []}')
        self.assertIsNone(block)
        mock.assert_not_awaited()

    def test_memory_written_after_session_start_is_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(ws.freeze_snapshots())
            asyncio.run(_seed(ws))
            block, _ = self._run(ws, '{"files": ["reference_gateway_port.md"]}')
        self.assertIn("9091", block)


class TestAgentRecallInjection(unittest.TestCase):
    def _agent(self, tmp, recall: bool):
        from agentica import Agent

        agent = Agent(
            model=_model(),
            workspace=Workspace(tmp),
            enable_long_term_memory=True,
            long_term_memory_config=WorkspaceMemoryConfig(recall_memories=recall),
        )
        agent.run_response = RunResponse()
        return agent

    def test_recall_is_appended_to_the_user_message_not_the_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ):
                system, users, _ = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        self.assertTrue(users[0].content.startswith("install deps please"))
        self.assertIn("Use uv, never pip", users[0].content)
        self.assertNotIn("Use uv, never pip", system.content)

    def test_disabled_recall_makes_no_side_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=False)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock()) as mock:
                _, users, _ = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        self.assertEqual(users[0].content, "install deps please")
        mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
