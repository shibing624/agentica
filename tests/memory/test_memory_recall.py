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
    RECALL_MAX_CONSECUTIVE_FAILURES,
    SELECT_MAX_OUTPUT_TOKENS,
    RecallBreaker,
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
    def test_parse_selection_tells_failure_from_an_empty_choice(self):
        allowed = {"a.md", "b.md"}
        self.assertEqual(parse_selection('{"files": ["b.md", "x.md", "a.md"]}', allowed, 5), ["b.md", "a.md"])
        self.assertEqual(parse_selection('```json\n{"files": ["a.md"]}\n```', allowed, 5), ["a.md"])
        self.assertEqual(parse_selection('{"files": []}', allowed, 5), [])
        self.assertIsNone(parse_selection("a.md", allowed, 5))
        self.assertIsNone(parse_selection("", allowed, 5))
        self.assertIsNone(parse_selection('{"files": "a.md"}', allowed, 5))
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


class TestMemoryOrigin(unittest.TestCase):
    def test_origin_is_recorded_shown_and_survives_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            proj = os.path.join(tmp, "Codes", "agentica")
            path = asyncio.run(ws.write_memory_entry(
                title="prefer uv", content="Use uv.\norigin_work_dir: /not/this",
                memory_type="user", origin_work_dir=proj,
            ))
            body, origin = asyncio.run(ws.read_memory_entry(os.path.basename(path)))
            asyncio.run(ws.write_memory_entry(
                title="gateway port", content="9091.", memory_type="reference",
                source="auto_extract", origin_work_dir=proj,
            ))
            promoted = asyncio.run(ws.promote_memory_candidate("reference_gateway_port.md"))
            _, promoted_origin = asyncio.run(ws.read_memory_entry(os.path.basename(promoted)))
            _, legacy_origin = asyncio.run(ws.read_memory_entry("missing.md"))
        self.assertEqual(origin, os.path.realpath(proj))
        self.assertIn("origin_work_dir: /not/this", body, "a body line is content, not provenance")
        self.assertEqual(promoted_origin, os.path.realpath(proj))
        self.assertEqual(legacy_origin, "")

    def test_block_shows_origin_tail_and_dedup_still_parses(self):
        home = os.path.expanduser("~")
        block = render_recall_block([
            ("prefer uv", "user_prefer_uv.md", "Use uv.", os.path.join(home, "Codes", "agentica")),
        ])
        self.assertIn("(written while working in ~/Codes/agentica)", block)
        outside = render_recall_block([("prefer uv", "user_prefer_uv.md", "Use uv.", "/srv/app")])
        self.assertIn("(written while working in /srv/app)", outside)
        self.assertEqual(surfaced_memory_files([Message(role="user", content=block)]), {"user_prefer_uv.md"})
        legacy = render_recall_block([("prefer uv", "user_prefer_uv.md", "Use uv.", "")])
        self.assertNotIn("written while", legacy)

    def test_recall_is_not_filtered_by_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(ws.write_memory_entry(
                title="prefer uv", content="Use uv.", memory_type="user",
                origin_work_dir=os.path.join(tmp, "other_project"),
            ))
            with patch.object(OpenAIChat, "response", new=AsyncMock(
                return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}'),
            )):
                block = asyncio.run(build_memory_recall(
                    ws, _model(), "install deps", [], 5, 5.0, RecallBreaker(), "s1",
                ))
        self.assertIn("Use uv.", block)
        self.assertIn("other_project", block)

class TestBuildMemoryRecall(unittest.TestCase):
    def _run(self, ws, picked_json, context=()):
        model = _model()
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=picked_json))) as m:
            block = asyncio.run(build_memory_recall(
                ws, model, "how do I install deps?", list(context), 5, 5.0, RecallBreaker(), "s1",
            ))
        return block, m

    def test_selector_request_is_tool_free_and_output_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key", max_tokens=64_000)
            seen = {}

            async def fake_response(self_model, messages):
                seen["max_tokens"] = self_model.max_tokens
                seen["tools"] = self_model.tools
                return ModelResponse(content='{"files": []}')

            with patch.object(OpenAIChat, "response", new=fake_response):
                asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, RecallBreaker(), "s1"))
        self.assertEqual(seen["max_tokens"], SELECT_MAX_OUTPUT_TOKENS)
        self.assertIsNone(seen["tools"])
        self.assertNotEqual(model.max_tokens, SELECT_MAX_OUTPUT_TOKENS, "the agent's own model must keep its cap")

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
                block = asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, RecallBreaker(), "s1"))
        self.assertIsNone(block)

    def test_breaker_skips_the_selector_after_three_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            model, breaker = _model(), RecallBreaker()
            failing = AsyncMock(side_effect=ConnectionError("down"))
            with patch.object(OpenAIChat, "response", new=failing):
                for _ in range(5):
                    asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, breaker, "s1"))
            self.assertEqual(failing.await_count, RECALL_MAX_CONSECUTIVE_FAILURES)

            ok = AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}'))
            with patch.object(OpenAIChat, "response", new=ok):
                still_open = asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, breaker, "s1"))
                new_session = asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, breaker, "s2"))
        self.assertIsNone(still_open)
        self.assertIn("Use uv", new_session)

    def test_timeouts_and_garbage_count_as_failures_but_an_empty_pick_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            asyncio.run(_seed(ws))
            model, breaker = _model(), RecallBreaker()

            async def slow(*args, **kwargs):
                await asyncio.sleep(1)
                return ModelResponse(content='{"files": []}')

            with patch.object(OpenAIChat, "response", new=slow):
                asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 0.05, breaker, "s1"))
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=""))):
                asyncio.run(build_memory_recall(ws, model, "install?", [], 5, 5.0, breaker, "s1"))
            self.assertEqual(breaker.failures, 2)
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content='{"files": []}'))):
                asyncio.run(build_memory_recall(ws, model, "17*23?", [], 5, 5.0, breaker, "s1"))
        self.assertEqual(breaker.failures, 0)

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


class TestSideCallOutputCap(unittest.TestCase):
    def test_each_wire_shape_takes_the_side_cap(self):
        from agentica.model.anthropic.claude import Claude
        from agentica.model.openai.responses import OpenAIResponses

        self.assertEqual(_model().side_call_copy(max_output_tokens=2048).max_tokens, 2048)
        self.assertEqual(
            OpenAIResponses(id="gpt-5.6-sol", api_key="fake_openai_key").side_call_copy(max_output_tokens=2048).max_output_tokens,
            2048,
        )
        self.assertEqual(Claude(id="claude-haiku-5-5", api_key="fake").side_call_copy(max_output_tokens=2048).max_tokens, 2048)

    def test_claude_fixed_thinking_budget_stays_below_the_cap(self):
        from agentica.model.anthropic.claude import Claude

        model = Claude(id="claude-haiku-5-5", api_key="fake", thinking={"type": "enabled", "budget_tokens": 4000})
        self.assertEqual(model.side_call_copy(max_output_tokens=2048).max_tokens, 6048)


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

    def test_recall_is_prepended_to_the_user_message_not_the_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ):
                system, users, _ = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        # 块前置、用户问题压底部（lost-in-the-middle：最新问题放最后读）
        self.assertTrue(users[0].content.startswith(MEMORY_RECALL_OPEN))
        self.assertTrue(users[0].content.endswith("install deps please"))
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

    def test_recall_fires_for_messages_entry_style_and_only_touches_the_last_user_turn(self):
        """IM-adapter style: caller rebuilds history every turn via ``messages=``."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ) as mock:
                _, users, _ = asyncio.run(agent.get_messages_for_run(
                    messages=[{"role": "user", "content": "earlier question"},
                              {"role": "assistant", "content": "earlier answer"},
                              {"role": "user", "content": "how do I install deps?"}],
                ))
            self.assertEqual(len(users), 3)
            self.assertEqual(users[0].content, "earlier question")          # 历史不动
            self.assertEqual(users[1].content, "earlier answer")
            self.assertTrue(users[2].content.startswith(MEMORY_RECALL_OPEN))
            self.assertTrue(users[2].content.endswith("how do I install deps?"))
            self.assertIn("Use uv, never pip", users[2].content)             # 块只拼本轮
            # selector 的 query 是本轮用户消息
            request = mock.await_args.kwargs["messages"]
            self.assertIn("how do I install deps?", request[1].content)

    def test_messages_entry_style_without_a_trailing_user_message_skips_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock()) as mock:
                _, _, _ = asyncio.run(agent.get_messages_for_run(
                    messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                ))
        mock.assert_not_awaited()

    def test_messages_entry_style_does_not_mutate_the_callers_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            caller_history = [Message(role="user", content="how do I install deps?")]
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ):
                _, users, _ = asyncio.run(agent.get_messages_for_run(messages=caller_history))
        self.assertEqual(caller_history[0].content, "how do I install deps?")
        self.assertTrue(users[0].content.startswith(MEMORY_RECALL_OPEN))

    def test_messages_entry_style_dedups_against_earlier_items_in_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, recall=True)
            asyncio.run(_seed(agent.workspace))
            prior = render_recall_block([("prefer uv", "user_prefer_uv.md", "Use uv.")])
            pick = AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}'))
            with patch.object(OpenAIChat, "response", new=pick):
                _, users, _ = asyncio.run(agent.get_messages_for_run(messages=[
                    {"role": "user", "content": f"{prior}\n\nhow do I install deps?"},
                    {"role": "assistant", "content": "use uv"},
                    {"role": "user", "content": "and for the lockfile?"},
                ]))
        offered = pick.await_args.kwargs["messages"][1].content
        self.assertNotIn("user_prefer_uv.md", offered)
        self.assertEqual(users[-1].content, "and for the lockfile?")


if __name__ == "__main__":
    unittest.main()
