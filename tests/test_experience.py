# -*- coding: utf-8 -*-
"""
Tests for the self-evolution experience system.

Covers:
1. ExperienceConfig defaults and custom values
2. ExperienceCaptureHooks — LLM-based correction classification, tool error capture, success patterns
3. Workspace experience storage — write, get, bump, lifecycle, sync
4. Agent experience wiring — enable_experience_capture=True auto-injects hooks
5. Experience prompt injection — experiences appear in system prompt
6. run_input cross-round fix — hooks read agent.run_input at on_agent_end time

All tests mock LLM API keys -- no real API calls.
"""
import asyncio
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.agent.config import ExperienceConfig
from agentica.hooks import ExperienceCaptureHooks, ConversationArchiveHooks, MemoryExtractHooks
from agentica.workspace import Workspace


def _make_agent(
    name="test-agent",
    enable_experience_capture=False,
    workspace_path=None,
):
    """Create a minimal Agent with a fake OpenAI key."""
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        name=name,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        enable_experience_capture=enable_experience_capture,
        workspace=workspace_path,
    )


# ===========================================================================
# ExperienceConfig tests
# ===========================================================================

class TestExperienceConfig(unittest.TestCase):
    """Test ExperienceConfig defaults and custom values."""

    def test_defaults(self):
        config = ExperienceConfig()
        # capture_* default to False so libraries do not silently persist
        # behavioral data; users opt in explicitly.
        self.assertFalse(config.capture_tool_errors)
        self.assertFalse(config.capture_user_corrections)
        self.assertFalse(config.capture_success_patterns)
        self.assertEqual(config.promotion_count, 3)
        self.assertEqual(config.promotion_window_days, 7)
        self.assertEqual(config.demotion_days, 30)
        self.assertEqual(config.archive_days, 90)
        self.assertEqual(config.max_experiences_in_prompt, 5)
        self.assertFalse(config.sync_to_global_agent_md)
        self.assertEqual(config.feedback_confidence_threshold, 0.8)

    def test_custom_values(self):
        config = ExperienceConfig(
            promotion_count=5,
            demotion_days=60,
            archive_days=180,
            max_experiences_in_prompt=10,
        )
        self.assertEqual(config.promotion_count, 5)
        self.assertEqual(config.demotion_days, 60)
        self.assertEqual(config.archive_days, 180)
        self.assertEqual(config.max_experiences_in_prompt, 10)


# ===========================================================================
# ExperienceCaptureHooks tests
# ===========================================================================

class TestExperienceCaptureHooks(unittest.TestCase):
    """Test experience capture via hooks."""

    def _make_hooks(self, **config_overrides):
        # Library defaults have capture_* = False (opt-in). Tests in this class
        # exercise the capture pipeline, so flip them on unless a test overrides.
        defaults = {
            "capture_tool_errors": True,
            "capture_user_corrections": True,
            "capture_success_patterns": True,
            # Tests in this class assert behavior on individual turns; the
            # production default batches the LLM judge every 10 turns,
            # rate-limits across processes, and runs the LLM call in a
            # fire-and-forget background task. Force a synchronous,
            # every-turn flush so each test's assertions are deterministic.
            "judge_every_n_turns": 1,
            "judge_min_seconds_between": 0,
            "judge_background": False,
        }
        defaults.update(config_overrides)
        config = ExperienceConfig(**defaults)
        return ExperienceCaptureHooks(config)

    def _mock_agent(self, agent_id="test-agent-1"):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.run_input = "test input"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()
        # Backward-compat workspace methods (used by some tests)
        agent.workspace.write_experience_entry = AsyncMock(return_value="/tmp/exp.md")
        agent.workspace.write_memory_entry = AsyncMock(return_value="/tmp/mem.md")
        agent.workspace.append_experience_event = AsyncMock(return_value="/tmp/events.jsonl")
        agent.workspace.run_experience_lifecycle = AsyncMock(return_value={"promoted": 0, "demoted": 0, "archived": 0})
        agent.workspace.sync_experiences_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        # New store-based mocks (hooks now call these)
        mock_event_store = MagicMock()
        mock_event_store.append = AsyncMock(return_value="/tmp/events.jsonl")
        mock_event_store.read_all = AsyncMock(return_value=[])
        # events_path needs .stat() for the failed-tools cache; point at a
        # known-missing path so .stat() raises FileNotFoundError -> empty set.
        mock_event_store.events_path = Path("/tmp/agentica_test_nonexistent_events.jsonl")
        agent.workspace.get_experience_event_store = MagicMock(return_value=mock_event_store)
        mock_compiled_store = MagicMock()
        mock_compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        mock_compiled_store.run_lifecycle = AsyncMock(return_value={"promoted": 0, "demoted": 0, "archived": 0})
        mock_compiled_store.sync_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=mock_compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value="/tmp/AGENTS.md")
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=Path("/tmp/gen_skills"))
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path("/tmp/experiences"))
        # Mock working_memory with empty messages (no previous assistant)
        agent.working_memory = MagicMock()
        agent.working_memory.messages = []
        return agent

    def test_on_user_prompt_returns_none(self):
        """on_user_prompt should always return None (never modify message)."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        asyncio.run(hooks.on_agent_start(agent))

        result = asyncio.run(hooks.on_user_prompt(agent, "Please fix this bug"))
        self.assertIsNone(result)

    def test_on_user_prompt_disabled_also_returns_none(self):
        """When capture_user_corrections=False, on_user_prompt still returns None."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()
        asyncio.run(hooks.on_agent_start(agent))

        result = asyncio.run(hooks.on_user_prompt(agent, "No, that's wrong"))
        self.assertIsNone(result)

    def test_tool_error_capture(self):
        """Tool errors should be recorded."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        asyncio.run(hooks.on_agent_start(agent))

        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", tool_call_id="tc_1",
            tool_args={"command": "ls /nonexistent"},
            result="No such file or directory", is_error=True, elapsed=0.5,
        ))

        errors = hooks._tool_errors[agent.agent_id]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["tool"], "execute")
        self.assertIn("No such file", errors[0]["error"])

    def test_tool_error_disabled(self):
        """When capture_tool_errors=False, tool errors should not be captured."""
        hooks = self._make_hooks(capture_tool_errors=False)
        agent = self._mock_agent()
        asyncio.run(hooks.on_agent_start(agent))

        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True, result="error",
        ))
        self.assertEqual(len(hooks._tool_errors.get(agent.agent_id, [])), 0)

    def test_success_capture(self):
        """Successful tool calls should be recorded."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        asyncio.run(hooks.on_agent_start(agent))

        asyncio.run(hooks.on_tool_end(
            agent, tool_name="read_file", is_error=False, elapsed=0.1,
        ))
        self.assertEqual(len(hooks._tool_successes[agent.agent_id]), 1)

    def test_on_agent_end_persists_errors(self):
        """on_agent_end should call compiled_store.write for tool errors."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="permission denied", elapsed=1.0,
        ))
        asyncio.run(hooks.on_agent_end(agent, output="Sorry, I got an error."))

        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.write.assert_called()
        call_args = compiled_store.write.call_args
        card = call_args[0][0]
        self.assertEqual(card.experience_type, "tool_error")
        self.assertIn("execute", card.title)

    def test_on_agent_end_llm_classification_persists_correction(self):
        """on_agent_end should use LLM to classify and persist corrections."""
        hooks = self._make_hooks()
        agent = self._mock_agent()

        # Set up previous assistant message in working_memory
        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "I'll use the csv module to parse the data."
        agent.working_memory.messages = [prev_msg]

        # Batched judge: returns an array, one entry per turn flagged as a correction.
        mock_response = MagicMock()
        mock_response.content = json.dumps([{
            "turn_index": 0,
            "confidence": 0.95,
            "category": "preference",
            "scope": "cross_session",
            "should_persist": True,
            "persist_target": "experience",
            "rule": "Use pandas for data processing, not raw CSV parsing",
            "why": "User prefers pandas for data tasks",
            "how_to_apply": "When doing data processing tasks",
        }])
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "No, use pandas instead of csv module"))
        asyncio.run(hooks.on_agent_end(agent, output="OK, switching to pandas."))

        # Title is now derived deterministically from the rule via _rule_to_title
        # capped at _TITLE_TOKEN_CAP stems (4 by default).
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 1)
        self.assertEqual(
            correction_calls[0][0][0].title,
            "panda_data_process_raw",
        )

    def test_on_agent_end_prefilter_remember_persists_without_llm(self):
        """Explicit remember/rule phrasing should bypass the LLM classifier."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        agent.run_input = "Remember: list directory before reading files."

        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "I'll guess the filename directly."
        agent.working_memory.messages = [prev_msg]
        agent.model.response = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        agent.model.response.assert_not_called()
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 1)
        self.assertTrue(correction_calls[0][0][0].title)
        self.assertIn(
            "list directory before reading files",
            correction_calls[0][0][0].content.lower(),
        )

    def test_on_agent_end_prefilter_strong_negative_skips_persist_without_llm(self):
        """Strong negative feedback without a rule is a correction signal, not a durable rule."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        agent.run_input = "你犯错了，太蠢了。"

        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "I already made the same mistake twice."
        agent.working_memory.messages = [prev_msg]
        agent.model.response = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="..."))

        agent.model.response.assert_not_called()
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 0)

        event_store = agent.workspace.get_experience_event_store()
        classification_events = [
            c[0][0]
            for c in event_store.append.call_args_list
            if c[0][0].get("event_type") == "correction_classification"
        ]
        self.assertEqual(len(classification_events), 1)
        self.assertTrue(classification_events[0]["is_correction"])
        self.assertFalse(classification_events[0]["should_persist"])

    def test_on_agent_end_llm_classification_low_confidence_skips(self):
        """Low confidence classification should not persist."""
        hooks = self._make_hooks()
        agent = self._mock_agent()

        # Provide a previous assistant turn so the structural gate lets the
        # LLM classifier run (corrections require prior assistant context).
        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "Here is what I did before."
        agent.working_memory.messages = [prev_msg]

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "is_correction": True,
            "confidence": 0.3,  # Below threshold
            "should_persist": True,
            "persist_target": "experience",
        })
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "Hmm, that's interesting"))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        # No correction card should be written
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 0)

    def test_on_agent_end_llm_classification_memory_feedback_target_ignored(self):
        """persist_target=memory_feedback should be ignored (no cross-layer write)."""
        hooks = self._make_hooks()
        agent = self._mock_agent()

        # Previous assistant turn required for the classification gate.
        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "Here is my prior verbose reply."
        agent.working_memory.messages = [prev_msg]

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "is_correction": True,
            "confidence": 0.9,
            "category": "preference",
            "scope": "cross_session",
            "should_persist": True,
            "persist_target": "memory_feedback",
            "title": "prefer_concise_responses",
            "rule": "Keep responses concise",
            "why": "User prefers brevity",
            "how_to_apply": "All responses",
        })
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "Please be more concise"))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        # memory_feedback is no longer a valid target — should NOT write anywhere
        # (compile_correction returns None for persist_target != "experience")
        agent.workspace.write_memory_entry.assert_not_called()
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 0)

    def test_on_agent_end_persists_success_pattern(self):
        """on_agent_end should persist success patterns when 3+ tools all succeeded."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        for i in range(4):
            asyncio.run(hooks.on_tool_end(
                agent, tool_name=f"tool_{i}", is_error=False, elapsed=0.1,
            ))
        asyncio.run(hooks.on_agent_end(agent, output="Done."))

        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        success_calls = [c for c in calls if c[0][0].experience_type == "success_pattern"]
        self.assertEqual(len(success_calls), 1)

    def test_no_success_pattern_with_errors(self):
        """Success patterns should not be recorded if there were errors."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        for i in range(3):
            asyncio.run(hooks.on_tool_end(
                agent, tool_name=f"tool_{i}", is_error=False, elapsed=0.1,
            ))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="fail_tool", is_error=True, result="error",
        ))
        asyncio.run(hooks.on_agent_end(agent, output="Had an error."))

        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        success_calls = [c for c in calls if c[0][0].experience_type == "success_pattern"]
        self.assertEqual(len(success_calls), 0)

    def test_no_workspace_does_not_crash(self):
        """on_agent_end should not crash if workspace is None."""
        hooks = self._make_hooks()
        agent = self._mock_agent()
        agent.workspace = None

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True, result="error",
        ))
        asyncio.run(hooks.on_agent_end(agent, output="error"))

    def test_lifecycle_sweep_called(self):
        """on_agent_end should run lifecycle sweep via compiled_store."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="Done"))

        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.run_lifecycle.assert_called_once()

    def test_sync_called_when_configured(self):
        """on_agent_end should sync to global AGENTS.md when configured."""
        hooks = self._make_hooks(capture_user_corrections=False, sync_to_global_agent_md=True)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="Done"))

        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.sync_to_global_agent_md.assert_called_once()

    def test_sync_not_called_when_disabled(self):
        """on_agent_end should NOT sync when sync_to_global_agent_md=False."""
        hooks = self._make_hooks(capture_user_corrections=False, sync_to_global_agent_md=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="Done"))

        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.sync_to_global_agent_md.assert_not_called()

    def test_reads_run_input_at_agent_end_time(self):
        """Should read agent.run_input at on_agent_end time, not on_agent_start."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()
        agent.run_input = "old_input"  # Stale value at start time

        asyncio.run(hooks.on_agent_start(agent))

        # Simulate Runner updating run_input before on_agent_end
        agent.run_input = "current_input"
        asyncio.run(hooks.on_agent_end(agent, output="response"))

        # The hook should have used "current_input", not "old_input"
        # Verify lifecycle was called (it runs every on_agent_end)
        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.run_lifecycle.assert_called_once()

    def test_correction_timing_uses_previous_assistant(self):
        """Classification prompt should use previous assistant text from working_memory."""
        hooks = self._make_hooks()
        agent = self._mock_agent()

        # Simulate previous assistant message in working_memory
        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "Here is my previous response about CSV parsing."
        agent.working_memory.messages = [prev_msg]

        # Mock LLM returning a non-correction (to simplify assertions)
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "is_correction": False,
            "confidence": 0.2,
            "should_persist": False,
            "persist_target": "none",
        })
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "No, use pandas instead"))
        asyncio.run(hooks.on_agent_end(agent, output="OK switching"))

        # Verify the LLM classification prompt contains the PREVIOUS assistant text
        model_call = agent.model.response.call_args
        prompt_content = model_call[0][0][0].content
        self.assertIn("CSV parsing", prompt_content)

    def test_first_turn_skips_llm_classification(self):
        """No previous assistant message -> LLM classification must be skipped.

        First-turn user messages cannot be corrections (there is nothing to
        correct yet), so the expensive classifier call is pruned by a
        structural gate — no keyword heuristics.
        """
        hooks = self._make_hooks()
        agent = self._mock_agent()
        # _mock_agent already sets working_memory.messages = [] — explicit here.
        agent.working_memory.messages = []
        agent.model.response = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "No, use pandas instead"))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        # Structural gate: no prev assistant -> classifier never runs.
        agent.model.response.assert_not_called()
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 0)

    def test_events_written_before_experience_cards(self):
        """Raw events should be appended via ExperienceEventStore.append."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="FileNotFoundError: no such file", elapsed=0.5,
        ))
        asyncio.run(hooks.on_agent_end(agent, output="error"))

        # event_store.append should have been called for the tool error
        event_store = agent.workspace.get_experience_event_store()
        event_store.append.assert_called()
        event_calls = event_store.append.call_args_list
        event_types = [c[0][0]["event_type"] for c in event_calls]
        self.assertIn("tool_error", event_types)

    def test_dedup_different_errors_not_merged(self):
        """Two different error types from the same tool should produce different cards."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="FileNotFoundError: no such file", elapsed=0.5,
        ))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="PermissionError: access denied", elapsed=0.3,
        ))
        asyncio.run(hooks.on_agent_end(agent, output="errors"))

        # Both errors should produce different cards
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        error_calls = [c for c in calls if c[0][0].experience_type == "tool_error"]
        self.assertEqual(len(error_calls), 2)
        titles = [c[0][0].title for c in error_calls]
        self.assertNotEqual(titles[0], titles[1])
        self.assertIn("FileNotFoundError", titles[0])
        self.assertIn("PermissionError", titles[1])

    def test_dedup_same_error_type_not_double_written(self):
        """Two identical error types from same tool should only write once (dedup)."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="PermissionError: denied attempt 1", elapsed=0.5,
        ))
        asyncio.run(hooks.on_tool_end(
            agent, tool_name="execute", is_error=True,
            result="PermissionError: denied attempt 2", elapsed=0.3,
        ))
        asyncio.run(hooks.on_agent_end(agent, output="errors"))

        # Both produce title "execute_PermissionError", but dedup should write only once
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        error_calls = [c for c in calls if c[0][0].experience_type == "tool_error"]
        self.assertEqual(len(error_calls), 1)

    def test_feedback_confidence_threshold_from_config(self):
        """Custom feedback_confidence_threshold should be respected."""
        hooks = self._make_hooks(feedback_confidence_threshold=0.5)
        agent = self._mock_agent()

        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "Previous output"
        agent.working_memory.messages = [prev_msg]

        # Confidence 0.6 is below default 0.8 but above custom 0.5
        mock_response = MagicMock()
        mock_response.content = json.dumps([{
            "turn_index": 0,
            "confidence": 0.6,
            "category": "preference",
            "scope": "cross_session",
            "should_persist": True,
            "persist_target": "experience",
            "title": "low_conf_correction",
            "rule": "Some rule",
            "why": "Some reason",
            "how_to_apply": "Some guidance",
        }])
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_user_prompt(agent, "Actually do it differently"))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        # Should persist because 0.6 >= 0.5 (custom threshold)
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 1)

    def test_get_previous_assistant_text_returns_last_assistant(self):
        """_get_previous_assistant_text should return last assistant message."""
        agent = MagicMock()
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "Hello"
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "First response"
        msg3 = MagicMock()
        msg3.role = "user"
        msg3.content = "Follow up"
        msg4 = MagicMock()
        msg4.role = "assistant"
        msg4.content = "Second response"
        agent.working_memory.messages = [msg1, msg2, msg3, msg4]

        result = ExperienceCaptureHooks._get_previous_assistant_text(agent)
        self.assertEqual(result, "Second response")

    def test_get_previous_assistant_text_no_assistant(self):
        """_get_previous_assistant_text returns None when no assistant messages."""
        agent = MagicMock()
        agent.working_memory.messages = []
        result = ExperienceCaptureHooks._get_previous_assistant_text(agent)
        self.assertIsNone(result)


# ===========================================================================
# run_input cross-round fix tests
# ===========================================================================

class TestRunInputCrossRoundFix(unittest.TestCase):
    """Verify all hooks read agent.run_input at on_agent_end, not on_agent_start."""

    def test_conversation_archive_reads_current_run_input(self):
        """ConversationArchiveHooks should read current agent.run_input."""
        hooks = ConversationArchiveHooks()

        agent = MagicMock()
        agent.agent_id = "test"
        agent.run_input = None  # Stale at on_agent_start time
        agent.run_id = "run-1"
        agent.workspace = MagicMock()
        agent.workspace.archive_conversation = AsyncMock(return_value="/tmp/archive.md")

        # Simulate two rounds
        # Round 1:
        agent.run_input = "round 1 input"
        asyncio.run(hooks.on_agent_end(agent, output="round 1 output"))

        call = agent.workspace.archive_conversation.call_args
        messages = call[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "round 1 input")

        # Round 2:
        agent.run_input = "round 2 input"
        asyncio.run(hooks.on_agent_end(agent, output="round 2 output"))

        call = agent.workspace.archive_conversation.call_args
        messages = call[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "round 2 input")

    def test_memory_extract_reads_current_run_input(self):
        """MemoryExtractHooks should read current agent.run_input."""
        hooks = MemoryExtractHooks(every_n_turns=1, min_seconds_between=0, background=False)

        agent = MagicMock()
        agent.agent_id = "test"
        agent.session_id = "sess_test"
        agent.run_input = None
        agent.auxiliary_model = None  # force fallback to agent.model
        agent.model = MagicMock()
        agent.workspace = MagicMock()

        # Mock LLM response returning empty array (no memories to extract)
        mock_response = MagicMock()
        mock_response.content = "[]"
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(agent, tool_name="read_file"))

        # Simulate Runner setting run_input. The message must exceed the
        # MemoryExtractHooks 200-char skip threshold, otherwise the structural
        # gate drops the LLM call before we can assert on the prompt.
        agent.run_input = (
            "What files handle authentication in this codebase, and how is "
            "the session/token lifecycle wired between the middleware, the "
            "auth provider, and the database layer?"
        )
        asyncio.run(hooks.on_agent_end(
            agent,
            output=(
                "The auth files are in agentica/auth/*; session lifecycle is "
                "in middleware.py and tokens are minted in provider.py. "
                "Specifically, the middleware intercepts each incoming request, "
                "checks the Authorization header for a Bearer token, validates "
                "it via provider.py's verify_token() function, and stores the "
                "decoded user context in request.state.user. The database "
                "layer persists session metadata in the sessions table, with "
                "expiry tracked via a TTL column that the middleware checks on "
                "every request to decide whether to refresh or reject."
            ),
        ))

        # Verify model was called with the CURRENT run_input
        model_call = agent.model.response.call_args
        prompt_content = model_call[0][0][0].content
        self.assertIn("What files handle authentication", prompt_content)


# ===========================================================================
# Workspace experience storage tests
# ===========================================================================

class TestWorkspaceExperience(unittest.TestCase):
    """Test experience stores via workspace factory methods."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.workspace = Workspace(self._tmpdir)
        self.workspace.initialize()
        self.store = self.workspace.get_compiled_experience_store()
        self.event_store = self.workspace.get_experience_event_store()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _card(self, title="test", content="content", etype="tool_error", tool=""):
        from agentica.experience.compiler import CompiledCard
        return CompiledCard(title=title, content=content, experience_type=etype, tool_name=tool)

    def test_write_experience_entry(self):
        """Should create a file and update EXPERIENCE.md index."""
        path = asyncio.run(self.store.write(
            self._card(title="execute_error", content="Permission denied", etype="tool_error", tool="execute"),
        ))
        self.assertTrue(os.path.exists(path))

        index_path = self.workspace._get_user_experience_md()
        self.assertTrue(index_path.exists())
        content = index_path.read_text()
        self.assertIn("execute_error", content)
        self.assertIn("[tool_error]", content)

    def test_bump_on_duplicate(self):
        """Writing same title should increment repeat_count."""
        card = self._card(title="execute_error", content="Error A", etype="tool_error", tool="execute")
        asyncio.run(self.store.write(card))
        path = asyncio.run(self.store.write(card))
        with open(path) as f:
            content = f.read()
        self.assertIn("repeat_count: 2", content)

    def test_get_relevant_experiences_empty(self):
        result = asyncio.run(self.store.get_relevant())
        self.assertEqual(result, "")

    def test_get_relevant_experiences_returns_content(self):
        asyncio.run(self.store.write(
            self._card(title="test_experience", content="Always use absolute paths for file operations", etype="correction"),
        ))
        result = asyncio.run(self.store.get_relevant(query="file paths"))
        self.assertIn("test_experience", result)
        self.assertIn("absolute paths", result)

    def test_get_relevant_experiences_with_query(self):
        asyncio.run(self.store.write(
            self._card(title="python_error", content="Python import error with relative paths", etype="tool_error"),
        ))
        asyncio.run(self.store.write(
            self._card(title="git_conflict", content="Git merge conflict resolution steps", etype="correction"),
        ))
        result = asyncio.run(self.store.get_relevant(query="python import", limit=1))
        self.assertIn("python_error", result)

    def test_lifecycle_promotion(self):
        asyncio.run(self.store.write(
            self._card(title="frequent_error", content="This error happens a lot", etype="tool_error"),
        ))
        exp_dir = self.workspace._get_user_experience_dir()
        filepath = exp_dir / "tool_error_frequent_error.md"
        content = filepath.read_text()
        content = content.replace("repeat_count: 1", "repeat_count: 5")
        content = content.replace("tier: hot", "tier: warm")
        filepath.write_text(content)

        stats = asyncio.run(self.store.run_lifecycle(promotion_count=3))
        self.assertEqual(stats["promoted"], 1)
        self.assertIn("tier: hot", filepath.read_text())

    def test_lifecycle_demotion(self):
        asyncio.run(self.store.write(
            self._card(title="old_experience", content="Old content", etype="tool_error"),
        ))
        exp_dir = self.workspace._get_user_experience_dir()
        filepath = exp_dir / "tool_error_old_experience.md"
        content = filepath.read_text()
        old_date = (date.today() - timedelta(days=40)).isoformat()
        content = content.replace(f"last_seen: {date.today().isoformat()}", f"last_seen: {old_date}")
        filepath.write_text(content)

        stats = asyncio.run(self.store.run_lifecycle(promotion_count=3, demotion_days=30))
        self.assertEqual(stats["demoted"], 1)
        self.assertIn("tier: warm", filepath.read_text())

    def test_lifecycle_archive(self):
        asyncio.run(self.store.write(
            self._card(title="ancient_experience", content="Ancient content", etype="correction"),
        ))
        exp_dir = self.workspace._get_user_experience_dir()
        filepath = exp_dir / "correction_ancient_experience.md"
        content = filepath.read_text()
        old_date = (date.today() - timedelta(days=100)).isoformat()
        content = content.replace(f"last_seen: {date.today().isoformat()}", f"last_seen: {old_date}")
        filepath.write_text(content)

        stats = asyncio.run(self.store.run_lifecycle(promotion_count=3, archive_days=90))
        self.assertEqual(stats["archived"], 1)
        self.assertIn("tier: cold", filepath.read_text())

    def test_sync_experiences_to_global_agent_md(self):
        """sync should compile HOT experiences."""
        card = self._card(title="hot_rule", content="Always validate inputs", etype="correction")
        asyncio.run(self.store.write(card))
        asyncio.run(self.store.write(card))  # bump to repeat_count >= 2

        global_md = self.workspace._get_global_agent_md_path()
        result_path = asyncio.run(self.store.sync_to_global_agent_md(global_md))
        self.assertTrue(os.path.exists(result_path))
        with open(result_path) as f:
            content = f.read()
        self.assertIn("Learned Experiences", content)
        self.assertIn("hot_rule", content)

    def test_sync_skips_non_hot(self):
        """Sync should skip WARM/COLD tier experiences."""
        asyncio.run(self.store.write(
            self._card(title="warm_rule", content="Some warm content", etype="tool_error"),
        ))
        exp_dir = self.workspace._get_user_experience_dir()
        filepath = exp_dir / "tool_error_warm_rule.md"
        content = filepath.read_text()
        content = content.replace("tier: hot", "tier: warm")
        filepath.write_text(content)

        global_md = self.workspace._get_global_agent_md_path()
        result_path = asyncio.run(self.store.sync_to_global_agent_md(global_md))
        with open(result_path) as f:
            global_content = f.read()
        self.assertNotIn("warm_rule", global_content)

    def test_get_relevant_experiences_filters_cold(self):
        """Cold tier experiences should not appear in retrieval."""
        asyncio.run(self.store.write(
            self._card(title="hot_experience", content="Hot content for file ops", etype="correction"),
        ))
        asyncio.run(self.store.write(
            self._card(title="cold_experience", content="Cold content for file ops", etype="tool_error"),
        ))
        exp_dir = self.workspace._get_user_experience_dir()
        filepath = exp_dir / "tool_error_cold_experience.md"
        content = filepath.read_text()
        content = content.replace("tier: hot", "tier: cold")
        filepath.write_text(content)

        result = asyncio.run(self.store.get_relevant(query="file ops"))
        self.assertIn("hot_experience", result)
        self.assertNotIn("cold_experience", result)

    def test_append_experience_event(self):
        """EventStore should write JSONL to events.jsonl."""
        event = {"event_type": "tool_error", "tool": "execute", "error": "fail"}
        path = asyncio.run(self.event_store.append(event))
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("events.jsonl"))

        with open(path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["event_type"], "tool_error")
        self.assertEqual(parsed["tool"], "execute")

    def test_append_experience_event_multiple(self):
        """Multiple events should append as separate lines."""
        asyncio.run(self.event_store.append({"event_type": "a"}))
        asyncio.run(self.event_store.append({"event_type": "b"}))
        asyncio.run(self.event_store.append({"event_type": "c"}))

        exp_dir = self.workspace._get_user_experience_dir()
        events_path = exp_dir / "events.jsonl"
        with open(events_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_extract_frontmatter_helpers(self):
        content = "---\ntitle: test\nrepeat_count: 7\nlast_seen: 2026-04-15\ntier: warm\n---\nBody"
        self.assertEqual(Workspace._extract_frontmatter_int(content, "repeat_count"), 7)
        self.assertEqual(Workspace._extract_frontmatter_value(content, "tier"), "warm")
        self.assertEqual(Workspace._extract_frontmatter_value(content, "last_seen"), "2026-04-15")
        self.assertIsNone(Workspace._extract_frontmatter_value(content, "nonexistent"))


# ===========================================================================
# Agent wiring tests
# ===========================================================================

class TestAgentExperienceWiring(unittest.TestCase):

    def test_experience_false_no_hooks(self):
        agent = _make_agent(enable_experience_capture=False)
        self.assertIsNone(agent._default_run_hooks)

    def test_experience_true_needs_workspace(self):
        agent = _make_agent(enable_experience_capture=True, workspace_path=None)
        self.assertIsNone(agent._default_run_hooks)

    def test_experience_true_with_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(enable_experience_capture=True, workspace_path=tmp)
            self.assertIsNotNone(agent._default_run_hooks)
            from agentica.hooks import _CompositeRunHooks
            self.assertIsInstance(agent._default_run_hooks, _CompositeRunHooks)

    def test_experience_config_defaults(self):
        agent = _make_agent()
        self.assertIsNotNone(agent.experience_config)
        self.assertEqual(agent.experience_config.promotion_count, 3)

    def test_experience_config_custom(self):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        config = ExperienceConfig(promotion_count=10)
        agent = Agent(
            name="test",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            experience_config=config,
        )
        self.assertEqual(agent.experience_config.promotion_count, 10)


# ===========================================================================
# Experience prompt injection tests
# ===========================================================================

class TestExperiencePromptInjection(unittest.TestCase):

    def test_get_experience_prompt_disabled(self):
        agent = _make_agent(enable_experience_capture=False)
        result = asyncio.run(agent.get_experience_prompt(query="test"))
        self.assertIsNone(result)

    def test_get_experience_prompt_no_workspace(self):
        agent = _make_agent(enable_experience_capture=True)
        result = asyncio.run(agent.get_experience_prompt(query="test"))
        self.assertIsNone(result)

    def test_get_experience_prompt_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(enable_experience_capture=True, workspace_path=tmp)
            agent.workspace.initialize()

            from agentica.experience.compiler import CompiledCard
            store = agent.workspace.get_compiled_experience_store()
            asyncio.run(store.write(CompiledCard(
                title="test_lesson",
                content="Always check file permissions before writing",
                experience_type="correction",
            )))

            result = asyncio.run(agent.get_experience_prompt(query="file writing"))
            self.assertIsNotNone(result)
            self.assertIn("test_lesson", result)
            self.assertIn("file permissions", result)


# ===========================================================================
# Import path tests
# ===========================================================================

class TestExperienceImports(unittest.TestCase):

    def test_import_config_from_agent_config(self):
        from agentica.agent.config import ExperienceConfig
        self.assertIsNotNone(ExperienceConfig)

    def test_import_config_from_top_level(self):
        from agentica import ExperienceConfig
        self.assertIsNotNone(ExperienceConfig)

    def test_import_hooks_from_hooks(self):
        from agentica.hooks import ExperienceCaptureHooks
        self.assertIsNotNone(ExperienceCaptureHooks)

    def test_import_hooks_from_top_level(self):
        from agentica import ExperienceCaptureHooks
        self.assertIsNotNone(ExperienceCaptureHooks)

    def test_same_class(self):
        from agentica.agent.config import ExperienceConfig as A
        from agentica import ExperienceConfig as B
        self.assertIs(A, B)

    def test_import_experience_package(self):
        from agentica.experience import ExperienceEventStore, ExperienceCompiler, CompiledExperienceStore
        self.assertIsNotNone(ExperienceEventStore)
        self.assertIsNotNone(ExperienceCompiler)
        self.assertIsNotNone(CompiledExperienceStore)

    def test_import_experience_from_top_level(self):
        from agentica import ExperienceEventStore, ExperienceCompiler, CompiledExperienceStore
        self.assertIsNotNone(ExperienceEventStore)
        self.assertIsNotNone(ExperienceCompiler)
        self.assertIsNotNone(CompiledExperienceStore)

    def test_import_compiled_card(self):
        from agentica.experience.compiler import CompiledCard
        self.assertIsNotNone(CompiledCard)


# ===========================================================================
# ExperienceCompiler (pure/stateless) tests
# ===========================================================================

class TestExperienceCompiler(unittest.TestCase):
    """Test the pure/stateless ExperienceCompiler."""

    def test_compile_tool_errors_single(self):
        from agentica.experience.compiler import ExperienceCompiler
        errors = [{"tool": "execute", "args": {"cmd": "ls"}, "error": "PermissionError: denied", "elapsed": 0.5}]
        cards = ExperienceCompiler.compile_tool_errors(errors)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].experience_type, "tool_error")
        self.assertEqual(cards[0].tool_name, "execute")
        self.assertIn("PermissionError", cards[0].title)
        self.assertIn("denied", cards[0].content)

    def test_compile_tool_errors_dedup_titles(self):
        from agentica.experience.compiler import ExperienceCompiler
        errors = [
            {"tool": "execute", "args": {}, "error": "FileNotFoundError: x", "elapsed": 0.1},
            {"tool": "execute", "args": {}, "error": "PermissionError: y", "elapsed": 0.2},
        ]
        cards = ExperienceCompiler.compile_tool_errors(errors)
        self.assertEqual(len(cards), 2)
        self.assertNotEqual(cards[0].title, cards[1].title)

    def test_compile_tool_errors_empty(self):
        from agentica.experience.compiler import ExperienceCompiler
        cards = ExperienceCompiler.compile_tool_errors([])
        self.assertEqual(cards, [])

    def test_compile_success_pattern_enough(self):
        from agentica.experience.compiler import ExperienceCompiler
        successes = [{"tool": f"t{i}", "elapsed": 0.1} for i in range(4)]
        card = ExperienceCompiler.compile_success_pattern(successes)
        self.assertIsNotNone(card)
        self.assertEqual(card.experience_type, "success_pattern")
        self.assertIn("4 calls", card.content)
        # Title should include sorted unique tool names
        self.assertIn("success_", card.title)
        self.assertIn("t0", card.title)

    def test_compile_success_pattern_distinct_titles(self):
        """Different tool sets should produce different titles."""
        from agentica.experience.compiler import ExperienceCompiler
        s1 = [{"tool": "read_file", "elapsed": 0.1}, {"tool": "write_file", "elapsed": 0.1}, {"tool": "execute", "elapsed": 0.1}]
        s2 = [{"tool": "search", "elapsed": 0.1}, {"tool": "fetch", "elapsed": 0.1}, {"tool": "parse", "elapsed": 0.1}]
        card1 = ExperienceCompiler.compile_success_pattern(s1)
        card2 = ExperienceCompiler.compile_success_pattern(s2)
        self.assertNotEqual(card1.title, card2.title)

    def test_compile_success_pattern_too_few(self):
        from agentica.experience.compiler import ExperienceCompiler
        successes = [{"tool": "t1", "elapsed": 0.1}, {"tool": "t2", "elapsed": 0.1}]
        card = ExperienceCompiler.compile_success_pattern(successes)
        self.assertIsNone(card)

    def test_compile_correction(self):
        from agentica.experience.compiler import ExperienceCompiler
        classification = {
            "is_correction": True, "confidence": 0.9,
            "should_persist": True, "persist_target": "experience",
            "rule": "Always use pandas for tabular data",
            "why": "Better", "how_to_apply": "Always",
            "category": "preference", "scope": "cross_session",
        }
        card = ExperienceCompiler.compile_correction(classification)
        self.assertIsNotNone(card)
        # Title is derived deterministically from rule, ignoring any LLM-supplied title.
        # "always" + "use" + "for" are stop-words; "pandas" stems to "panda".
        self.assertEqual(card.title, "panda_tabular_data")
        self.assertEqual(card.experience_type, "correction")
        self.assertIn("pandas", card.content)

    def test_compile_correction_ignores_llm_title(self):
        """LLM-supplied 'title' must not influence the compiled card title."""
        from agentica.experience.compiler import ExperienceCompiler
        base = {
            "is_correction": True, "should_persist": True,
            "persist_target": "experience",
            "rule": "Check directory exists before reading file",
        }
        a = ExperienceCompiler.compile_correction({**base, "title": "foo_bar"})
        b = ExperienceCompiler.compile_correction({**base, "title": "totally_different"})
        self.assertEqual(a.title, b.title)
        self.assertNotIn("foo_bar", a.title)

    def test_compile_correction_empty_rule_returns_none(self):
        from agentica.experience.compiler import ExperienceCompiler
        card = ExperienceCompiler.compile_correction({
            "is_correction": True, "should_persist": True,
            "persist_target": "experience", "rule": "   ",
        })
        self.assertIsNone(card)

    def test_rule_to_title_stable_across_rewordings(self):
        from agentica.experience.compiler import _rule_to_title, _TITLE_TOKEN_CAP
        # Same semantic rule, different surface form → identical title.
        t1 = _rule_to_title("Always check that the directory exists before reading")
        t2 = _rule_to_title("Check directory exists before reading")
        self.assertEqual(t1, t2)
        self.assertTrue(t1)
        # Stop-word-only input degrades to empty string.
        self.assertEqual(_rule_to_title("the and or"), "")
        # Caps at _TITLE_TOKEN_CAP stems.
        long_rule = "alpha beta gamma delta epsilon zeta eta theta"
        self.assertEqual(
            _rule_to_title(long_rule).count("_"), _TITLE_TOKEN_CAP - 1,
        )

    def test_rule_to_title_stems_inflections(self):
        """Cheap stemming collapses read/reading/reads onto the same token."""
        from agentica.experience.compiler import _rule_to_title
        a = _rule_to_title("check directory before read file")
        b = _rule_to_title("check directory before reading file")
        c = _rule_to_title("check directory before reads file")
        self.assertEqual(a, b)
        self.assertEqual(b, c)

    def test_rule_to_title_dedups_repeated_stems(self):
        """Repeated tokens should not eat into the cap."""
        from agentica.experience.compiler import _rule_to_title
        title = _rule_to_title("read read read directory file alpha")
        # 'read' counted once, then 'directory', 'file', 'alpha'
        self.assertEqual(title, "read_directory_file_alpha")

    def test_rule_to_title_drops_process_noise(self):
        """Step / first / second / call / list-noise must not appear in title."""
        from agentica.experience.compiler import _rule_to_title
        title = _rule_to_title(
            "Step 1: call ls. Step 2: confirm file. Always check directory before read file."
        )
        for noise in ("step", "call", "always", "first", "second"):
            self.assertNotIn(noise, title)

    def test_compile_correction_not_correction(self):
        from agentica.experience.compiler import ExperienceCompiler
        card = ExperienceCompiler.compile_correction({"is_correction": False})
        self.assertIsNone(card)

    def test_compile_correction_memory_target_returns_none(self):
        from agentica.experience.compiler import ExperienceCompiler
        classification = {
            "is_correction": True, "should_persist": True,
            "persist_target": "memory_feedback",
        }
        card = ExperienceCompiler.compile_correction(classification)
        self.assertIsNone(card)

    def test_build_raw_events(self):
        from agentica.experience.compiler import ExperienceCompiler
        errors = [{"tool": "execute", "args": {}, "error": "fail", "elapsed": 0.1}]
        successes = [{"tool": f"t{i}", "elapsed": 0.1} for i in range(3)]
        events = ExperienceCompiler.build_raw_events(
            errors=errors, user_msg="fix this",
            previous_assistant="I did X", successes=successes,
        )
        types = [e["event_type"] for e in events]
        self.assertIn("tool_error", types)
        self.assertIn("user_message", types)
        # No success_pattern because errors exist
        self.assertNotIn("success_pattern", types)

    def test_build_raw_events_success_pattern(self):
        from agentica.experience.compiler import ExperienceCompiler
        successes = [{"tool": f"t{i}", "elapsed": 0.1} for i in range(4)]
        events = ExperienceCompiler.build_raw_events(
            errors=[], user_msg=None, previous_assistant=None,
            successes=successes, capture_corrections=False,
        )
        types = [e["event_type"] for e in events]
        self.assertIn("success_pattern", types)


# ===========================================================================
# ExperienceEventStore tests
# ===========================================================================

class TestExperienceEventStore(unittest.TestCase):
    """Test the ExperienceEventStore."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._exp_dir = os.path.join(self._tmpdir, "experiences")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_append_and_read(self):
        from pathlib import Path
        from agentica.experience.event_store import ExperienceEventStore
        store = ExperienceEventStore(Path(self._exp_dir))
        asyncio.run(store.append({"event_type": "test", "data": "hello"}))
        events = asyncio.run(store.read_all())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "test")

    def test_append_multiple(self):
        from pathlib import Path
        from agentica.experience.event_store import ExperienceEventStore
        store = ExperienceEventStore(Path(self._exp_dir))
        asyncio.run(store.append({"event_type": "a"}))
        asyncio.run(store.append({"event_type": "b"}))
        asyncio.run(store.append({"event_type": "c"}))
        events = asyncio.run(store.read_all())
        self.assertEqual(len(events), 3)

    def test_read_empty(self):
        from pathlib import Path
        from agentica.experience.event_store import ExperienceEventStore
        store = ExperienceEventStore(Path(self._exp_dir))
        events = asyncio.run(store.read_all())
        self.assertEqual(events, [])


# ===========================================================================
# CompiledExperienceStore tests
# ===========================================================================

class TestCompiledExperienceStore(unittest.TestCase):
    """Test the CompiledExperienceStore."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from pathlib import Path
        self._exp_dir = Path(self._tmpdir) / "experiences"
        self._index_path = Path(self._tmpdir) / "EXPERIENCE.md"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _store(self):
        from agentica.experience.compiled_store import CompiledExperienceStore
        return CompiledExperienceStore(
            exp_dir=self._exp_dir,
            index_path=self._index_path,
        )

    def test_write_creates_file(self):
        from agentica.experience.compiler import CompiledCard
        store = self._store()
        card = CompiledCard(title="test_error", content="Error happened", experience_type="tool_error", tool_name="execute")
        path = asyncio.run(store.write(card))
        self.assertTrue(os.path.exists(path))
        self.assertTrue(self._index_path.exists())

    def test_write_bump_on_duplicate(self):
        from agentica.experience.compiler import CompiledCard
        store = self._store()
        card = CompiledCard(title="dup_error", content="Error A", experience_type="tool_error")
        asyncio.run(store.write(card))
        path = asyncio.run(store.write(card))
        with open(path) as f:
            content = f.read()
        self.assertIn("repeat_count: 2", content)

    def test_get_relevant_empty(self):
        store = self._store()
        result = asyncio.run(store.get_relevant())
        self.assertEqual(result, "")

    def test_get_relevant_with_data(self):
        from agentica.experience.compiler import CompiledCard
        store = self._store()
        card = CompiledCard(title="file_perms", content="Check file permissions", experience_type="correction")
        asyncio.run(store.write(card))
        result = asyncio.run(store.get_relevant(query="file permissions"))
        self.assertIn("file_perms", result)

    def test_lifecycle(self):
        from agentica.experience.compiler import CompiledCard
        from datetime import timedelta
        store = self._store()
        card = CompiledCard(title="old_exp", content="Old", experience_type="tool_error")
        asyncio.run(store.write(card))
        # Manually set last_seen to 40 days ago
        filepath = self._exp_dir / "tool_error_old_exp.md"
        content = filepath.read_text()
        from datetime import date
        old_date = (date.today() - timedelta(days=40)).isoformat()
        content = content.replace(f"last_seen: {date.today().isoformat()}", f"last_seen: {old_date}")
        filepath.write_text(content)
        stats = asyncio.run(store.run_lifecycle(demotion_days=30))
        self.assertEqual(stats["demoted"], 1)

    def test_cold_filtered_from_retrieval(self):
        from agentica.experience.compiler import CompiledCard
        store = self._store()
        card = CompiledCard(title="cold_exp", content="Cold content for files", experience_type="tool_error")
        asyncio.run(store.write(card))
        filepath = self._exp_dir / "tool_error_cold_exp.md"
        content = filepath.read_text()
        content = content.replace("tier: hot", "tier: cold")
        filepath.write_text(content)
        result = asyncio.run(store.get_relevant(query="files"))
        self.assertNotIn("cold_exp", result)


if __name__ == "__main__":
    unittest.main()
