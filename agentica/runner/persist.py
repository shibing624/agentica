# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner persistence for incomplete turns and tool-call transcripts
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)


from agentica.utils.log import logger
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.usage import split_prompt_usage
from agentica.run_response import RunEvent, RunResponse, ToolCallInfo
from agentica.memory import AgentRun
from agentica.memory.session_log import iso_timestamp

if TYPE_CHECKING:
    from agentica.agent import Agent



class PersistMixin:
    """Extracted Runner methods."""

    agent: Any

    @staticmethod
    def _tool_records_from_messages(
        messages: List[Message],
        *,
        fallback_compacted: bool = False,
        fallback_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role != "tool" or not msg.tool_name:
                continue
            record = {
                "tool_call_id": msg.tool_call_id,
                "tool_name": msg.tool_name,
                "tool_args": msg.tool_args,
                "content": msg.content,
                "tool_call_error": msg.tool_call_error or False,
                "metrics": msg.metrics if msg.metrics else {},
            }
            if fallback_compacted:
                record["fallback_compacted"] = True
                record["replay"] = False
                if fallback_model:
                    record["fallback_model"] = fallback_model
            records.append(record)
        return records

    @staticmethod
    def _provider_replay_meta(message: Message) -> Dict[str, Any]:
        """Return provider state required for faithful same-provider replay."""
        meta: Dict[str, Any] = {}
        if message.provider_data is not None:
            meta["provider_data"] = message.provider_data
        if message.provider_checkpoint is not None:
            meta["provider_checkpoint"] = message.provider_checkpoint
        if message.reasoning_content is not None:
            meta["reasoning_content"] = message.reasoning_content
        if message.finish_reason is not None:
            meta["finish_reason"] = message.finish_reason
        if message.metrics:
            meta["metrics"] = message.metrics
        return meta

    @staticmethod
    def _persist_turn_user_message(
        agent: "Agent",
        message: Any,
        user_messages: List[Message],
    ) -> None:
        """Append this turn's ``user`` entry, exactly once.

        Requested from three places — the first in-turn flush, the end-of-turn
        write and the interrupted-turn path — because whichever runs first must
        put the question on disk BEFORE that turn's assistant/tool entries, or
        the replay would show the answer before the question. The log's
        turn bookkeeping (``SessionLog.begin_turn``) makes the later calls
        no-ops.
        """
        session_log = agent._session_log
        if session_log is None or session_log._turn_user_uuid is not None:
            return
        text: Optional[str] = None
        if isinstance(message, str):
            text = message
        elif isinstance(message, Message):
            text = message.content if isinstance(message.content, str) else str(message.content)
        if not text:
            return
        meta = PersistMixin._provider_replay_meta(user_messages[-1]) if user_messages else {}
        session_log._turn_user_uuid = session_log.append("user", text, **meta)

    @staticmethod
    def _should_flush_turn_tool_rounds(
        agent: "Agent",
        messages: Any,
        loop_state: "LoopState",
        fallback_transaction_model: Any,
    ) -> bool:
        """Whether in-turn persistence applies to the turn being run.

        Skipped when:
        - there is no session log, or ``messages`` was pre-built (those runs
          manage their own history — the end-of-turn write skips them too)
        - a compression stage collapsed the context: the ``num_input_messages``
          prefix boundary is gone, so "this turn's messages" can no longer be
          sliced out reliably
        - the turn is inside a fallback transaction: its tool results become
          non-replayable ``tool_audit`` entries, and that is only decided at
          turn end — writing them early as replayable ``tool`` entries would
          corrupt the replay
        """
        return (
            agent._session_log is not None
            and messages is None
            and not loop_state.context_collapsed
            and fallback_transaction_model is None
        )

    @staticmethod
    def _flush_turn_tool_rounds(
        agent: "Agent",
        message: Any,
        user_messages: List[Message],
        turn_messages: List[Message],
    ) -> None:
        """Persist the tool rounds finished so far, without waiting for turn end.

        The whole turn used to be written only at the end, so a SIGKILL / OOM
        kill / power loss lost every round of an agentic turn that had been
        running for minutes. Writing each finished round costs one append per
        round and leaves the end-of-turn write as a backfill.

        Only *answered* rounds are written (``_drop_unanswered_tool_calls``): an
        ``assistant(tool_calls)`` entry with no tool result on disk is exactly
        the shape a provider rejects on replay.

        Best-effort: the end-of-turn write still persists everything, so a
        failure here must never take the running turn down with it.
        """
        if agent._session_log is None:
            return
        try:
            answered = PersistMixin._drop_unanswered_tool_calls(
                [m for m in turn_messages if isinstance(m, Message)]
            )
            if not any(
                (m.role == "assistant" and m.tool_calls) or m.role == "tool"
                for m in answered
            ):
                return
            PersistMixin._persist_turn_user_message(agent, message, user_messages)
            PersistMixin._persist_assistant_tool_calls(
                agent,
                messages=answered,
                tool_records=PersistMixin._tool_records_from_messages(answered),
            )
        except Exception:
            logger.warning("in-turn session-log flush failed", exc_info=True)

    @staticmethod
    def _persist_assistant_tool_calls(
        agent: "Agent",
        messages: Optional[List[Message]] = None,
        tool_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist the turn's assistant tool-call messages to the session log.

        The session log otherwise records only ``user`` -> ``tool`` results ->
        final ``assistant`` text. The intermediate assistant messages that
        CARRY the ``tool_calls`` are never written, so on resume the tool
        results become orphaned (a ``tool`` message with no preceding assistant
        holding ``tool_calls``) and the provider rejects the replay with
        "messages with role 'tool' must be a response to a preceding message
        with 'tool_calls'".

        A single agentic turn may issue tool calls across several assistant
        rounds (e.g. ``read_file`` then ``grep``). For each round we must log
        ``assistant(tool_calls)`` immediately followed by its ``tool`` result,
        preserving the exact interleaving — OpenAI-compatible providers require
        every ``tool`` message to immediately follow the assistant message that
        requested it. Grouping all assistants before all tools (the previous
        implementation) re-introduced the same 400 on resume for any multi-round
        tool turn.

        We walk ``run_response.messages`` (this turn only, see the
        ``num_input_messages`` slice in ``_run_impl``) in order: for each
        assistant-with-tool_calls we log the assistant, and for each
        ``role="tool"`` message we log the matching tool result (rich metadata
        taken from ``run_response.tools`` by id). ``_build_messages`` already
        lists ``tool_calls`` / ``tool_call_id`` in its replay fields, so the
        replay reconstructs a valid, interleaved sequence.

        Called once per turn (before the final assistant text is logged) so the
        JSONL order is
        ``user -> assistant(tool_calls) -> tool -> assistant(tool_calls) -> tool -> ... -> assistant(text)``.

        Idempotent within a turn: rounds already written by an in-turn flush
        (``_flush_turn_tool_rounds``) are recorded on the session log by
        ``begin_turn`` bookkeeping and skipped here, so this stays a backfill
        instead of a second copy. ``messages`` / ``tool_records`` let the
        mid-turn caller pass the in-flight turn slice; both default to
        ``run_response``, which only exists once the turn is over.
        """
        if agent._session_log is None:
            return
        session_log = agent._session_log
        _msgs = messages if messages is not None else (agent.run_response.messages or [])
        _records = tool_records if tool_records is not None else (agent.run_response.tools or [])
        tool_by_id = {
            tc.get("tool_call_id"): tc
            for tc in _records
        }
        _functions = (agent.model.functions or {}) if agent.model else {}
        for msg in _msgs:
            if not isinstance(msg, Message):
                continue
            if msg.role == "assistant" and msg.tool_calls:
                _ids = tuple(
                    str(tc.get("id") or "") for tc in msg.tool_calls if isinstance(tc, dict)
                )
                _round_key: Any = _ids if _ids and all(_ids) else ("obj", id(msg))
                if _round_key in session_log._turn_written_tool_call_rounds:
                    continue
                session_log._turn_written_tool_call_rounds.add(_round_key)
                _text = msg.content if isinstance(msg.content, str) else ""
                session_log.append(
                    "assistant",
                    _text,
                    tool_calls=msg.tool_calls,
                    **PersistMixin._provider_replay_meta(msg),
                )
            elif msg.role == "tool":
                if msg.tool_call_id and msg.tool_call_id in session_log._turn_written_tool_call_ids:
                    continue
                if msg.tool_call_id:
                    session_log._turn_written_tool_call_ids.add(msg.tool_call_id)
                _tc = tool_by_id.get(msg.tool_call_id)
                if _tc is not None:
                    _tool_content = _tc.get("content", "") or ""
                    if len(_tool_content) > 2000:
                        _tool_content = _tool_content[:2000] + "\n... [truncated]"
                    _origin_meta: Dict[str, Any] = {}
                    _fn = _functions.get(_tc.get("tool_name", ""))
                    if _fn is not None and _fn.origin is not None:
                        _origin_meta["origin_type"] = _fn.origin.type
                        if _fn.origin.provider_name:
                            _origin_meta["origin_provider_name"] = _fn.origin.provider_name
                        if _fn.origin.agent_name:
                            _origin_meta["origin_agent_name"] = _fn.origin.agent_name
                        if _fn.origin.source_tool_name:
                            _origin_meta["origin_source_tool_name"] = _fn.origin.source_tool_name
                    session_log.append(
                        "tool_audit" if _tc.get("replay") is False else "tool",
                        _tool_content,
                        tool_name=_tc.get("tool_name", ""),
                        tool_call_id=_tc.get("tool_call_id", ""),
                        is_error=_tc.get("tool_call_error", False),
                        fallback_compacted=_tc.get("fallback_compacted", False),
                        fallback_model=_tc.get("fallback_model"),
                        replay=_tc.get("replay", True),
                        **PersistMixin._provider_replay_meta(msg),
                        **_origin_meta,
                    )
                else:
                    # Tool message without matching FunctionCall metadata: log a
                    # minimal entry so resume still has a valid assistant->tool pair.
                    session_log.append(
                        "tool",
                        msg.content if isinstance(msg.content, str) else "",
                        tool_call_id=msg.tool_call_id or "",
                        **PersistMixin._provider_replay_meta(msg),
                    )

    @staticmethod
    def _strip_tool_artifacts(msgs: List[Message], *, drop_system: bool = False) -> List[Message]:
        """Drop tool-call/tool-result artifacts (OpenAI *and* Anthropic wire formats).

        Keeps only plain user/assistant text so history recorded under one
        provider can be replayed on another. Handles both OpenAI-style
        (role="tool" + assistant.tool_calls) and Anthropic-style (list content
        blocks with tool_use/tool_result) encodings. Used to recover from
        cross-provider tool-call format mismatches — see
        _sanitize_tool_history_after_error.
        """
        from agentica.agent.history_filter import strip_all_tool_artifacts

        return strip_all_tool_artifacts(msgs, drop_system=drop_system)

    @staticmethod
    def _check_session_log_trajectory(agent: "Agent", turn_start_uuid: Optional[str]) -> None:
        """Verify the log just written projects back to this turn's live trajectory.

        The log is rebuilt at the end of a turn from ``run_response.messages``
        (see ``_persist_assistant_tool_calls``); a regression in that rebuild is
        invisible until a later ``/resume`` is rejected by the provider. This
        checks the invariant while both sides are still in hand.

        Deliberately weak in production: DEBUG-only, warning-only, and wrapped
        so it can never become a new crash source — log fidelity is an
        observability concern and must not kill a live conversation. Its real
        job is to be assertable in tests (``assert_trajectory_equivalent``).
        """
        if not logger.isEnabledFor(10):  # logging.DEBUG == 10
            return
        try:
            from agentica.memory.session_log import assert_trajectory_equivalent

            session_log = agent._session_log
            if session_log is None:
                return
            # A fallback-compacted turn writes its tool results as ``tool_audit``
            # entries, which ``_build_messages`` deliberately does not replay —
            # log and live trajectory legally differ, so there is nothing to check.
            if any(
                isinstance(tc, dict) and tc.get("replay") is False
                for tc in (agent.run_response.tools or [])
            ):
                logger.debug("session-log trajectory check skipped: fallback-compacted tool audit turn")
                return
            derived = session_log.derive_messages(
                model=agent.model.id if agent.model is not None else None,
                since_uuid=turn_start_uuid,
            )
            live = [m for m in (agent.run_response.messages or []) if isinstance(m, Message)]
            divergence = assert_trajectory_equivalent(derived, live)
            if divergence is not None:
                logger.warning(
                    "session log does not match the turn sent to the provider "
                    "(resume would replay a different trajectory): %s",
                    divergence,
                )
            else:
                logger.debug(
                    "session-log trajectory check ok (%d derived messages)", len(derived)
                )
        except Exception:
            logger.debug("session-log trajectory check failed", exc_info=True)

    @staticmethod
    def _sanitize_tool_history_after_error(agent: "Agent", messages: List[Message]) -> None:
        """Strip tool-call artifacts from history after a tool-history API error.

        Recovery path for resuming a session whose history was recorded under
        a different model provider (e.g. Claude) on a provider that rejects
        the resulting 'tool' role messages (e.g. "Messages with role 'tool'
        must be a response to a preceding message with 'tool_calls'"). Only
        user/assistant text is needed going forward, so tool_calls/tool
        results are dropped entirely — both from working-memory-backed
        history (so future turns stay clean) and the in-flight ``messages``
        list (so the immediate retry succeeds).
        """
        for run in agent.working_memory.runs:
            if run.response and run.response.messages:
                run.response.messages = PersistMixin._strip_tool_artifacts(run.response.messages, drop_system=True)
        messages[:] = PersistMixin._strip_tool_artifacts(messages)

    def save_run_response_to_file(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        save_response_to_file: Optional[str] = None,
    ) -> None:
        _save_path = save_response_to_file
        if _save_path is None or self.agent.run_response is None:
            return
        message_str = None
        if message is not None:
            if isinstance(message, str):
                message_str = message
            else:
                logger.warning("Did not use message in output file name: message is not a string")
        try:
            fn = _save_path.format(name=self.agent.name, message=message_str)
            fn_path = Path(fn)
            if not fn_path.parent.exists():
                fn_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(self.agent.run_response.content, str):
                fn_path.write_text(self.agent.run_response.content)
            else:
                fn_path.write_text(json.dumps(self.agent.run_response.content, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(
                f"Failed to save output to file '{save_response_to_file}': {e} "
                f"[agent={self.agent.identifier}, run_id={self.agent.run_id}]"
            )

    def _aggregate_metrics_from_run_messages(self, messages: List[Message]) -> Dict[str, Any]:
        aggregated_metrics: Dict[str, Any] = defaultdict(list)
        for m in messages:
            if m.role == "assistant" and m.metrics is not None:
                for k, v in m.metrics.items():
                    aggregated_metrics[k].append(v)
        return aggregated_metrics

    def generic_run_response(
        self, content: Optional[str] = None, event: RunEvent = RunEvent.run_response,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> RunResponse:
        """Build a RunResponse for a mid-run event.

        ``tool_call`` is the raw tool-call dict a tool event is about; it is
        surfaced as the typed ``RunResponse.tool_call`` so consumers never have to
        infer the subject from ``tools`` by position.
        """
        return RunResponse(
            run_id=self.agent.run_id,
            agent_id=self.agent.agent_id,
            content=content,
            tools=self.agent.run_response.tools,
            images=self.agent.run_response.images,
            videos=self.agent.run_response.videos,
            model=self.agent.run_response.model,
            messages=self.agent.run_response.messages,
            reasoning_content=self.agent.run_response.reasoning_content,
            extra_data=self.agent.run_response.extra_data,
            event=event.value,
            tool_call=ToolCallInfo.from_dict(tool_call) if tool_call else None,
        )

    @staticmethod
    def _drop_unanswered_tool_calls(msgs: List[Message]) -> List[Message]:
        """Drop tool-call rounds whose results never arrived.

        An assistant message carrying ``tool_calls`` must be followed by one
        ``tool`` message per ``tool_call_id``. A turn that died between the
        request and its results would otherwise persist a shape the provider
        rejects on *every* later request in the session, not just this one.
        """
        def tool_result_ids(message: Message) -> set:
            if message.role == "tool" and message.tool_call_id:
                return {message.tool_call_id}
            if message.role != "user" or not isinstance(message.content, list):
                return set()
            return {
                block.get("tool_use_id")
                for block in message.content
                if isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id")
            }

        def filter_tool_result_message(message: Message, kept_ids: set) -> Optional[Message]:
            if message.role == "tool":
                if message.tool_call_id in kept_ids:
                    return message
                return None
            if message.role != "user" or not isinstance(message.content, list):
                return message
            blocks: List[Any] = []
            for block in message.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") not in kept_ids
                ):
                    continue
                blocks.append(block)
            if not blocks:
                return None
            if len(blocks) == len(message.content):
                return message
            return message.model_copy(update={"content": blocks})

        answered = set()
        for m in msgs:
            answered.update(tool_result_ids(m))
        kept: List[Message] = []
        kept_ids: set = set()
        for m in msgs:
            if m.role == "assistant" and m.tool_calls:
                ids = {tc.get("id") for tc in m.tool_calls if isinstance(tc, dict)}
                if not ids.issubset(answered):
                    continue
                kept_ids |= ids
            kept.append(m)
        filtered: List[Message] = []
        for m in kept:
            filtered_msg = filter_tool_result_message(m, kept_ids)
            if filtered_msg is not None:
                filtered.append(filtered_msg)
        return filtered

    def _try_persist_incomplete_turn(self, *args: Any, **kwargs: Any) -> None:
        """Best-effort ``_persist_incomplete_turn``: keeping history is never
        worth masking the cancel or failure that is already propagating."""
        try:
            self._persist_incomplete_turn(*args, **kwargs)
        except Exception:
            logger.warning("incomplete-turn persistence failed", exc_info=True)

    def _persist_incomplete_turn(
        self,
        agent,
        message: Any,
        messages: Any,
        user_messages: List[Message],
        system_message: Optional[Message],
        messages_for_model: List[Message],
        num_input_messages: int,
        model_response: Any,
        loop_state: "LoopState",
        input_message_ids: set,
        *,
        marker: str,
        finish_reason: str,
    ) -> None:
        """Preserve a turn that ended early instead of discarding it.

        Used by both terminal paths that are not natural completion: a user
        cancel and a failed run. Mirrors the success-path memory + session-log
        persistence — the user question plus whatever the assistant produced
        before it stopped are kept as a completed Q&A turn with ``marker``
        appended, so a follow-up "continue" (and ``/retry``) still see the
        instruction that turn carried.

        Only applies to message-based runs (``messages`` is None) — the CLI
        path; pre-built ``messages`` runs manage their own history. A turn that
        died before message assembly has no ``user_messages`` and nothing worth
        keeping.
        """
        if messages is not None or not user_messages:
            return
        # Capture the partial streamed content as the authoritative answer.
        if model_response.content:
            agent.run_response.content = model_response.content
        partial = agent.run_response.content or ""
        persisted = (f"{partial}\n\n{marker}") if partial else marker
        agent.run_response.content = persisted

        # The model layer appends the assistant message only AFTER the stream
        # completes, so on a mid-stream cancel it's absent from
        # messages_for_model. Patch the turn's last assistant message if one
        # exists (cancel during tool exec / between turns), else synthesize one
        # (cancel mid-stream) so /history surfaces the partial + marker.
        if loop_state.context_collapsed:
            turn_msgs = [
                m for m in messages_for_model
                if id(m) not in input_message_ids and m.role != "system"
            ]
        else:
            turn_msgs = list(messages_for_model[num_input_messages:])
        turn_msgs = self._drop_unanswered_tool_calls(turn_msgs)
        last_asst = None
        for _m in reversed(turn_msgs):
            if isinstance(_m, Message) and _m.role == "assistant":
                last_asst = _m
                break
        synthesized: Optional[Message] = None
        if last_asst is not None and last_asst.tool_calls:
            synthesized = Message(role="assistant", content=persisted)
            turn_msgs.append(synthesized)
        elif last_asst is not None:
            last_asst.content = persisted
        else:
            synthesized = Message(role="assistant", content=persisted)
            turn_msgs.append(synthesized)

        # ``get_messages_from_last_n_runs()`` builds the next prompt from
        # ``AgentRun.response.messages``, not from ``working_memory.messages``.
        # Keep the interrupted turn in that canonical history source so a
        # follow-up such as "continue" sees the partial answer it must resume.
        if loop_state.context_collapsed:
            run_messages = self._drop_unanswered_tool_calls(
                [m for m in messages_for_model if m.role != "system"]
            )
            if synthesized is not None:
                run_messages.append(synthesized)
        else:
            run_messages = user_messages + turn_msgs
        if system_message is not None:
            run_messages.insert(0, system_message)
        agent.run_response.messages = run_messages

        # working_memory so /history shows this exchange.
        if system_message is not None:
            agent.working_memory.add_system_message(
                system_message,
                system_message_role=agent.prompt_config.system_message_role,
            )
        agent.working_memory.add_messages(messages=(user_messages + turn_msgs))
        agent_run = AgentRun(response=agent.run_response)
        if user_messages:
            agent_run.message = user_messages[0]
            agent_run.messages = list(user_messages)
        if loop_state.context_collapsed:
            # Mirror the success path: the run must carry the whole surviving
            # conversation before the runs it supersedes are dropped, or the
            # cancel would take the entire history down with it.
            agent.working_memory.runs.clear()
        agent.working_memory.add_run(agent_run)

        # session log so /resume restores this turn.
        if agent._session_log is not None:
            # No-op when an in-turn flush already wrote the question.
            PersistMixin._persist_turn_user_message(agent, message, user_messages)
            # Log assistant tool-call messages AND their tool results in the
            # exact interleaved order so /resume rebuilds a valid
            # assistant(tool_calls)->tool sequence instead of orphaned tools
            # (skipping the rounds an in-turn flush already persisted).
            self._persist_assistant_tool_calls(agent)
            _assistant_meta = PersistMixin._provider_replay_meta(last_asst) if last_asst is not None else {}
            _assistant_meta["finish_reason"] = finish_reason
            agent._session_log.append("assistant", persisted, **_assistant_meta)

    @staticmethod
    def _trace_event(
        agent: "Agent", name: str, at: Optional[float] = None, **payload: Any
    ) -> None:
        """Append a SessionLog lifecycle event. No-op without a session log.

        ``at`` is a ``time.time()`` reading of when the event actually happened,
        for the phases that are only known to be over once the stream has moved
        on from them.
        """
        slog = agent._session_log
        if slog is None:
            return
        slog.append_event(
            name, timestamp=iso_timestamp(at) if at is not None else None, **payload
        )

    @staticmethod
    def _trace_session_prelude(agent: "Agent", messages: List[Message]) -> None:
        """Record model / tool table / system prompt for the Trace page.

        Called before every request; the SessionLog drops the repeat unless the
        content actually changed, so a mid-session profile switch is visible
        and a steady session pays one row set.
        """
        slog = agent._session_log
        if slog is None:
            return
        system_prompt = ""
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else ""
                break
        model = agent.model
        slog.append_trace_prelude(
            model=model.id if model is not None else None,
            provider=model.provider if model is not None else None,
            context_window=model.context_window if model is not None else None,
            tools=sorted((model.functions or {}).keys()) if model is not None else [],
            system_prompt=system_prompt,
        )

    @staticmethod
    def _tool_call_id_and_name(tc: Dict[str, Any]) -> tuple[str, str]:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        call_id = str(tc.get("id") or tc.get("tool_call_id") or "")
        name = str(tc.get("tool_name") or fn.get("name") or "")
        return call_id, name

    @staticmethod
    def _trace_request_segments(
        agent: "Agent",
        messages: List[Message],
        phase_ends: Optional[Dict[str, float]] = None,
    ) -> None:
        """Write thinking / text / tool_call completion events for this request.

        Each event marks the *end* of its phase — the analyzer chains them, so
        one row's timestamp is the next row's start and the first starts at
        ``request_begin``. ``phase_ends`` carries the moments the streaming loop
        observed (``"thinking"`` when the last reasoning token arrived,
        ``"text"`` when the last content token did); anything missing falls back
        to now, which is also the whole story for a non-streamed call where the
        response arrives in one piece.

        The rows are written in the order those moments happened, not in a fixed
        thinking → text → tool_call order: the chain would otherwise run
        backwards for a model that answers before it calls a tool, and a
        negative segment is drawn as a flat edge.
        """
        assistant = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                assistant = msg
                break
        if assistant is None:
            return
        ends = phase_ends or {}
        # Tool call arguments are only complete when the stream is: whatever
        # time we are reading now is that moment.
        now = time.time()
        marks: List[tuple[float, str, Dict[str, Any]]] = []

        reasoning = assistant.reasoning_content or assistant.thinking
        if isinstance(reasoning, str) and reasoning.strip():
            marks.append((ends.get("thinking", now), "thinking", {}))
        content = assistant.content if isinstance(assistant.content, str) else ""
        if content.strip():
            marks.append((ends.get("text", now), "text", {}))
        for tc in assistant.tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_id, name = PersistMixin._tool_call_id_and_name(tc)
            if not call_id:
                continue
            marks.append((now, "tool_call", {"tool_call_id": call_id, "tool_name": name}))

        marks.sort(key=lambda m: m[0])
        for at, name, payload in marks:
            PersistMixin._trace_event(agent, name, at=at, **payload)

    @staticmethod
    def _trace_token_usage(agent: "Agent", messages: List[Message]) -> None:
        assistant = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                assistant = msg
                break
        metrics = (assistant.metrics if assistant is not None else None) or {}

        def _num(value: Any) -> int:
            if isinstance(value, list):
                return int(sum(x for x in value if isinstance(x, (int, float))))
            if isinstance(value, (int, float)):
                return int(value)
            return 0

        details = metrics.get("prompt_tokens_details") if isinstance(metrics.get("prompt_tokens_details"), dict) else {}
        # Same split the cost path uses: the three prompt buckets must be
        # disjoint or the cached prefix is counted (and priced) twice.
        prompt_tokens = _num(metrics.get("prompt_tokens")) or _num(metrics.get("input_tokens"))
        fresh_input, cache_read, cache_write = split_prompt_usage(prompt_tokens, details)
        if not (cache_read or cache_write):
            cache_read = _num(metrics.get("cache_read_tokens") or metrics.get("cached_tokens"))
            cache_write = _num(metrics.get("cache_write_tokens"))
            fresh_input = max(prompt_tokens - cache_read - cache_write, 0)
        output = _num(metrics.get("output_tokens"))
        total = _num(metrics.get("total_tokens")) or (prompt_tokens + output)
        PersistMixin._trace_event(
            agent,
            "token_usage",
            request={
                "input": fresh_input,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "output": output,
                "total": total,
            },
        )
