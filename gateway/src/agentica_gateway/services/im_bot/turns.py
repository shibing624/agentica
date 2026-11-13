# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM turn submission, CLI AttachServer execution, Gateway streaming, and file push.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.agent.approvals import ApprovalDecision, PendingApproval
from agentica.cli.approvals import visible_approval_decisions
from agentica.peers import PeerInfo, PeerMessageRefused
from agentica.tools.background_processes import touch_activity_stamp
from agentica.utils.log import logger

from ...channels.base import Channel, ChannelType, Message
from .. import live_turn
from ..agent_service import ChatResult
from ..im_files import collect_auto_push_files, format_file_size
from .attach import _AttachRpcClient, _read_attach_token
from .commands import (
    ACK_DELAY_SECS,
    CHUNK_CHARS,
    ImChatState,
    QueuedTurn,
    _APPROVAL_LABELS,
    _fmt_elapsed,
    _short_dir,
)


def _pkg():
    return sys.modules["agentica_gateway.services.im_bot"]


class ImTurnsMixin:
    """Turn queuing, live CLI attach execution, Gateway local streaming, and replies."""

    async def _submit_turn(
        self,
        state: ImChatState,
        message: Message,
        *,
        text_override: Optional[str] = None,
    ) -> None:
        # Capture immutable target at submission time (OpenBitFun RemoteBotTarget rule)
        target_work_dir = state.work_dir
        target_session_id = state.session_id
        uid = self._user_id()

        # Fetch inbound media if any
        fetched_media = []
        wanted_media = message.metadata.get("media") or []
        if wanted_media and self._channel_manager is not None:
            ch = self._channel_manager.get_channel(message.channel)
            if ch is not None:
                fetched_media = await ch.fetch_media(message)

        text = (text_override if text_override is not None else (message.content or "")).strip()
        if not text and not fetched_media:
            if wanted_media:
                await self._reply(state, "没能下载这条图片/语音，请再发一次，或配一句文字。")
            return

        if not text and fetched_media:
            kind = fetched_media[0].kind
            text = {
                "image": "请看这条图片。",
                "voice": "请听这条语音。",
                "video": "请看这条视频。",
            }.get(kind, "请查看这条媒体。")

        peer = self.find_live_cli_peer(target_session_id) if not fetched_media else None
        if peer is not None:
            task = asyncio.create_task(
                self._run_attached_cli_turn(state, peer, target_work_dir, target_session_id, text)
            )
            self._turn_tasks[target_session_id] = task
            return

        # Gateway local execution: steer if already running on this session
        if (
            not fetched_media
            and self._agent_service is not None
            and self._agent_service.is_session_active(target_session_id, owner=uid)
        ):
            q = self._queued_turns.get(target_session_id)
            if (q is None or q.empty()) and self._agent_service.steer_session(
                target_session_id, text, owner=uid,
            ):
                logger.info(f"[im_bot] steered follow-up into running session {target_session_id}")
                return

        q = self._queued_turns.get(target_session_id)
        if q is None:
            q = asyncio.Queue()
            self._queued_turns[target_session_id] = q
        await q.put(QueuedTurn(state, target_work_dir, target_session_id, text, fetched_media))

        worker = self._turn_tasks.get(target_session_id)
        if worker is None or worker.done():
            self._turn_tasks[target_session_id] = asyncio.create_task(
                self._drain_session_queue(target_session_id, q)
            )

    async def deliver_background_report(self, owner: str, session_id: str, report: str) -> bool:
        """Run a finished background command's report in the IM chats bound to
        this session, so the phone gets the conclusion without asking.

        False — the caller then runs it on the web — when no connected chat of
        this gateway is on that session: IM chats belong to the home account,
        and a gateway that does not serve IM has their bindings but no channel.
        """
        if self._agent_service is None or self._channel_manager is None or owner != self._user_id():
            return False
        chats = [
            st for st in self._states.values()
            if st.session_id == session_id and self._channel_connected(st.channel)
        ]
        if not chats:
            return False
        peer = self.find_live_cli_peer(session_id)
        if peer is not None:
            # The terminal owns this session now; a gateway turn would write
            # the same log alongside it. Its agent picks the report up instead.
            await self._hand_report_to_cli(chats, peer, report)
            return True
        q = self._queued_turns.setdefault(session_id, asyncio.Queue())
        await q.put(QueuedTurn(
            chats[0], chats[0].work_dir, session_id, report, automated=True, also_notify=chats[1:],
        ))
        worker = self._turn_tasks.get(session_id)
        if worker is None or worker.done():
            self._turn_tasks[session_id] = asyncio.create_task(self._drain_session_queue(session_id, q))
        return True

    async def deliver_scheduled_run(
        self, owner: str, session_id: str, text: str, input_meta: Dict[str, Any],
    ) -> Optional[Any]:
        """Run a scheduled task's turn in the IM chats bound to this conversation.

        Queued behind whatever those chats are doing, answered on the phone,
        and logged like any other scheduled run. Returns the run's
        ``ChatResult``, or None — the caller then runs it on the web — when no
        connected chat of this gateway is on the conversation.
        """
        if self._agent_service is None or self._channel_manager is None or owner != self._user_id():
            return None
        chats = [
            st for st in self._states.values()
            if st.session_id == session_id and self._channel_connected(st.channel)
        ]
        if not chats:
            return None
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        q = self._queued_turns.setdefault(session_id, asyncio.Queue())
        await q.put(QueuedTurn(
            chats[0], chats[0].work_dir, session_id, text, automated=True, also_notify=chats[1:],
            input_meta=input_meta, done=done,
        ))
        worker = self._turn_tasks.get(session_id)
        if worker is None or worker.done():
            self._turn_tasks[session_id] = asyncio.create_task(self._drain_session_queue(session_id, q))
        return await done

    def _channel_connected(self, channel: ChannelType) -> bool:
        ch = self._channel_manager.get_channel(channel) if self._channel_manager is not None else None
        return ch is not None and ch.is_connected

    async def _hand_report_to_cli(self, chats: List[ImChatState], peer: PeerInfo, report: str) -> None:
        """Give a finished job's report to the terminal that owns the session.

        Attach first: it returns the terminal's answer, so the phone learns the
        conclusion instead of a note that something was handed to a terminal it
        cannot see. The peer mailbox stays as the fallback for a terminal whose
        socket is not usable yet — it cannot tell us anything, which is exactly
        why it is second.
        """
        from agentica.cli.attach import prompt_live_session

        # The default 620s window is deliberate here: the report's answer is
        # what the phone is waiting for, so the claim and this session's
        # delivery slot stay held while the terminal works.
        result = await asyncio.to_thread(prompt_live_session, peer, report, automated=True)
        if result is not None:
            answer = str(result.get("agenticaAnswer") or "").strip()
            if answer:
                notice = f"🔔 后台任务已结束（终端 [{peer.name}] 处理）：\n{answer}"
            elif result.get("stopReason") == "cancelled":
                notice = f"🔔 后台任务已结束，终端 [{peer.name}] 取消了这次处理。"
            else:
                notice = f"🔔 后台任务已结束，终端 [{peer.name}] 仍在处理，稍后可在终端查看结果。"
            for st in chats:
                await self._reply(st, notice)
            return
        try:
            _pkg().send_peer_message(
                peer, text=report, from_name="gateway", from_peer_id="gateway", from_kind="agent",
            )
            notice = f"🔔 后台任务已结束，结果已交给终端 [{peer.name}] 继续处理。"
        except PeerMessageRefused as exc:
            notice = f"🔔 后台任务已结束，但没能交给终端 [{peer.name}]：{exc}"
        for st in chats:
            await self._reply(st, notice)

    async def _run_attached_cli_turn(
        self,
        state: ImChatState,
        peer: PeerInfo,
        work_dir: str,
        session_id: str,
        text: str,
    ) -> None:
        token = _read_attach_token(peer)
        if not token or not peer.attach_socket:
            # Fallback to local execution if socket disappeared
            q = self._queued_turns.setdefault(session_id, asyncio.Queue())
            await q.put(QueuedTurn(state, work_dir, session_id, text))
            await self._drain_session_queue(session_id, q)
            return

        if self._agent_service is not None:
            uid = self._user_id()
            self._agent_service.set_session_work_dir(session_id, work_dir, owner=uid)
            live_turn.bump_pulse(uid)

        started = time.monotonic()
        done_event = asyncio.Event()
        short = _short_dir(work_dir)

        async def _ack_if_slow() -> None:
            try:
                await asyncio.wait_for(done_event.wait(), timeout=ACK_DELAY_SECS)
            except asyncio.TimeoutError:
                await self._reply(
                    state,
                    f"⏳ 已发送到终端 [{peer.name}] 执行中，跑完自动汇报（发 /cancel 可中止）",
                )

        async def _watch_interactions() -> None:
            seen_req_ids = set()
            while not done_event.is_set():
                await asyncio.sleep(0.35)
                if done_event.is_set():
                    break
                try:
                    probe = _AttachRpcClient(peer.attach_socket, token, timeout=3.0)
                    snap = await asyncio.to_thread(probe.call, "session/load", {})
                except Exception:
                    continue
                pi = snap.get("pendingInteraction")
                if not isinstance(pi, dict):
                    continue
                req_id = str(pi.get("requestId") or "")
                if not req_id or req_id in seen_req_ids:
                    continue
                seen_req_ids.add(req_id)
                ikind = "approval" if pi.get("kind") == "approval" else "ask"
                raw_opts = [str(o) for o in (pi.get("options") or [])]
                if ikind == "approval":
                    labels = [_APPROVAL_LABELS.get(o, o) for o in raw_opts]
                    values = raw_opts
                    title = "终端请求工具执行权限"
                else:
                    labels = raw_opts
                    # A question's answer is free text: the CLI hands whatever it
                    # gets straight to the model. The number is only how the
                    # terminal/IM lists the options, so a pick must resolve to
                    # that option's text — sending "2" would answer the model
                    # with the string "2" instead of the choice it names.
                    values = raw_opts
                    title = str(pi.get("prompt") or "终端正在等待你的输入")

                def _resolve_cli(ans: str, rid: str = req_id) -> bool:
                    # The IM side answers in decision keys ("allow"/"deny"/…,
                    # what the card and the y/n letters map to). The terminal's
                    # ``session/answer`` takes either a letter or a 1-based index
                    # among the *visible* options it published — the same order
                    # this snapshot lists them in — so translate instead of
                    # handing it a name it answers ``accepted: false`` to.
                    if ikind == "approval" and ans in raw_opts:
                        ans = str(raw_opts.index(ans) + 1)
                    try:
                        c = _AttachRpcClient(peer.attach_socket, token, timeout=5.0)
                        r = c.call("session/answer", {"answer": ans, "requestId": rid})
                        return bool(r.get("accepted"))
                    except Exception:
                        return False

                await self._publish_interaction(
                    state,
                    interaction_kind=ikind,
                    title=title,
                    text=str(pi.get("prompt") or "") if ikind == "approval" else "",
                    options=labels,
                    values=values,
                    session_id=session_id,
                    request_id=req_id,
                    resolve_fn=_resolve_cli,
                    attach_peer=peer,
                )

        ack_task = asyncio.create_task(_ack_if_slow())
        watch_task = asyncio.create_task(_watch_interactions())
        try:
            client = _AttachRpcClient(peer.attach_socket, token, timeout=620.0)
            result = await asyncio.to_thread(
                client.call,
                "session/prompt",
                {"sessionId": session_id, "prompt": text},
            )
        except Exception as exc:
            await self._reply(state, f"❌ 终端 [{peer.name}] 执行异常：{exc}")
            return
        finally:
            done_event.set()
            ack_task.cancel()
            watch_task.cancel()
            if self._agent_service is not None:
                # The CLI records its own turns as "cli"; this one was typed in
                # the IM channel, so that stays the latest entry point.
                self._agent_service.session_log_for(session_id, owner=self._user_id()).set_source(
                    state.channel.value
                )
            live_turn.bump_pulse(self._user_id())

        elapsed = time.monotonic() - started
        stop_reason = result.get("stopReason")
        answer = (result.get("agenticaAnswer") or "").strip()
        if stop_reason == "cancelled":
            await self._reply(state, f"🛑 终端 [{peer.name}] 任务已取消。")
            return
        if stop_reason == "agentica_pending" and not answer:
            await self._reply(
                state,
                f"⏳ 终端 [{peer.name}] 仍在后台继续执行（可发 /status 查看状态）。",
            )
            return
        if answer:
            header = f"[{short} · {_fmt_elapsed(elapsed)}]\n" if elapsed >= 10.0 or state.verbose else ""
            await self._reply(state, f"{header}{answer}")
            await self._push_deliverable_files(state, answer, work_dir)

    async def _drain_session_queue(self, session_id: str, q: asyncio.Queue) -> None:
        uid = self._user_id()
        carried: Optional[QueuedTurn] = None
        while True:
            if carried is not None:
                batch, carried = [carried], None
            else:
                if q.empty():
                    return
                batch = [await q.get()]
            # Lines the user sent together run as one turn; a background report
            # is a turn of its own, never folded into what the user wrote.
            while not batch[0].automated and not q.empty():
                item = q.get_nowait()
                if item.automated:
                    carried = item
                    break
                batch.append(item)

            last = batch[-1]
            media: List[Any] = []
            for item in batch:
                media.extend(item.media)
            try:
                result = await self._run_gateway_turn(
                    last.state, last.work_dir, last.session_id,
                    "\n\n".join(item.text for item in batch if item.text), media,
                    automated=batch[0].automated, also_notify=batch[0].also_notify,
                    input_meta=batch[0].input_meta,
                )
                if batch[0].done is not None and not batch[0].done.done():
                    batch[0].done.set_result(result)
            except BaseException as exc:
                if batch[0].done is not None and not batch[0].done.done():
                    batch[0].done.set_exception(exc if isinstance(exc, Exception) else RuntimeError("cancelled"))
                raise
            finally:
                for item in batch:
                    if item.from_queue:
                        q.task_done()

            if self._agent_service is not None and carried is None:
                late = self._agent_service.take_undelivered_steer(last.session_id, owner=uid)
                if late:
                    carried = QueuedTurn(
                        last.state, last.work_dir, last.session_id, "\n\n".join(late), from_queue=False,
                    )
                    continue
            if carried is None and q.empty():
                return

    async def _run_gateway_turn(
        self,
        state: ImChatState,
        work_dir: str,
        session_id: str,
        text: str,
        media: List[Any],
        *,
        automated: bool = False,
        also_notify: Optional[List[ImChatState]] = None,
        input_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Run one turn of an IM chat; returns its ``ChatResult`` (None if it did not run)."""
        if self._agent_service is None:
            return None
        uid = self._user_id()
        self._agent_service.set_session_work_dir(session_id, work_dir, owner=uid)

        from ...routes import settings as settings_routes

        mode = settings_routes.approval_mode_pref(uid)
        if mode:
            self._agent_service.set_session_approval_mode(session_id, mode, owner=uid)

        turn = live_turn.start(session_id, owner=uid, user_message="" if automated else text)

        async def _notify_question(question: live_turn.PendingQuestion) -> None:
            # The turn owns the question; this card is one more place to answer it.
            await self._publish_interaction(
                state,
                interaction_kind="ask",
                title=question.prompt,
                text="",
                options=question.options,
                values=question.options,
                session_id=session_id,
                request_id=question.question_id,
                resolve_fn=lambda ans, qid=question.question_id: turn.answer_question(qid, ans, by="IM"),
            )

        turn.question_notifier = _notify_question

        def _close_question_card(question_id: str, by: str) -> None:
            # Answered on the Web / Desktop or timed out: the phone's card and
            # its numbered reply must not take a second answer.
            card = self._cards.find_open(request_id=question_id)
            if card is not None:
                self._cards.expire(card.token, reason=f"已在 {by} 处理" if by else "该提问已超时关闭")
            if state.pending is not None and state.pending.request_id == question_id:
                state.clear_pending()

        turn.question_closed = _close_question_card
        if input_meta and input_meta.get("cron_job"):
            turn.cron_job = input_meta["cron_job"]
            turn.automated_message = text
        touch_activity_stamp(uid)
        last_stamp_mono = [time.monotonic()]
        sub_q = turn.subscribe()
        done_event = asyncio.Event()
        started = time.monotonic()
        short = _short_dir(work_dir)

        def _maybe_touch_stamp() -> None:
            now_m = time.monotonic()
            if now_m - last_stamp_mono[0] >= 1.0:
                last_stamp_mono[0] = now_m
                touch_activity_stamp(uid)

        def _on_content(delta: str) -> None:
            turn.publish({"event": "content", "data": delta})

        def _on_tool_call(name: str, args: dict, tool_call_id: str = "") -> None:
            turn.publish({
                "event": "tool_call",
                "data": {"name": name, "args": args, "tool_call_id": tool_call_id},
            })
            _maybe_touch_stamp()

        def _on_tool_result(name: str, res_str: str, extra: Optional[dict] = None) -> None:
            data = {"name": name, "result": res_str}
            if extra:
                data.update(extra)
            turn.publish({"event": "tool_result", "data": data})
            _maybe_touch_stamp()

        def _on_thinking(delta: str) -> None:
            turn.publish({"event": "thinking", "data": delta})

        async def _ack_if_slow() -> None:
            try:
                await asyncio.wait_for(done_event.wait(), timeout=ACK_DELAY_SECS)
            except asyncio.TimeoutError:
                await self._reply(
                    state,
                    f"⏳ 已在 [{short}] 启动执行，跑完自动通知你（发 /status 查实时步骤，/ps 查后台任务，/cancel 中止）",
                )

        async def _watch_turn_events() -> None:
            while not done_event.is_set():
                try:
                    ev = await asyncio.wait_for(sub_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if ev is None:
                    break
                if ev.get("event") == "approval_request":
                    data = ev.get("data") or {}
                    tc_id = str(data.get("tool_call_id") or "")
                    if not tc_id:
                        continue
                    pending_obj = PendingApproval(
                        tool_call_id=tc_id,
                        name=str(data.get("name") or "tool"),
                        arguments=data.get("args") or {},
                        question=str(data.get("question") or ""),
                        preview=str(data.get("preview") or ""),
                        similar_label=str(data.get("similar_label") or ""),
                        options=tuple(data.get("options") or ("allow", "deny")),
                    )
                    vis_opts = list(visible_approval_decisions(pending_obj.options))
                    labels = [_APPROVAL_LABELS.get(o, o) for o in vis_opts]

                    def _resolve_approval(decision_str: str, tid: str = tc_id) -> bool:
                        d: ApprovalDecision = decision_str  # type: ignore[assignment]
                        return turn.approvals.decide(tid, d)

                    await self._publish_interaction(
                        state,
                        interaction_kind="approval",
                        title=pending_obj.question or f"允许执行 {pending_obj.name}？",
                        text=f"$ {pending_obj.preview}" if pending_obj.preview else "",
                        options=labels,
                        values=vis_opts,
                        session_id=session_id,
                        request_id=tc_id,
                        resolve_fn=_resolve_approval,
                    )

        # Nobody is waiting on a report they did not send: no "started" ack.
        ack_task = asyncio.create_task(done_event.wait() if automated else _ack_if_slow())
        watch_task = asyncio.create_task(_watch_turn_events())
        try:
            if hasattr(self._agent_service, "chat_stream"):
                result = await self._agent_service.chat_stream(
                    message=text,
                    session_id=session_id,
                    user_id=uid,
                    owner=uid,
                    on_content=_on_content,
                    on_tool_call=_on_tool_call,
                    on_tool_result=_on_tool_result,
                    on_thinking=_on_thinking,
                    media=media or None,
                    include_usage=False,
                    channel=state.channel.value,
                    automated=automated,
                    input_meta=input_meta,
                )
            else:
                result = await self._agent_service.chat(
                    message=text,
                    session_id=session_id,
                    user_id=uid,
                    owner=uid,
                    media=media or None,
                    channel=state.channel.value,
                )
        except Exception as exc:
            turn.finish("failed")
            await self._reply(state, f"❌ 执行出错：{exc}")
            return ChatResult(content=f"Error: {exc}", session_id=session_id, user_id=uid, error=str(exc))
        finally:
            done_event.set()
            turn.unsubscribe(sub_q)
            ack_task.cancel()
            watch_task.cancel()

        elapsed = time.monotonic() - started
        reply = (result.content or "").strip()
        if result.media_notes:
            notes = "\n".join(result.media_notes)
            reply = f"{notes}\n\n{reply}" if reply else notes

        # Send the IM text reply FIRST, before finishing the LiveTurn or pushing files,
        # so the phone receives the answer at the exact moment generation completes.
        if automated and not reply:
            # The phone is the reason this turn ran; never leave it silent.
            reply = "agent 处理完了，但没有给出文字结论，可以发消息追问。"
        if reply:
            job = (input_meta or {}).get("cron_job")
            if job:
                header = f"⏰ 定时任务「{job.get('name') or job.get('id')}」\n"
            else:
                header = "🔔 后台任务已结束\n" if automated else ""
            if state.verbose or elapsed >= 10.0 or (result.tool_calls and result.tool_calls >= 2):
                tools_part = f" · {result.tool_calls} tools" if result.tool_calls else ""
                header += f"[{short} · {_fmt_elapsed(elapsed)}{tools_part}]\n"
            for target in [state, *(also_notify or [])]:
                await self._reply(target, f"{header}{reply}")

        turn.publish({
            "event": "done",
            "data": {
                "session_id": session_id,
                "tool_calls": result.tool_calls,
                "tools_used": result.tools_used,
                "response_time": round(elapsed, 2),
                "turn_usage": result.turn_usage,
            },
        })
        turn.finish("completed")
        touch_activity_stamp(uid)

        if reply:
            await self._push_deliverable_files(state, result.content or "", work_dir)
        return result

    async def _push_deliverable_files(
        self,
        state: ImChatState,
        text: str,
        work_dir: str,
    ) -> None:
        if not text or self._channel_manager is None:
            return
        files = collect_auto_push_files(text, work_dir)
        for item in files:
            try:
                data = Path(item.path).read_bytes()
            except OSError:
                continue
            sent = await self._channel_manager.send_file(
                state.channel,
                state.channel_id,
                data=data,
                filename=item.name,
                mime_type=item.mime_type,
            )
            if sent:
                logger.info(
                    f"[im_bot] auto-pushed file {item.name} ({format_file_size(item.size)}) "
                    f"to {state.channel.value}:{state.channel_id}"
                )

    async def _reply(self, state: ImChatState, text: str) -> None:
        if not text or self._channel_manager is None:
            return
        for chunk in Channel.split_text(text, CHUNK_CHARS):
            await self._channel_manager.send(state.channel, state.channel_id, chunk)
