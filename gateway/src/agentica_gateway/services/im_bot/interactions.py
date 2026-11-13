# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM approval / ask_user_question cards, CLI hook bridge, and turn cancellation.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

from agentica.cli.approvals import (
    approval_decision_from_key,
    visible_approval_decisions,
)
from agentica.peers import PeerInfo
from agentica.utils.log import logger

from ...card_page import CardItem, gateway_public_url, phone_card_base_url
from .. import live_turn
from .attach import _AttachRpcClient, _read_attach_token
from .commands import ImChatState, PendingAction, _APPROVAL_LABELS


def _pkg():
    return sys.modules["agentica_gateway.services.im_bot"]


class ImInteractionsMixin:
    """Capability-URL cards (/c/<token>), numbered answers, hook events, and /cancel."""

    def _card_url(self, card: CardItem) -> str:
        base = gateway_public_url(_pkg().settings.host, _pkg().settings.port)
        return f"{base}/c/{card.token}"

    async def _publish_interaction(
        self,
        state: ImChatState,
        *,
        interaction_kind: str,
        title: str,
        text: str,
        options: List[str],
        values: List[str],
        session_id: str,
        request_id: str,
        resolve_fn: Callable[[str], bool],
        attach_peer: Optional[PeerInfo] = None,
    ) -> CardItem:
        def _on_card_resolved(card: CardItem) -> None:
            val = card.selected_value
            resolve_fn(val)
            if state.pending is not None and state.pending.card_token == card.token:
                state.clear_pending()

        card = self._cards.create(
            kind="approval" if interaction_kind == "approval" else "ask",
            title=title,
            text=text,
            options=options,
            values=values,
            recommend=0,
            allow_custom_text=(interaction_kind != "approval"),
            session_id=session_id,
            request_id=request_id,
            on_resolve=_on_card_resolved,
        )

        action = PendingAction(
            kind="interaction",
            options=list(values),
            interaction_kind=interaction_kind,
            request_id=request_id,
            session_id=session_id,
            card_token=card.token,
            attach_peer=attach_peer,
            resolve_fn=resolve_fn,
        )
        state.set_pending(action)

        header = "⚠️ [需要审批]" if interaction_kind == "approval" else "❓ [需要你的确认]"
        lines = [f"{header} {title}"]
        if text:
            lines.append(text)
        if options:
            lines.append("")
            for idx, opt_label in enumerate(options, 1):
                lines.append(f"  {idx}. {opt_label}")
        base = phone_card_base_url(_pkg().settings.host, _pkg().settings.port)
        if base is not None:
            lines.append(f"\n👉 一键处理：{base}/c/{card.token}")
        reply_hint = (
            "回复数字 1..N / y 允许 / n 拒绝" if interaction_kind == "approval"
            else "回复数字 1..N / 直接输入文字回答"
        )
        lines.append(f"（可直接点链接，或在此{reply_hint}）" if base is not None else f"（在此{reply_hint}）")
        try:
            await self._reply(state, "\n".join(lines))
        except BaseException:
            # The user never saw this card: their next ordinary message must
            # not be read as its answer, and its link must not take one.
            if state.pending is not None and state.pending.card_token == card.token:
                state.clear_pending()
            self._cards.expire(card.token, reason="发送失败")
            raise
        return card

    async def _handle_interaction_number(self, state: ImChatState, n: int) -> str:
        pending = state.pending
        if pending is None or pending.kind != "interaction":
            return self._render_menu(state)
        idx = n - 1
        if not (0 <= idx < len(pending.options)):
            return self._on_invalid_pending(state, max_n=len(pending.options))
        chosen_val = str(pending.options[idx])
        card_token = pending.card_token
        resolve_fn = pending.resolve_fn
        state.clear_pending()

        if card_token:
            self._cards.answer(card_token, choice=idx, by="IM", fire_callback=False)
        ok = resolve_fn(chosen_val) if resolve_fn is not None else False
        card = self._cards.get_by_token(card_token) if card_token else None
        label = card.selected_label if card is not None else chosen_val
        return f"✓ 已选择：{label}" if ok else "该请求已处理或已过期。"

    async def _handle_interaction_text(self, state: ImChatState, text: str) -> str:
        pending = state.pending
        if pending is None or pending.kind != "interaction":
            return self._render_menu(state)
        if pending.interaction_kind == "approval":
            decision = approval_decision_from_key(text, pending.options)
            if decision is None:
                return f"请回复数字 1..{len(pending.options)} 或 y / n，或点击上方卡片链接处理。"
            idx = pending.options.index(decision) if decision in pending.options else 0
            return await self._handle_interaction_number(state, idx + 1)

        card_token = pending.card_token
        resolve_fn = pending.resolve_fn
        state.clear_pending()
        if card_token:
            self._cards.answer(card_token, text=text, by="IM", fire_callback=False)
        ok = resolve_fn(text) if resolve_fn is not None else False
        return f"✓ 已发送回答：{text}" if ok else "该提问已处理或已过期。"

    async def handle_hook_event(
        self,
        payload: Dict[str, Any],
        *,
        wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Handle a CLI shell_hook event (needs.approval / needs.input / request.resolved).

        Forwards the CLI's blocking prompt to the active IM chat with a ``/c/<token>``
        capability link and numbered options, and optionally awaits the user's answer.
        """
        event = str(payload.get("hook_event_name") or "")
        request_id = str(payload.get("request_id") or uuid.uuid4().hex[:10])
        session_id = str(payload.get("session_id") or "")

        if event == "request.resolved":
            card = self._cards.find_open(request_id=request_id)
            if card is not None:
                by = str(payload.get("decided_by") or "terminal")
                self._cards.expire(card.token, reason=f"已在 {by} 处理")
            for st in self._states.values():
                if st.pending is not None and st.pending.request_id == request_id:
                    st.clear_pending()
            return {"ok": True, "request_id": request_id}

        if event not in ("needs.approval", "needs.input"):
            return {"ok": False, "error": f"unsupported hook event: {event}"}

        # Pick the IM chat bound to this session, or the most recently active IM chat
        target_state: Optional[ImChatState] = None
        for st in self._states.values():
            if session_id and st.session_id == session_id:
                target_state = st
                break
        if target_state is None and self._states:
            target_state = list(self._states.values())[-1]

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def _resolve_hook(val: str) -> bool:
            if not fut.done():
                fut.set_result(val)
                return True
            return False

        if event == "needs.approval":
            raw_opts = [str(o) for o in (payload.get("options") or ["allow", "deny"])]
            vis_opts = list(visible_approval_decisions(raw_opts))
            labels = [_APPROVAL_LABELS.get(o, o) for o in vis_opts]
            tool_name = str(payload.get("tool_name") or "tool")
            question = str(payload.get("question") or f"允许执行 {tool_name}？")
            preview = str(payload.get("preview") or "")
            text = f"$ {preview}" if preview else ""
            ikind = "approval"
            values = vis_opts
        else:
            raw_opts = [str(o) for o in (payload.get("options") or [])]
            labels = raw_opts
            values = raw_opts
            question = str(payload.get("question") or payload.get("prompt") or "请输入你的回答")
            text = ""
            ikind = "ask"

        if target_state is not None:
            card = await self._publish_interaction(
                target_state,
                interaction_kind=ikind,
                title=question,
                text=text,
                options=labels,
                values=values,
                session_id=session_id,
                request_id=request_id,
                resolve_fn=_resolve_hook,
            )
        else:
            def _on_card(c: CardItem) -> None:
                _resolve_hook(c.selected_value)

            card = self._cards.create(
                kind="approval" if ikind == "approval" else "ask",
                title=question,
                text=text,
                options=labels,
                values=values,
                allow_custom_text=(ikind != "approval"),
                session_id=session_id,
                request_id=request_id,
                on_resolve=_on_card,
            )

        out: Dict[str, Any] = {
            "card_id": card.id,
            "token": card.token,
            "link": self._card_url(card),
            "request_id": request_id,
        }
        if wait_seconds > 0:
            try:
                chosen_val = await asyncio.wait_for(fut, timeout=wait_seconds)
                if event == "needs.approval":
                    out["decision"] = chosen_val
                else:
                    out["answer"] = chosen_val
            except asyncio.TimeoutError:
                out["pending"] = True
        return out

    async def _cancel_active_turn(self, state: ImChatState) -> str:
        sid = state.session_id
        peer = self.find_live_cli_peer(sid)
        cancelled_any = False
        if peer is not None and peer.attach_socket:
            token = _read_attach_token(peer)
            if token:
                try:
                    client = _AttachRpcClient(peer.attach_socket, token, timeout=5.0)
                    await asyncio.to_thread(client.call, "session/cancel", {})
                    cancelled_any = True
                except Exception as exc:
                    logger.debug(f"im_bot: attach cancel failed: {exc}")

        uid = self._user_id()
        if self._agent_service is not None:
            res = await live_turn.cancel_and_wait(
                self._agent_service, session_id=sid, owner=uid, timeout=5.0,
            )
            if res.get("cancelled") or self._agent_service.cancel_session(sid, owner=uid):
                cancelled_any = True

        q = self._queued_turns.get(sid)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                    cancelled_any = True
                except asyncio.QueueEmpty:
                    break

        if state.has_active_pending() and state.pending.kind == "interaction":
            if state.pending.card_token:
                self._cards.expire(state.pending.card_token, "已取消")
            state.clear_pending()

        if cancelled_any:
            running_bg = self._running_bg_count(sid, state.work_dir)
            extra = f"\n（另有 {running_bg} 个后台子进程仍在运行，发 /ps 可查看或停止）" if running_bg else ""
            return f"🛑 已发送中止指令（会话 {sid[:8]}）。{extra}"
        running_bg = self._running_bg_count(sid, state.work_dir)
        if running_bg:
            return (
                f"当前会话（{sid[:8]}）没有正在生成的对话轮次，但有 {running_bg} 个后台任务正在运行。\n\n"
                + self._render_bg_jobs(state)
            )
        return f"当前会话（{sid[:8]}）没有正在运行的任务。"
