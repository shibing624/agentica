# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM menus, /project & /session pickers, /status, /model, and /ps views.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentica.config.profiles import get_active_profile_name, get_profiles
from agentica.memory.session_log import SessionLog, local_turn_stamp
from agentica.peers import PeerInfo, list_live_peers
from agentica.tools.background_processes import (
    count_running_session_jobs,
    load_session_job_views,
)

import sys
from ...card_page import phone_card_base_url
from ...channels.base import Message
from .. import chat_images, live_turn
from ..im_files import format_file_size
from .commands import (
    HELP_TEXT,
    ImChatState,
    ImCommand,
    MENU_COMMANDS,
    PENDING_INVALID_LIMIT,
    PendingAction,
    SESSION_PAGE_SIZE,
    WORKSPACE_PAGE_SIZE,
    _SOURCE_LABELS,
    _fmt_elapsed,
    _short_dir,
    _workspace_labels,
    parse_im_command,
)


def _pkg():
    return sys.modules["agentica_gateway.services.im_bot"]


class ImMenusMixin:
    """Command dispatch, numbered menus, /project, /session, /model, /status, /ps."""

    async def _dispatch_command(self, state: ImChatState, cmd: ImCommand, message: Message) -> str:
        if cmd.kind == "menu":
            return self._render_menu(state)
        if cmd.kind == "help":
            return HELP_TEXT + "\n\n" + self._render_model_line(state)
        if cmd.kind == "status":
            return self._render_status(state)
        if cmd.kind == "ps":
            return await self._handle_ps_command(state, cmd.arg)
        if cmd.kind == "stop_bg":
            return await self._stop_bg_command(state, cmd.arg)
        if cmd.kind == "verbose":
            state.verbose = cmd.arg == "on"
            self._save_states()
            mode_str = "详细模式（附带工具统计与思考摘要）" if state.verbose else "简洁模式（仅推送最终结论与产物文件）"
            return f"已切换为：{mode_str}"
        if cmd.kind == "project":
            if cmd.arg and Path(os.path.expanduser(cmd.arg)).is_dir():
                target_dir = os.path.realpath(os.path.expanduser(cmd.arg))
                return self._enter_workspace(state, target_dir)
            return self._open_workspace_picker(state, page=0)
        if cmd.kind == "new":
            new_sid = str(uuid.uuid4())
            self._bind_session(state, state.work_dir, new_sid)
            if cmd.arg:
                await self._submit_turn(state, message, text_override=cmd.arg)
                return ""
            return self._render_bound_banner(state, prefix="🆕 已新建并绑定会话")
        if cmd.kind == "session":
            return self._open_session_picker(state, page=0)
        if cmd.kind == "model":
            if cmd.arg:
                return await self._apply_model_choice(state, cmd.arg)
            return self._open_model_picker(state)
        if cmd.kind == "cancel":
            return await self._cancel_active_turn(state)
        return HELP_TEXT

    def _session_title(self, work_dir: str, session_id: str) -> str:
        uid = self._user_id()
        log = SessionLog(session_id=session_id, work_dir=work_dir, user_id=uid)
        name = log.get_name()
        if name:
            return name
        if log.path.is_file() and log.path.stat().st_size > 0:
            preview = SessionLog.session_preview(str(log.path), max_chars=48)
            first = chat_images.strip_markers(preview.get("first_user") or "").strip()
            if first:
                return " ".join(first.split())[:40]
            cron_name = str(preview.get("cron_job_name") or "").strip()
            if cron_name:
                return cron_name[:48]
        return f"会话 {session_id[:8]}"

    def _is_busy(self, session_id: str) -> bool:
        peer = self.find_live_cli_peer(session_id)
        if peer is not None and peer.busy:
            return True
        uid = self._user_id()
        if self._agent_service is not None and self._agent_service.is_session_active(session_id, owner=uid):
            return True
        return session_id in self._turn_tasks and not self._turn_tasks[session_id].done()

    def _session_bg_jobs(self, session_id: str, work_dir: str = "") -> List[Dict[str, Any]]:
        uid = self._user_id()
        if self._agent_service is not None:
            return self._agent_service.list_background_jobs(session_id, owner=uid)
        if not work_dir:
            return []
        log = SessionLog(session_id=session_id, work_dir=work_dir, user_id=uid)
        return load_session_job_views(log.base_dir / session_id / "background")

    def _running_bg_count(self, session_id: str, work_dir: str = "") -> int:
        uid = self._user_id()
        if work_dir:
            log = SessionLog(session_id=session_id, work_dir=work_dir, user_id=uid)
            bg_dir = log.base_dir / session_id / "background"
            if self._agent_service is None:
                return count_running_session_jobs(bg_dir)
        jobs = self._session_bg_jobs(session_id, work_dir)
        return sum(1 for j in jobs if j.get("status") == "running")

    def _live_turn_step_line(self, session_id: str) -> str:
        turn = live_turn.active(session_id, owner=self._user_id())
        if turn is None:
            return ""
        prog = turn.progress_summary()
        elapsed = _fmt_elapsed(float(prog.get("elapsed_s") or 0.0))
        tools = int(prog.get("tool_calls") or 0)
        pending_app = str(prog.get("pending_approval") or "")
        active_tool = str(prog.get("active_tool") or "")
        detail = str(prog.get("active_detail") or "")
        head = f"已耗时 {elapsed}" + (f" · 第 {tools} 次工具调用" if tools else "")
        if pending_app:
            return f"{head} · ⚠️ 等待审批: {pending_app}"
        if active_tool:
            return f"{head} · 正在执行: {active_tool}" + (f" ({detail[:60]})" if detail else "")
        if detail:
            return f"{head} · 正在处理: {detail[:60]}"
        return head

    def _render_bound_banner(self, state: ImChatState, *, prefix: str = "📌 当前控制目标") -> str:
        peer = self.find_live_cli_peer(state.session_id)
        title = self._session_title(state.work_dir, state.session_id)
        busy = self._is_busy(state.session_id)
        bg_jobs = self._session_bg_jobs(state.session_id, state.work_dir)
        running_bg = [j for j in bg_jobs if j.get("status") == "running"]
        runtime_tag = (
            f"🟢 终端在线 ({peer.name})" if peer is not None else "💻 后台共享会话"
        )
        if busy:
            status_tag = "⏳ 运行中"
        elif running_bg:
            status_tag = f"🔄 {len(running_bg)} 个后台任务运行中"
        else:
            status_tag = "空闲"
        if busy and running_bg:
            status_tag += f" · 🔄 {len(running_bg)} 个后台任务"
        lines = [
            f"{prefix}：",
            f"  项目：{_short_dir(state.work_dir)} ({state.work_dir})",
            f"  会话：{title} [{state.session_id[:8]}]",
            f"  模式：{runtime_tag} · {status_tag}",
        ]
        if peer is not None and peer.task:
            lines.append(f"  当前任务：{peer.task}")
        step_line = self._live_turn_step_line(state.session_id)
        if step_line:
            lines.append(f"  实时进度：{step_line}")
        return "\n".join(lines)

    def _render_menu(self, state: ImChatState) -> str:
        banner = self._render_bound_banner(state)
        lines = [banner, "", "回复数字选择操作："]
        cmds: List[str] = []
        for idx, (cmd_str, label) in enumerate(MENU_COMMANDS, 1):
            lines.append(f"  {idx}. {label} ({cmd_str})")
            cmds.append(cmd_str)
        lines.append("\n直接输入文字即可向当前会话派发任务。")
        state.set_pending(PendingAction(kind="menu", options=cmds))
        return "\n".join(lines)

    def _render_status(self, state: ImChatState) -> str:
        uid = self._user_id()
        banner = self._render_bound_banner(state, prefix="📊 当前会话状态")
        lines = [banner]
        model_line = self._render_model_line(state)
        if model_line:
            lines.append(f"  {model_line}")
        log = SessionLog(session_id=state.session_id, work_dir=state.work_dir, user_id=uid)
        if log.path.is_file():
            lines.append(f"  日志：{format_file_size(log.path.stat().st_size)}")
        from ...routes import settings as settings_routes

        mode = settings_routes.approval_mode_pref(uid) or "auto"
        lines.append(f"  审批：{mode}" + ("（从不询问，不会出现审批卡片）" if mode == "allow-all" else ""))
        if phone_card_base_url(_pkg().settings.host, _pkg().settings.port) is None:
            # The card page posts the decision back to this gateway, so it needs
            # an address the phone can reach; a static share link (publish_*)
            # cannot take the click.
            lines.append(
                "  审批卡片链接：不发（卡片要把点击提交回本机网关，手机连不到 127.0.0.1）；"
                "审批/提问直接回数字即可。要发链接需让手机能访问网关，并设置 AGENTICA_PUBLIC_URL"
            )
        lines.append(f"  汇报：{'详细 (/verbose)' if state.verbose else '简洁 (/concise)'}")
        bg_jobs = self._session_bg_jobs(state.session_id, state.work_dir)
        running_bg = [j for j in bg_jobs if j.get("status") == "running"]
        settled_bg = [j for j in bg_jobs if j.get("status") != "running"]
        if running_bg:
            lines.append(f"  后台任务（{len(running_bg)} 个运行中，发 /ps 查看日志或停止）：")
            now_ms = int(time.time() * 1000)
            for j in running_bg[:5]:
                dur = _fmt_elapsed(max(0.0, (now_ms - int(j.get("started_at_ms") or now_ms)) / 1000.0))
                lbl = " ".join(str(j.get("label") or "").split())[:56]
                prog = str(j.get("progress") or "").strip()
                prog_part = f" → {prog[:56]}" if prog else ""
                lines.append(f"    • #{j.get('num')} [{j.get('id')}] {lbl} ({dur}){prog_part}")
        elif settled_bg:
            lines.append(f"  后台任务：无运行中（历史已结束 {len(settled_bg)} 个，发 /ps 查看输出）")
        return "\n".join(lines)

    def _render_model_line(self, state: ImChatState) -> str:
        if self._agent_service is None:
            return ""
        cfg = self._agent_service.session_model_config(state.session_id, owner=self._user_id())
        prof = cfg.get("profile") or "default"
        line = f"模型：{cfg.get('model_provider')}/{cfg.get('model_name')} (profile: {prof})"
        if cfg.get("auxiliary_model_name"):
            line += f"，子代理/辅助：{cfg.get('auxiliary_model_name')}"
        return line

    def _collect_workspace_items(self, current_dir: str) -> List[Dict[str, Any]]:
        uid = self._user_id()
        cur_real = os.path.realpath(os.path.expanduser(current_dir)) if current_dir else ""
        base_raw = str(_pkg().settings.base_dir)
        base_real = os.path.realpath(os.path.expanduser(base_raw)) if base_raw else ""

        live_by_dir: Dict[str, PeerInfo] = {}
        live_ids: set = set()
        for peer in list_live_peers():
            if peer.session_id:
                live_ids.add(peer.session_id)
            if peer.cwd:
                p_real = os.path.realpath(os.path.expanduser(peer.cwd))
                if Path(p_real).is_dir() and p_real not in live_by_dir:
                    live_by_dir[p_real] = peer

        by_dir: Dict[str, Dict[str, Any]] = {}

        def _ensure(real: str) -> Optional[Dict[str, Any]]:
            if not real or not Path(real).is_dir():
                return None
            item = by_dir.get(real)
            if item is None:
                item = {
                    "work_dir": real,
                    "mtime": 0.0,
                    "session_count": 0,
                    "latest_sid": "",
                    "latest_name": "",
                }
                by_dir[real] = item
            return item

        _ensure(cur_real)
        for p_real in live_by_dir:
            _ensure(p_real)
        _ensure(base_real)

        for proj in SessionLog.list_projects(user_id=uid):
            wd = proj.get("work_dir") or ""
            if not wd:
                continue
            real = os.path.realpath(os.path.expanduser(wd))
            if not Path(real).is_dir():
                continue
            counted = []
            for entry in SessionLog.list_sessions(base_dir=proj.get("base_dir"), work_dir=real, user_id=uid):
                sid = str(entry.get("session_id") or "")
                if not sid or sid.startswith("scheduled_"):
                    continue
                if int(entry.get("size_bytes") or 0) <= 0 and sid not in live_ids:
                    continue
                counted.append(entry)
            pinned = real == cur_real or real == base_real or real in live_by_dir
            if not pinned and _pkg().is_system_temp_path(real):
                continue
            item = _ensure(real)
            if item is None:
                continue
            for entry in counted:
                sid = str(entry.get("session_id") or "")
                item["session_count"] += 1
                mtime = float(entry.get("mtime") or 0.0)
                if mtime >= item["mtime"]:
                    item["mtime"] = mtime
                    item["latest_sid"] = sid
                    item["latest_name"] = str(entry.get("name") or "")

        items = list(by_dir.values())
        items.sort(
            key=lambda it: (
                it["work_dir"] == cur_real,
                it["work_dir"] in live_by_dir,
                it["mtime"],
                it["work_dir"] == base_real,
            ),
            reverse=True,
        )
        labels = _workspace_labels([it["work_dir"] for it in items])
        for it in items:
            wd = it["work_dir"]
            it["label"] = labels[wd]
            it["live_peer"] = live_by_dir.get(wd)
        return items

    def _list_workspace_choices(self, current_dir: str) -> List[str]:
        return [it["work_dir"] for it in self._collect_workspace_items(current_dir)]

    def _open_workspace_picker(self, state: ImChatState, page: int = 0) -> str:
        all_items = self._collect_workspace_items(state.work_dir)
        if not all_items:
            return "未找到可切换的项目工作区。"
        start = page * WORKSPACE_PAGE_SIZE
        if start >= len(all_items):
            page = 0
            start = 0
        page_items = all_items[start : start + WORKSPACE_PAGE_SIZE]
        has_more = (start + WORKSPACE_PAGE_SIZE) < len(all_items)
        cur_real = os.path.realpath(state.work_dir) if state.work_dir else ""
        page_note = f"，第 {page + 1} 页" if (page > 0 or has_more) else ""
        lines = [f"📂 选择要控制的项目工作区（回复数字 1..{len(page_items)}{page_note}）："]
        for idx, item in enumerate(page_items, 1):
            path = item["work_dir"]
            cur = " [当前]" if path == cur_real else ""
            peer: Optional[PeerInfo] = item.get("live_peer")
            live_tag = f" 🟢终端:{peer.name}" if peer else ""
            count = int(item.get("session_count") or 0)
            latest_sid = str(item.get("latest_sid") or "")
            if peer and peer.session_id:
                recent = peer.task or self._session_title(path, peer.session_id)
                meta = f" · 最近：{recent[:28]}" + (f"（共 {count} 个会话）" if count > 1 else "")
            elif latest_sid:
                recent = item.get("latest_name") or self._session_title(path, latest_sid)
                meta = f" · 最近：{recent[:28]}" + (f"（共 {count} 个会话）" if count > 1 else "")
            else:
                meta = ""
            # One line per choice. WeChat renders Markdown: an indented second
            # line under "N." is folded into the neighbouring list item, which
            # printed each path glued to the next project's name.
            lines.append(f"  {idx}. {item['label']}{cur}{live_tag}{meta} · {path}")
        if has_more:
            lines.append("  0. 下一页")
        state.set_pending(
            PendingAction(
                kind="project",
                options=[it["work_dir"] for it in page_items],
                page=page,
                has_more=has_more,
            )
        )
        return "\n".join(lines)

    def _enter_workspace(self, state: ImChatState, work_dir: str) -> str:
        real_dir = os.path.realpath(os.path.expanduser(work_dir))
        sessions = self._collect_session_items(real_dir)
        if not sessions:
            sid = str(uuid.uuid4())
            self._bind_session(state, real_dir, sid)
            return self._render_bound_banner(state, prefix="✅ 已切换项目")
        if len(sessions) == 1:
            self._bind_session(state, real_dir, sessions[0]["session_id"])
            return self._render_bound_banner(state, prefix="✅ 已切换项目")
        self._bind_session(state, real_dir, sessions[0]["session_id"])
        return self._open_session_picker(
            state,
            page=0,
            include_new=True,
            heading=f"✅ 已切换到项目 [{_short_dir(real_dir)}]，共有 {len(sessions)} 个会话，请选择要挂载的一个",
        )

    def _collect_session_items(self, work_dir: str) -> List[Dict[str, Any]]:
        uid = self._user_id()
        real_dir = os.path.realpath(os.path.expanduser(work_dir)) if work_dir else ""
        live_map: Dict[str, PeerInfo] = {}
        items: List[Dict[str, Any]] = []
        seen_ids = set()

        for peer in list_live_peers():
            if peer.session_id:
                live_map[peer.session_id] = peer
                if peer.cwd and os.path.realpath(os.path.expanduser(peer.cwd)) == real_dir:
                    seen_ids.add(peer.session_id)
                    items.append({
                        "session_id": peer.session_id,
                        "work_dir": real_dir,
                        "title": peer.task or self._session_title(real_dir, peer.session_id),
                        "stamp": "终端在线",
                        "source": "cli",
                        "live_peer": peer,
                    })

        for entry in SessionLog.list_sessions(work_dir=real_dir, user_id=uid):
            sid = str(entry.get("session_id") or "")
            if not sid or sid.startswith("scheduled_") or sid in seen_ids:
                continue
            if int(entry.get("size_bytes") or 0) <= 0 and sid not in live_map:
                continue
            seen_ids.add(sid)
            peer = live_map.get(sid)
            title = entry.get("name") or self._session_title(real_dir, sid)
            stamp = local_turn_stamp(entry.get("last_timestamp") or entry.get("mtime"))
            src = str(entry.get("source") or ("cli" if entry.get("cli") else ""))
            items.append({
                "session_id": sid,
                "work_dir": real_dir,
                "title": title,
                "stamp": stamp,
                "source": src,
                "live_peer": peer,
            })
        return items

    def _open_session_picker(
        self,
        state: ImChatState,
        page: int = 0,
        *,
        include_new: bool = False,
        heading: str = "",
    ) -> str:
        all_items = self._collect_session_items(state.work_dir)
        if not all_items:
            return f"当前项目 [{_short_dir(state.work_dir)}] 暂无历史会话。发送 /new 可新建会话。"
        if include_new:
            all_items = all_items + [{"new_session": True, "work_dir": state.work_dir}]
        start = page * SESSION_PAGE_SIZE
        if start >= len(all_items):
            page = 0
            start = 0
        page_items = all_items[start : start + SESSION_PAGE_SIZE]
        has_more = (start + SESSION_PAGE_SIZE) < len(all_items)

        head = heading or f"🗂 选择要挂载的会话（项目：{_short_dir(state.work_dir)}，第 {page + 1} 页）"
        lines = [f"{head}："]
        for idx, item in enumerate(page_items, 1):
            if item.get("new_session"):
                lines.append(f"  {idx}. 🆕 在此项目新建会话")
                continue
            sid = item["session_id"]
            cur = " [当前]" if sid == state.session_id else ""
            src_label = _SOURCE_LABELS.get(str(item.get("source") or ""))
            src_tag = f" [{src_label}]" if src_label else ""
            peer: Optional[PeerInfo] = item.get("live_peer")
            bg_n = self._running_bg_count(sid, str(item.get("work_dir") or state.work_dir))
            if peer is not None:
                badge = "⏳终端运行中" if peer.busy else f"🟢终端:{peer.name}"
                if bg_n:
                    badge += f"·🔄{bg_n}个后台"
            elif self._is_busy(sid):
                badge = "⏳后台运行中" + (f"·🔄{bg_n}个后台" if bg_n else "")
            elif bg_n:
                badge = f"🔄{bg_n}个后台任务"
            else:
                badge = item.get("stamp") or ""
            badge_str = f" ({badge})" if badge else ""
            lines.append(f"  {idx}. {item['title']}{src_tag}{cur}{badge_str} [{sid[:6]}]")
        if has_more:
            lines.append("  0. 下一页")
        state.set_pending(
            PendingAction(
                kind="session_with_new" if include_new else "session",
                options=page_items,
                page=page,
                has_more=has_more,
            )
        )
        return "\n".join(lines)

    def _open_model_picker(self, state: ImChatState) -> str:
        profiles = get_profiles()
        if not profiles:
            return "未在 config.yaml 中找到配置的模型 Profile。"
        uid = self._user_id()
        current_prof = (
            self._agent_service.session_profile_name(state.session_id, owner=uid)
            if self._agent_service
            else ""
        ) or get_active_profile_name()
        options: List[Tuple[str, str]] = []
        lines = ["🤖 选择当前会话使用的模型（回复数字 1..N）："]
        for idx, (name, prof) in enumerate(profiles.items(), 1):
            prov = prof.get("model_provider", "")
            mname = prof.get("model_name", "")
            label = f"{name} ({prov}/{mname})"
            cur = " [当前]" if name == current_prof else ""
            lines.append(f"  {idx}. {label}{cur}")
            options.append((name, label))
        state.set_pending(PendingAction(kind="model", options=options))
        return "\n".join(lines)

    async def _apply_model_choice(self, state: ImChatState, name: str) -> str:
        profiles = get_profiles()
        if name not in profiles:
            return f"未知模型 Profile：{name}"
        if self._agent_service is not None:
            cfg = await self._agent_service.set_session_profile(
                state.session_id, name, owner=self._user_id(),
            )
            return f"✅ 已切换模型为：{name} ({cfg.get('model_provider')}/{cfg.get('model_name')})"
        return f"✅ 已切换模型为：{name}"

    async def _handle_number(self, state: ImChatState, n: int) -> str:
        pending = state.pending
        if pending is None:
            return self._render_menu(state)

        if pending.kind == "interaction":
            return await self._handle_interaction_number(state, n)

        if n == 0:
            if pending.kind == "project" and pending.has_more:
                return self._open_workspace_picker(state, page=pending.page + 1)
            if pending.kind in ("session", "session_with_new") and pending.has_more:
                return self._open_session_picker(
                    state,
                    page=pending.page + 1,
                    include_new=(pending.kind == "session_with_new"),
                )
            if pending.kind == "bg_detail":
                state.clear_pending()
                return self._render_bg_jobs(state)
            state.clear_pending()
            return self._render_menu(state)

        idx = n - 1
        if not (0 <= idx < len(pending.options)):
            return self._on_invalid_pending(state, max_n=len(pending.options))

        kind = pending.kind
        chosen = pending.options[idx]
        state.clear_pending()

        if kind == "bg_list":
            return self._render_bg_job_detail(state, str(chosen))

        if kind == "bg_detail":
            action, job_id = chosen
            if action == "refresh":
                return self._render_bg_job_detail(state, str(job_id))
            if action == "stop":
                return await self._stop_bg_command(state, str(job_id))
            return self._render_bg_jobs(state)

        if kind == "menu":
            fake_msg = Message(
                channel=state.channel,
                channel_id=state.channel_id,
                sender_id="user",
                sender_name="user",
                content=str(chosen),
                message_id="menu-select",
            )
            return await self._dispatch_command(state, parse_im_command(str(chosen)), fake_msg)

        if kind == "project":
            return self._enter_workspace(state, str(chosen))

        if kind in ("session", "session_with_new"):
            item = chosen
            if item.get("new_session"):
                new_sid = str(uuid.uuid4())
                self._bind_session(state, item["work_dir"], new_sid)
                return self._render_bound_banner(state, prefix="🆕 已新建并绑定会话")
            self._bind_session(state, item["work_dir"], item["session_id"])
            return self._render_bound_banner(state, prefix="✅ 已挂载会话")

        if kind == "model":
            prof_name, _ = chosen
            return await self._apply_model_choice(state, prof_name)

        return self._render_menu(state)

    def _on_invalid_pending(self, state: ImChatState, max_n: int) -> str:
        state.pending_invalid_count += 1
        if state.pending_invalid_count >= PENDING_INVALID_LIMIT:
            state.clear_pending()
            return "多次输入无效，已退出当前菜单。发送 /menu 重新打开。"
        return f"请输入有效编号（1..{max_n}），或发 0 返回菜单。"

    # ── Background jobs (/ps & /stop) ──────────────────────────────────

    async def _handle_ps_command(self, state: ImChatState, arg: str) -> str:
        raw = (arg or "").strip()
        if raw:
            head, _, tail = raw.partition(" ")
            if head.lower() in ("stop", "kill", "停止") and tail.strip():
                return await self._stop_bg_command(state, tail.strip())
            return self._render_bg_job_detail(state, raw)
        return self._render_bg_jobs(state)

    def _ordered_bg_jobs(self, state: ImChatState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        jobs = self._session_bg_jobs(state.session_id, state.work_dir)
        live = sorted(
            [j for j in jobs if j.get("status") == "running"],
            key=lambda j: int(j.get("started_at_ms") or 0),
        )
        settled = sorted(
            [j for j in jobs if j.get("status") != "running"],
            key=lambda j: int(j.get("finished_at_ms") or j.get("started_at_ms") or 0),
            reverse=True,
        )
        return live, settled

    def _render_bg_jobs(self, state: ImChatState) -> str:
        title = self._session_title(state.work_dir, state.session_id)
        step_line = self._live_turn_step_line(state.session_id)
        live, settled = self._ordered_bg_jobs(state)
        shown_settled = settled[:6]
        all_shown = live + shown_settled

        lines = [f"🖥 后台任务 · {title} [{state.session_id[:8]}]"]
        if step_line:
            lines.append(f"⏳ 当前轮次执行中：{step_line}")
        if not all_shown:
            if step_line:
                lines.append("\n当前暂无独立的 background=True 后台子进程（发 /cancel 可中止当前对话轮次）。")
            else:
                lines.append("\n当前会话暂无后台任务。")
            return "\n".join(lines)

        now_ms = int(time.time() * 1000)
        options: List[str] = []

        def _fmt_item(idx: int, job: Dict[str, Any]) -> str:
            st = str(job.get("status") or "")
            icon = "🔄" if st == "running" else ("✅" if st == "completed" else ("🛑" if st == "stopped" else "❌"))
            end_ms = now_ms if st == "running" else int(job.get("finished_at_ms") or job.get("started_at_ms") or now_ms)
            dur = _fmt_elapsed(max(0.0, (end_ms - int(job.get("started_at_ms") or end_ms)) / 1000.0))
            lbl = " ".join(str(job.get("label") or "").split())[:52]
            if st == "running":
                prog = str(job.get("progress") or "").strip()
                tail = f" · 最新: {prog[:48]}" if prog else ""
                return f"  {idx}. {icon} [#{job.get('num')} {job.get('id')}] {lbl} ({dur}){tail}"
            code = job.get("exit_code")
            st_txt = f"失败(exit {code})" if st == "failed" and code is not None else (
                "已完成" if st == "completed" else ("已停止" if st == "stopped" else "失败")
            )
            return f"  {idx}. {icon} [#{job.get('num')} {job.get('id')}] {lbl} · {st_txt} ({dur})"

        if live:
            lines.append(f"\n🔄 运行中（{len(live)}）：")
            for job in live:
                options.append(str(job["id"]))
                lines.append(_fmt_item(len(options), job))
        if shown_settled:
            lines.append(f"\n📋 最近已结束（{len(shown_settled)}）：")
            for job in shown_settled:
                options.append(str(job["id"]))
                lines.append(_fmt_item(len(options), job))

        lines.append("\n回复数字 1..N 查看实时输出日志或停止任务（0 返回主菜单，/stop all 停止全部运行中后台任务）。")
        state.set_pending(PendingAction(kind="bg_list", options=options))
        return "\n".join(lines)

    def _render_bg_job_detail(self, state: ImChatState, target: str) -> str:
        if self._agent_service is None:
            return "服务尚未就绪。"
        uid = self._user_id()
        detail = self._agent_service.background_job_output(
            state.session_id, target, owner=uid, max_lines=25, max_chars=1400,
        )
        if detail is None:
            return f"未找到后台任务 `{target}`。发送 /ps 查看当前会话任务列表。"

        st = str(detail.get("status") or "")
        now_ms = int(time.time() * 1000)
        end_ms = now_ms if st == "running" else int(detail.get("finished_at_ms") or detail.get("started_at_ms") or now_ms)
        dur = _fmt_elapsed(max(0.0, (end_ms - int(detail.get("started_at_ms") or end_ms)) / 1000.0))
        st_badge = {
            "running": f"🔄 运行中（已运行 {dur}）",
            "completed": f"✅ 已完成（耗时 {dur}）",
            "stopped": f"🛑 已停止（耗时 {dur}）",
            "failed": f"❌ 失败 exit {detail.get('exit_code')}（耗时 {dur}）",
        }.get(st, f"{st}（{dur}）")

        label = str(detail.get("label") or "").strip()
        cmd = str(detail.get("command") or "").strip()
        lines = [
            f"🖥 后台任务 #{detail.get('num')} [{detail.get('id')}] · {st_badge}",
        ]
        if detail.get("pid"):
            lines[0] += f" · PID {detail.get('pid')}"
        if label:
            lines.append(f"描述：{label[:120]}")
        if cmd and cmd != label:
            flat_cmd = " ".join(cmd.split())
            lines.append(f"命令：$ {flat_cmd[:180]}{'…' if len(flat_cmd) > 180 else ''}")

        out = str(detail.get("output") or "").strip()
        lines.append("\n── 最新输出 ──")
        lines.append(out if out else "（暂无输出）")

        if st == "running":
            jid = str(detail["id"])
            lines.append("\n回复数字操作：\n  1. 🔄 刷新最新输出\n  2. 🛑 停止该后台任务\n  0. 返回后台任务列表")
            state.set_pending(
                PendingAction(kind="bg_detail", options=[("refresh", jid), ("stop", jid)])
            )
        else:
            lines.append("\n回复数字操作：\n  1. 返回后台任务列表\n  0. 打开主菜单")
            state.set_pending(PendingAction(kind="bg_detail", options=[("back", "")]))
        return "\n".join(lines)

    async def _stop_bg_command(self, state: ImChatState, target: str) -> str:
        if self._agent_service is None:
            return "服务尚未就绪。"
        uid = self._user_id()
        clean = target.strip()
        if clean.lower() in ("all", "*", "全部"):
            stopped = await self._agent_service.stop_all_session_background_jobs(state.session_id, owner=uid)
            state.clear_pending()
            if not stopped:
                return "当前会话没有正在运行的后台任务。"
            ids = ", ".join(f"#{j.get('num')}({j.get('id')})" for j in stopped)
            return f"🛑 已停止全部 {len(stopped)} 个后台任务：{ids}"

        res = await self._agent_service.stop_background_job(state.session_id, clean, owner=uid)
        state.clear_pending()
        if res is None:
            return f"未找到后台任务 `{clean}`。发送 /ps 查看任务列表。"
        lbl = " ".join(str(res.get("label") or "").split())[:60]
        return f"🛑 已停止后台任务 #{res.get('num')} [{res.get('id')}]：{lbl}"
