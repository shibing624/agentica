# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for ImBotController, capability-URL card pages (/c/<token>),
deliverable file auto-push, and live CLI AttachServer integration.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

pytest.importorskip("fastapi", reason="gateway tests require fastapi")
pytest.importorskip("agentica_gateway", reason="requires agentica_gateway")

from fastapi.testclient import TestClient

from agentica import peers
from agentica.cli.attach import AttachServer
from agentica.memory.session_log import SessionLog
from agentica_gateway import deps
from agentica_gateway.card_page import (
    PAGE_HEADERS,
    STYLE_SHA,
    CardRegistry,
    card_page,
    credential,
)
from agentica_gateway.channels.base import Channel, ChannelType, Message
from agentica_gateway.main import app
from agentica_gateway.services.agent_service import ChatResult
from agentica_gateway.services import im_bot as im_bot_mod
from agentica_gateway.services.im_bot import (
    ImBotController,
    normalize_im_command_text,
    parse_im_command,
)
from agentica_gateway.services.im_files import (
    collect_auto_push_files,
    extract_output_file_references,
    format_file_size,
)
from agentica.cli.approvals import approval_decision_from_key


class FakeChannel(Channel):
    """In-memory IM channel recording outbound texts and files."""

    def __init__(self) -> None:
        super().__init__()
        self._connected = True
        self.sent: List[Tuple[str, str]] = []
        self.files: List[Tuple[str, str, bytes, str]] = []

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WECHAT

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        self.sent.append((channel_id, content))
        return True

    async def send_file(
        self,
        channel_id: str,
        data: bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
    ) -> bool:
        self.files.append((channel_id, filename, data, mime_type))
        return True


class FakeChannelManager:
    def __init__(self, channel: FakeChannel) -> None:
        self._channel = channel

    def get_channel(self, channel_type: ChannelType):
        return self._channel

    async def send(self, channel_type: ChannelType, channel_id: str, content: str, **kwargs) -> bool:
        return await self._channel.send(channel_id, content, **kwargs)

    async def send_file(
        self,
        channel_type: ChannelType,
        channel_id: str,
        data: bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
    ) -> bool:
        return await self._channel.send_file(
            channel_id, data, filename=filename, mime_type=mime_type,
        )


class FakeAgentService:
    """Stub AgentService recording chat calls and supporting work_dir/profile/cancel."""

    def __init__(self, default_work_dir: str) -> None:
        self.work_dirs: Dict[str, str] = {}
        self.profiles: Dict[str, str] = {}
        self.ask_callbacks: Dict[str, object] = {}
        self.approval_modes: Dict[str, str] = {}
        self.active_sessions: set = set()
        self.cancelled_sessions: List[str] = []
        self.steered: List[Tuple[str, str]] = []
        self.undelivered_steer: Dict[str, List[str]] = {}
        self.chat_calls: List[Tuple[str, str, str]] = []
        self.chat_channels: List[Optional[str]] = []
        self.chat_handler = None
        self.default_work_dir = default_work_dir

    def set_session_work_dir(self, session_id: str, work_dir: str, owner=None) -> None:
        self.work_dirs[session_id] = work_dir

    def get_session_work_dir(self, session_id: str, owner=None) -> str:
        return self.work_dirs.get(session_id, self.default_work_dir)

    def session_log_for(self, session_id: str, owner=None) -> SessionLog:
        wd = self.get_session_work_dir(session_id, owner)
        return SessionLog(session_id=session_id, work_dir=wd, user_id=owner or "default")

    def session_profile_name(self, session_id: str, owner=None) -> str:
        return self.profiles.get(session_id, "")

    def session_model_config(self, session_id: str, owner=None) -> dict:
        prof = self.profiles.get(session_id, "default")
        return {
            "profile": prof,
            "model_provider": "openai",
            "model_name": "gpt-4o",
        }

    async def set_session_profile(self, session_id: str, name: str, owner=None) -> dict:
        self.profiles[session_id] = name
        return self.session_model_config(session_id, owner)

    def set_session_approval_mode(self, session_id: str, mode: str, owner=None) -> None:
        self.approval_modes[session_id] = mode

    def set_session_ask_callback(self, session_id: str, callback, owner=None) -> None:
        self.ask_callbacks[session_id] = callback

    def is_session_active(self, session_id: str, owner=None) -> bool:
        return session_id in self.active_sessions

    def has_cached_session(self, session_id: str, owner=None) -> bool:
        return True

    def cancel_session(self, session_id: str, owner=None) -> bool:
        self.cancelled_sessions.append(session_id)
        return True

    def steer_session(self, session_id: str, text: str, owner=None) -> bool:
        if session_id in self.active_sessions:
            self.steered.append((session_id, text))
            return True
        return False

    def take_undelivered_steer(self, session_id: str, owner=None) -> List[str]:
        return self.undelivered_steer.pop(session_id, [])

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        owner=None,
        media=None,
        channel=None,
    ) -> ChatResult:
        wd = self.get_session_work_dir(session_id, owner)
        self.chat_calls.append((session_id, wd, message))
        self.chat_channels.append(channel)
        if self.chat_handler is not None:
            return await self.chat_handler(message, session_id, wd)
        return ChatResult(
            content=f"done: {message}",
            tool_calls=1,
            session_id=session_id,
            user_id=user_id,
        )


def _msg(text: str, chat_id: str = "wx-chat-1", sender: str = "user-1") -> Message:
    return Message(
        channel=ChannelType.WECHAT,
        channel_id=chat_id,
        sender_id=sender,
        sender_name="Alice",
        content=text,
        message_id=f"msg-{text[:10]}",
    )


class TestCardPageAndRoutes:
    """Zero-JS capability-URL card page (/c/<token>) modeled on agent-autoloop."""

    def test_credential_is_deterministic_and_tamper_evident(self):
        t1 = credential("secret-a", "card", "id1", "100.0")
        t2 = credential("secret-a", "card", "id1", "100.0")
        t3 = credential("secret-b", "card", "id1", "100.0")
        assert t1 == t2
        assert t1 != t3
        assert "=" not in t1

    def test_card_page_renders_two_forms_and_strict_csp(self):
        reg = CardRegistry(secret_provider=lambda: "s3cr3t")
        card = reg.create(
            kind="ask",
            title="Select deployment target <prod>?",
            text="Line 1 & Line 2",
            options=["Staging (recommended)", "Production"],
            recommend=0,
        )
        html = card_page(card)
        assert f"sha256-{STYLE_SHA}" in PAGE_HEADERS["content-security-policy"]
        assert "default-src 'none'" in PAGE_HEADERS["content-security-policy"]
        assert "&lt;prod&gt;" in html
        assert "Line 1 &amp; Line 2" in html
        assert html.count('<form method="post">') == 2
        assert '<button name="choice" value="0" class="best" autofocus>Staging (recommended)</button>' in html
        assert '<input name="text" placeholder="别的想法" aria-label="别的想法" required>' in html

    def test_http_get_and_post_card_prg_without_login(self, monkeypatch):
        reg = CardRegistry(secret_provider=lambda: "test-secret")
        resolved = []
        card = reg.create(
            kind="approval",
            title="Allow command?",
            text="$ git push origin main",
            options=["允许一次 (Allow)", "拒绝 (Deny)"],
            values=["allow", "deny"],
            on_resolve=lambda c: resolved.append(c.selected_value),
        )
        monkeypatch.setattr(deps, "card_registry", reg)
        client = TestClient(app)

        # GET /c/<token> works without any auth cookie/token
        r_get = client.get(f"/c/{card.token}")
        assert r_get.status_code == 200
        assert "Allow command?" in r_get.text
        assert r_get.headers["content-security-policy"] == PAGE_HEADERS["content-security-policy"]

        # POST /c/<token> answers and returns 303 See Other
        r_post = client.post(
            f"/c/{card.token}",
            content="choice=0",
            headers={"content-type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert r_post.status_code == 303
        assert r_post.headers["location"] == f"/c/{card.token}"
        assert resolved == ["allow"]

        # Subsequent GET shows read-only settled outcome
        r_after = client.get(f"/c/{card.token}")
        assert r_after.status_code == 200
        assert "你选了「允许一次 (Allow)」" in r_after.text
        assert "<form" not in r_after.text

        # Second POST is a no-op
        client.post(
            f"/c/{card.token}",
            content="choice=1",
            headers={"content-type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert resolved == ["allow"]
        assert card.selected_value == "allow"

        # Unknown token returns 404
        assert client.get("/c/nonexistent-token").status_code == 404


class TestImFileExtraction:
    """Deliverable file extraction modeled on OpenBitFun bot/mod.rs."""

    def test_extracts_deliverables_and_skips_code_or_fenced_blocks(self, tmp_path):
        chart = tmp_path / "chart.png"
        chart.write_bytes(b"\x89PNG\r\n")
        report = tmp_path / "report.pdf"
        report.write_bytes(b"%PDF-1.4")
        code_file = tmp_path / "main.py"
        code_file.write_text("print(1)\n", encoding="utf-8")

        md = (
            "已完成分析，图表见 ![Chart](chart.png) 和报告 [PDF](report.pdf)。\n"
            f"源码在 [main.py]({code_file})。\n"
            "```markdown\n"
            "![Fake](fake.png)\n"
            "```\n"
            "行内示例 `[ign](ignore.pdf)` 不应触发上传。"
        )
        refs = extract_output_file_references(md)
        assert "chart.png" in refs
        assert "report.pdf" in refs
        assert "fake.png" not in refs
        assert "ignore.pdf" not in refs

        files = collect_auto_push_files(md, str(tmp_path))
        names = [f.name for f in files]
        assert names == ["chart.png", "report.pdf"]
        assert format_file_size(1024 * 1024 * 2) == "2.0MB"
        assert format_file_size(2048) == "2KB"

    def test_blocks_path_traversal_outside_work_dir(self, tmp_path):
        outside = tmp_path / "secret.pdf"
        outside.write_bytes(b"%PDF")
        sub = tmp_path / "sub"
        sub.mkdir()
        md = "Here is [escape](../secret.pdf)"
        assert collect_auto_push_files(md, str(sub)) == []


class TestCommandParsing:
    """Command parsing with Chinese aliases and full-width digit normalization."""

    def test_parses_commands_and_aliases(self):
        assert parse_im_command("/menu").kind == "menu"
        assert parse_im_command("菜单").kind == "menu"
        assert parse_im_command("/switch").kind == "switch"
        assert parse_im_command("切换工作区").kind == "switch"
        assert parse_im_command("/new 帮我跑测试") == parse_im_command("/n 帮我跑测试")
        assert parse_im_command("/new 帮我跑测试").arg == "帮我跑测试"
        assert parse_im_command("/resume").kind == "resume"
        assert parse_im_command("恢复会话").kind == "resume"
        assert parse_im_command("/cancel").kind == "cancel"
        assert parse_im_command("停止").kind == "cancel"
        assert parse_im_command("/verbose").arg == "on"
        assert parse_im_command("简洁").arg == "off"

    def test_parses_fullwidth_and_suffixed_numbers(self):
        assert normalize_im_command_text("１２") == "12"
        cmd1 = parse_im_command("１。")
        assert cmd1.kind == "number" and cmd1.number == 1
        cmd2 = parse_im_command("2)")
        assert cmd2.kind == "number" and cmd2.number == 2
        assert parse_im_command("100").kind == "chat"


class TestImBotControllerFlows:
    """End-to-end controller state machine, shared SessionLog, and CLI attach."""

    @pytest.fixture(autouse=True)
    def isolated_env(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        proj_root = home / "projects"
        cache_dir = home / "cache"
        short_attach = tempfile.mkdtemp(prefix="ag-im-")
        monkeypatch.setenv("AGENTICA_HOME", str(home))
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(proj_root))
        monkeypatch.setenv("AGENTICA_ATTACH_DIR", short_attach)
        monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(cache_dir))
        yield tmp_path
        shutil.rmtree(short_attach, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_menu_workspace_switch_session_resume_and_chat_with_auto_push(self, tmp_path):
        proj_a = tmp_path / "proj_a"
        proj_b = tmp_path / "proj_b"
        proj_a.mkdir()
        proj_b.mkdir()

        # Pre-populate a CLI session in proj_b
        log_b = SessionLog(session_id="sess-b-11111111", work_dir=str(proj_b), user_id="default")
        log_b.set_source("cli")
        log_b.set_name("修复缓存命中率")
        log_b.append("user", "分析一下缓存命中率")
        log_b.append("assistant", "已经定位到问题")

        ch = FakeChannel()
        cm = FakeChannelManager(ch)
        svc = FakeAgentService(default_work_dir=str(proj_a))
        bot = ImBotController(
            channel_manager=cm,
            agent_service=svc,
            state_file=tmp_path / "im_state.json",
        )

        # 1. /menu opens menu and sets pending options
        await bot.handle(_msg("菜单"))
        assert "回复数字选择操作" in ch.sent[-1][1]

        # 2. Reply '3' (selects /switch from menu) -> opens workspace picker
        await bot.handle(_msg("３"))
        assert "选择要控制的项目工作区" in ch.sent[-1][1]
        assert "proj_b" in ch.sent[-1][1]
        # Every choice is a single line carrying its full path: an indented
        # continuation line under "N." gets folded into the next list item by
        # WeChat's Markdown and glued to the next project's name.
        picker_lines = ch.sent[-1][1].splitlines()[1:]
        assert picker_lines and all(line.lstrip()[0].isdigit() for line in picker_lines)
        assert any(line.endswith(f"· {proj_b.resolve()}") for line in picker_lines)

        # Find which index proj_b is in the workspace picker
        state = bot.state_for(_msg("noop"))
        idx_b = [ Path(p).name for p in state.pending.options ].index("proj_b") + 1

        # 3. Reply with proj_b's number -> switches to proj_b and binds its session
        await bot.handle(_msg(f"{idx_b}。"))
        assert "已切换工作区" in ch.sent[-1][1]
        assert state.work_dir == str(proj_b.resolve())
        assert state.session_id == "sess-b-11111111"

        # 4. /resume lists sessions in proj_b
        await bot.handle(_msg("/resume"))
        assert "修复缓存命中率" in ch.sent[-1][1]
        await bot.handle(_msg("1"))
        assert "已挂载会话" in ch.sent[-1][1]

        # 5. Without pending menu, sending a bare number '1' goes to the agent as chat
        plot_path = proj_b / "result.png"
        plot_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        async def _handler(message, sid, wd):
            return ChatResult(
                content=f"收到指令({message})，生成图表 ![图](result.png)",
                tool_calls=2,
                session_id=sid,
            )

        svc.chat_handler = _handler
        await bot.handle(_msg("1"))
        # Wait for the background session queue task to finish
        task = bot._turn_tasks.get("sess-b-11111111")
        if task:
            await task

        assert svc.chat_calls[-1] == ("sess-b-11111111", str(proj_b.resolve()), "1")
        assert "收到指令(1)" in ch.sent[-1][1]
        # Deliverable image was automatically pushed over the channel!
        assert len(ch.files) == 1
        assert ch.files[0][1] == "result.png"
        assert ch.files[0][3] == "image/png"
        # The turn is recorded under the channel it came in through.
        assert svc.chat_channels[-1] == "wechat"

    @pytest.mark.asyncio
    async def test_gateway_turn_approval_resolved_via_card_link(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        proj.mkdir()
        ch = FakeChannel()
        cm = FakeChannelManager(ch)
        svc = FakeAgentService(default_work_dir=str(proj))
        reg = CardRegistry(secret_provider=lambda: "card-sec")
        monkeypatch.setattr(deps, "card_registry", reg)
        monkeypatch.setenv("AGENTICA_PUBLIC_URL", "http://192.168.1.20:8881")
        bot = ImBotController(
            channel_manager=cm,
            agent_service=svc,
            card_registry=reg,
            state_file=tmp_path / "im_state.json",
        )

        from agentica.agent.approvals import PendingApproval
        from agentica_gateway.services import live_turn

        async def _chat_needing_approval(message, sid, wd):
            turn = live_turn.active(sid, "default")
            assert turn is not None
            pending = PendingApproval(
                tool_call_id="tc-deploy-1",
                name="execute",
                arguments={"command": "git push origin main"},
                question="Would you like to run the following command?",
                preview="git push origin main",
                options=("allow", "deny"),
            )
            fut = turn.approvals.wait(pending)
            turn.publish(live_turn.approval_event(pending))
            decision = await asyncio.wait_for(fut, timeout=5.0)
            return ChatResult(content=f"approval={decision}", tool_calls=1, session_id=sid)

        svc.chat_handler = _chat_needing_approval
        await bot.handle(_msg("部署上线"))
        state = bot.state_for(_msg("noop"))

        # Wait until the approval card is published to IM
        for _ in range(50):
            if state.has_active_pending() and state.pending.kind == "interaction":
                break
            await asyncio.sleep(0.02)

        assert state.pending is not None
        card_token = state.pending.card_token
        assert f"http://192.168.1.20:8881/c/{card_token}" in ch.sent[-1][1]

        # User clicks "允许一次 (Allow)" (choice=0) on the mobile webview /c/<token>
        client = TestClient(app)
        resp = client.post(
            f"/c/{card_token}",
            content="choice=0",
            headers={"content-type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

        task = bot._turn_tasks.get(state.session_id)
        if task:
            await task
        assert "approval=allow" in ch.sent[-1][1]

    @pytest.mark.asyncio
    async def test_loopback_gateway_sends_no_card_link_the_phone_cannot_open(self, tmp_path, monkeypatch):
        """A 127.0.0.1 link opens the phone itself; offer numbered replies only."""
        proj = tmp_path / "proj"
        proj.mkdir()
        ch = FakeChannel()
        svc = FakeAgentService(default_work_dir=str(proj))
        monkeypatch.delenv("AGENTICA_PUBLIC_URL", raising=False)
        monkeypatch.delenv("PUBLIC_URL", raising=False)
        monkeypatch.setattr(im_bot_mod.settings, "host", "127.0.0.1")
        bot = ImBotController(
            channel_manager=FakeChannelManager(ch), agent_service=svc,
            state_file=tmp_path / "im_state.json",
        )
        state = bot.state_for(_msg("noop"))
        await bot._publish_interaction(
            state, interaction_kind="approval", title="允许执行 execute？", text="$ git push",
            options=["允许一次 (Allow)", "拒绝 (Deny)"], values=["allow", "deny"],
            session_id=state.session_id, request_id="tc-1", resolve_fn=lambda _v: True,
        )
        sent = ch.sent[-1][1]
        assert "/c/" not in sent
        assert "1. 允许一次 (Allow)" in sent and "回复数字" in sent
        await bot.handle(_msg("/status"))
        assert "卡片链接：未启用" in ch.sent[-1][1]

    @pytest.mark.asyncio
    async def test_help_and_status_show_the_session_model(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        ch = FakeChannel()
        bot = ImBotController(
            channel_manager=FakeChannelManager(ch),
            agent_service=FakeAgentService(default_work_dir=str(proj)),
            state_file=tmp_path / "im_state.json",
        )
        await bot.handle(_msg("/help"))
        assert "模型：openai/gpt-4o (profile: default)" in ch.sent[-1][1]
        assert "/switch" in ch.sent[-1][1] and "self_manage" in ch.sent[-1][1]
        await bot.handle(_msg("/status"))
        assert "模型：openai/gpt-4o" in ch.sent[-1][1]

    @pytest.mark.asyncio
    async def test_live_cli_attach_turn_and_interaction_bridge(self, tmp_path):
        """When the bound session is open in a local CLI terminal, IM attaches directly to it."""
        proj = tmp_path / "cli_proj"
        proj.mkdir()

        injected: List[str] = []
        running = [False]
        last_ans = [""]
        pending_inter: List[dict] = [None]  # type: ignore[list-item]
        answered_inter: List[Tuple[str, str]] = []

        peer_session = peers.PeerSession(
            name="cli-proj-01",
            cwd=str(proj),
            session_id="cli-live-sess-9999",
        )
        peer_session.publish()

        def _inject(text: str) -> str:
            injected.append(text)

            def _sim_cli_run():
                import time as _t
                _t.sleep(0.03)
                running[0] = True
                _t.sleep(0.05)
                # Simulate CLI parking on ask_user_question mid-run
                pending_inter[0] = {
                    "requestId": "req-cli-ask-1",
                    "kind": "ask",
                    "prompt": "用哪个分支发布？",
                    "options": ["main (推荐)", "release"],
                }
                deadline = _t.monotonic() + 5.0
                while pending_inter[0] is not None and _t.monotonic() < deadline:
                    _t.sleep(0.03)
                last_ans[0] = f"已在终端完成({text})，分支选择={answered_inter[-1][0] if answered_inter else 'none'}"
                running[0] = False

            import threading
            threading.Thread(target=_sim_cli_run, daemon=True).start()
            return "queued"

        def _answer_inter(ans: str, req_id):
            answered_inter.append((ans, req_id or ""))
            pending_inter[0] = None
            return True

        server = AttachServer(
            peer_session.peer_id,
            inject=_inject,
            is_running=lambda: running[0],
            session_id=lambda: "cli-live-sess-9999",
            answer=lambda: last_ans[0],
            answer_interaction=_answer_inter,
            grace=5.0,
            snapshot=lambda: {
                "cwd": str(proj),
                "busy": running[0],
                "pendingInteraction": pending_inter[0],
            },
        )
        assert server.start()
        peer_session.publish(attach_socket=str(server.path))

        try:
            ch = FakeChannel()
            cm = FakeChannelManager(ch)
            svc = FakeAgentService(default_work_dir=str(proj))
            bot = ImBotController(
                channel_manager=cm,
                agent_service=svc,
                state_file=tmp_path / "im_state.json",
            )

            # First message auto-binds to the live CLI session in proj
            await bot.handle(_msg("开始构建发布"))
            state = bot.state_for(_msg("noop"))
            assert state.session_id == "cli-live-sess-9999"

            # Wait for _watch_interactions to surface the CLI's ask_user_question to IM
            for _ in range(60):
                if state.has_active_pending() and state.pending.kind == "interaction":
                    break
                await asyncio.sleep(0.05)

            assert state.pending is not None
            assert "用哪个分支发布？" in ch.sent[-1][1]

            # User replies '1' in IM -> forwarded via session/answer to the live CLI!
            await bot.handle(_msg("1"))

            task = bot._turn_tasks.get("cli-live-sess-9999")
            if task:
                await task

            assert injected == ["开始构建发布"]
            # A question's answer is free text handed to the model, so picking
            # "1" in IM must send the option's text, not the literal "1".
            assert answered_inter == [("main (推荐)", "req-cli-ask-1")]
            assert "已在终端完成(开始构建发布)" in ch.sent[-1][1]
            # AgentService.chat was NOT called because the live CLI handled the turn!
            assert svc.chat_calls == []
        finally:
            server.stop()
            peer_session.unpublish()

    @pytest.mark.asyncio
    async def test_shell_hook_endpoint_forwards_to_im_and_resolves(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        ch = FakeChannel()
        cm = FakeChannelManager(ch)
        svc = FakeAgentService(default_work_dir=str(proj))
        reg = CardRegistry(secret_provider=lambda: "hook-sec")
        bot = ImBotController(
            channel_manager=cm,
            agent_service=svc,
            card_registry=reg,
            state_file=tmp_path / "im_state.json",
        )
        # Prime an active IM chat state
        await bot.handle(_msg("/status"))

        hook_task = asyncio.create_task(
            bot.handle_hook_event(
                {
                    "hook_event_name": "needs.approval",
                    "session_id": "sess-hook-1",
                    "request_id": "req-hook-77",
                    "tool_name": "execute",
                    "question": "Run rm -rf build/?",
                    "preview": "rm -rf build/",
                    "options": ["allow", "deny"],
                },
                wait_seconds=5.0,
            )
        )
        await asyncio.sleep(0.05)
        assert "Run rm -rf build/?" in ch.sent[-1][1]

        # Reply "y" directly in IM
        await bot.handle(_msg("y"))
        hook_res = await hook_task
        assert hook_res["decision"] == "allow"
        assert hook_res["request_id"] == "req-hook-77"

    @pytest.mark.asyncio
    async def test_im_turn_streams_to_live_turn_and_bumps_pulse(self, tmp_path):
        """IM turns stream events into LiveTurn and bump /api/sessions/pulse for real-time Web sync."""
        from agentica_gateway.services import live_turn

        live_turn.reset()
        proj = tmp_path / "proj_stream"
        proj.mkdir()
        ch = FakeChannel()
        cm = FakeChannelManager(ch)
        svc = FakeAgentService(default_work_dir=str(proj))

        mid_turn_pulse = {}
        mid_turn_public = {}

        async def _fake_chat_stream(
            message,
            session_id,
            user_id="default",
            owner=None,
            on_content=None,
            on_tool_call=None,
            on_tool_result=None,
            on_thinking=None,
            media=None,
            include_usage=True,
            channel=None,
        ):
            assert include_usage is False
            assert channel == "wechat"
            svc.chat_calls.append((session_id, svc.get_session_work_dir(session_id, owner), message))
            mid_turn_pulse.update(live_turn.pulse("default"))
            active = live_turn.active(session_id, "default")
            assert active is not None
            mid_turn_public.update(active.public())
            if on_thinking:
                on_thinking("先思考一下")
            if on_tool_call:
                on_tool_call("apply_patch", {"file": "a.py"}, "tc-1")
            if on_tool_result:
                on_tool_result("apply_patch", "patched", {"files": ["a.py"]})
            if on_content:
                on_content("已改好 a.py")
            return ChatResult(content="已改好 a.py", tool_calls=1, session_id=session_id)

        svc.chat_stream = _fake_chat_stream  # type: ignore[attr-defined]
        bot = ImBotController(
            channel_manager=cm,
            agent_service=svc,
            state_file=tmp_path / "im_state.json",
        )

        v0 = live_turn.pulse("default")["ver"]
        await bot.handle(_msg("帮我改一下 a.py"))
        state = bot.state_for(_msg("noop"))
        task = bot._turn_tasks.get(state.session_id)
        if task:
            await task

        # 1. Mid-turn pulse had incremented ver and listed the session as active with user_message
        assert mid_turn_pulse["ver"] == v0 + 1
        assert state.session_id in mid_turn_pulse["active"]
        assert mid_turn_public["user_message"] == "帮我改一下 a.py"

        # 2. LiveTurn recorded thinking, tool_call, tool_result, content, and done events for Web SSE
        finished_turn = live_turn.get(state.session_id, "default")
        assert finished_turn is not None
        event_names = [ev["event"] for ev in finished_turn.events]
        assert "thinking" in event_names
        assert "tool_call" in event_names
        assert "tool_result" in event_names
        assert "content" in event_names
        assert "done" in event_names

        # 3. Post-turn pulse incremented ver again and cleared active
        after_pulse = live_turn.pulse("default")
        assert after_pulse["ver"] == v0 + 2
        assert after_pulse["active"] == []
        assert "已改好 a.py" in ch.sent[-1][1]

    @pytest.mark.asyncio
    async def test_cli_approval_answer_is_translated_to_an_accepted_key(self, tmp_path):
        """An approval forwarded from a live CLI must answer in a form that CLI accepts.

        The card and the numbered menu answer in decision keys ("allow"/"deny").
        The terminal's ``session/answer`` only understands a letter or a 1-based
        index, so handing it the key name gets ``accepted: false`` and the
        approval silently stays open (the CLI keeps waiting, IM says "已处理").
        """
        proj = tmp_path / "cli_approval"
        proj.mkdir()

        running = [False]
        pending_inter: List[dict] = [None]  # type: ignore[list-item]
        answered: List[Tuple[str, str]] = []
        accepted_results: List[bool] = []

        peer_session = peers.PeerSession(
            name="cli-approval-01",
            cwd=str(proj),
            session_id="cli-approval-sess-1",
        )
        peer_session.publish()

        def _inject(text: str) -> str:
            def _sim_cli_run():
                import time as _t
                _t.sleep(0.03)
                running[0] = True
                _t.sleep(0.05)
                pending_inter[0] = {
                    "requestId": "req-cli-approval-1",
                    "kind": "approval",
                    "prompt": "允许执行 rm -rf build/ ？",
                    # What ``_pending_interaction_snapshot`` publishes: the
                    # visible decisions in allow → allow_prefix → deny order.
                    "options": ["allow", "allow_prefix", "deny"],
                }
                deadline = _t.monotonic() + 5.0
                while pending_inter[0] is not None and _t.monotonic() < deadline:
                    _t.sleep(0.03)
                running[0] = False

            import threading
            threading.Thread(target=_sim_cli_run, daemon=True).start()
            return "queued"

        def _answer_inter(ans: str, req_id):
            # Mirror the CLI: it parses the answer against its own option list.
            decision = approval_decision_from_key(ans, ["allow", "allow_prefix", "deny"])
            answered.append((ans, req_id or ""))
            accepted_results.append(decision is not None)
            if decision is None:
                return False
            pending_inter[0] = None
            return True

        server = AttachServer(
            peer_session.peer_id,
            inject=_inject,
            is_running=lambda: running[0],
            session_id=lambda: "cli-approval-sess-1",
            answer_interaction=_answer_inter,
            grace=5.0,
            snapshot=lambda: {
                "cwd": str(proj),
                "busy": running[0],
                "pendingInteraction": pending_inter[0],
            },
        )
        assert server.start()
        peer_session.publish(attach_socket=str(server.path))

        try:
            ch = FakeChannel()
            bot = ImBotController(
                channel_manager=FakeChannelManager(ch),
                agent_service=FakeAgentService(default_work_dir=str(proj)),
                state_file=tmp_path / "im_state.json",
            )
            await bot.handle(_msg("开始构建"))
            state = bot.state_for(_msg("noop"))

            for _ in range(80):
                if state.has_active_pending() and state.pending.kind == "interaction":
                    break
                await asyncio.sleep(0.05)
            assert state.pending is not None

            # Deny is option 3 in the card/menu the user sees.
            await bot.handle(_msg("3"))
            task = bot._turn_tasks.get("cli-approval-sess-1")
            if task:
                await task

            assert answered, "the CLI never received an answer"
            assert accepted_results == [True], (
                f"CLI rejected the IM answer {answered!r}; it must be a letter or a "
                "1-based index among the options it published"
            )
            # Option 3 is 'deny' — the ordered list the CLI published decides.
            assert approval_decision_from_key(answered[0][0], ["allow", "allow_prefix", "deny"]) == "deny"
        finally:
            server.stop()
            peer_session.unpublish()


