# -*- coding: utf-8 -*-
"""The local token gate, the published runtime record, and dying with a parent.

The suite-wide conftest turns the gate off (every other gateway test is about
what a route does). Everything here turns it back on, because the whole point
is what happens to a request that does not carry the token.
"""
import asyncio
import json
import os
import stat
import unicodedata

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

TOKEN = "test-token-not-a-real-secret"

#: The one web account. Not `settings.default_user_id` — that names a data
#: partition on disk and deliberately has nothing to do with signing in.
ADMIN_ID = "admin"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """A gateway with the gate ON and a pinned token.

    Startup seeds the `admin` account, so this client is a *fresh install*: a
    generated password exists and is flagged as generated. A test that wants no
    password at all has to clear it and say why.
    """
    from agentica.gateway import auth, deps, runtime
    from agentica.gateway.main import app

    monkeypatch.setenv("GATEWAY_AUTH", "true")
    monkeypatch.setenv("AGENTICA_GATEWAY_TOKEN", TOKEN)
    monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "gateway" / "runtime.json")
    auth.reset_token_for_tests()

    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc.list_sessions = MagicMock(return_value=[])
    with TestClient(app, raise_server_exceptions=False) as c:
        original = deps.agent_service
        deps.agent_service = svc
        yield c
        deps.agent_service = original


def _auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


class TestGate:
    def test_api_without_a_token_is_refused(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 401
        # The refusal names where the token lives, not the token.
        assert TOKEN not in resp.text
        assert "runtime.json" in resp.json()["detail"]

    def test_bearer_header_is_accepted(self, client):
        assert client.get("/api/status", headers=_auth()).status_code == 200

    def test_dedicated_header_is_accepted(self, client):
        assert client.get("/api/status", headers={"X-Agentica-Token": TOKEN}).status_code == 200

    def test_wrong_token_is_refused(self, client):
        assert client.get("/api/status", headers=_auth("nope")).status_code == 401

    def test_query_token_is_swapped_for_a_session(self, client):
        """The Jupyter hop: the printed URL works once, then the bookmark does.

        Without it the user would have to keep the token in the address bar
        forever, and every SPA fetch would have to carry it. What lands in the
        cookie is a *session*, not the token — see
        `test_the_cookie_is_not_the_machine_token`.
        """
        first = client.get(f"/chat?token={TOKEN}")
        assert first.status_code == 200
        assert client.cookies.get("agentica_session")
        # Same client, no token anywhere in the request now.
        assert client.get("/chat").status_code == 200
        assert client.get("/api/status").status_code == 200

    def test_the_cookie_is_not_the_machine_token(self, client):
        """The cookie used to hold the token itself, which made a leaked cookie
        the master credential with no way to revoke one browser — and tied
        every browser's login to a value that changes on restart."""
        client.get(f"/chat?token={TOKEN}")
        cookie = client.cookies.get("agentica_session")
        assert cookie and cookie != TOKEN

    def test_a_header_token_mints_no_session(self, client):
        """A script polling with a header must not write a session per call."""
        from agentica.gateway import accounts

        for _ in range(3):
            assert client.get("/api/status", headers=_auth()).status_code == 200
        assert client.cookies.get("agentica_session") is None
        assert accounts.store()._read()["sessions"] == {}

    def test_shell_route_refuses_by_sending_the_browser_to_the_login_page(self, client):
        """A browser is the only thing that opens /chat, and it always has
        somewhere to go now: first start seeds an account. The hand-written
        "you need a token" HTML page is gone with the flow it described."""
        resp = client.get("/chat", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login?next=/chat"

    @pytest.mark.parametrize("path", ["/", "/health", "/api/health"])
    def test_probes_stay_open(self, client, path):
        """A desktop shell polls one of these before it knows the token."""
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("path", ["/favicon.ico", "/favicon.png"])
    def test_favicon_stays_open(self, client, path):
        """The tab icon is fetched from the origin root, including /login."""
        resp = client.get(path)
        assert resp.status_code == 200
        assert not resp.headers["content-type"].startswith("application/json")

    def test_third_party_webhook_stays_open(self, client):
        """Feishu signs its own callbacks and cannot carry our token; gating
        /webhook would silently take IM offline."""
        resp = client.post("/webhook/feishu", json={"type": "url_verification", "challenge": "abc"})
        assert resp.status_code != 401

    def test_preflight_is_not_gated(self, client):
        """A CORS preflight carries no credential by specification."""
        resp = client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_a_foreign_origin_gets_no_cors_grant(self, client):
        """With the token in a cookie, echoing any origin back would let a page
        you happened to visit read this API. Vite's dev origin still may."""
        allowed = client.get(
            "/api/status", headers={**_auth(), "Origin": "http://127.0.0.1:5173"}
        )
        assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

        evil = client.get("/api/status", headers={**_auth(), "Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in evil.headers

    def test_env_token_pins_the_value(self, client):
        from agentica.gateway import auth
        assert auth.get_token() == TOKEN


class TestPasswordLogin:
    """The password half: what a browser does when the token is not at hand."""

    def test_status_is_readable_before_signing_in(self, client):
        """The login page has to render before anybody is authorized."""
        body = client.get("/api/auth/status").json()
        assert body["auth_enabled"] is True
        assert body["authenticated"] is False
        # A fresh gateway: seeded, and honest that it is still the generated one.
        assert body["password_set"] is True
        assert body["password_is_initial"] is True
        assert body["account_id"] == ADMIN_ID
        assert body["min_password_length"] == 6

    def test_the_seeded_password_is_the_way_in(self, client):
        """What a user does on a fresh install: read the password out of the
        startup banner and type it."""
        from agentica.gateway import accounts

        password = accounts.store().read_initial_password()
        assert password and len(password) >= 6
        assert client.post("/api/auth/login", json={"password": password}).status_code == 200
        assert client.get("/api/status").status_code == 200

    def test_seeding_is_idempotent(self, client):
        """It runs on every boot, so a second call must not mint a new password
        and lock the user out of the one they wrote down."""
        from agentica.gateway import accounts

        first = accounts.store().read_initial_password()
        assert accounts.store().seed_admin() is None
        assert accounts.store().read_initial_password() == first

    def test_the_generated_password_is_readable_but_only_by_its_owner(self, client):
        """It is kept in plaintext on purpose — a gateway started detached or by
        the desktop shell prints to a log nobody reads — so the mode matters."""
        from agentica.gateway import accounts

        path = accounts.store().initial_password_path
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_changing_it_retires_the_plaintext_copy(self, client):
        from agentica.gateway import accounts

        path = accounts.store().initial_password_path
        assert path.is_file()
        accounts.store().set_password(ADMIN_ID, "chosen one")
        assert not path.exists()
        assert accounts.store().password_is_initial() is False

    def test_login_page_is_reachable_while_signed_out(self, client):
        assert client.get("/login").status_code == 200

    def test_login_then_api_works_without_any_token(self, client):
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        assert client.post(
            "/api/auth/login", json={"password": "correct horse battery"}
        ).status_code == 200
        assert client.cookies.get("agentica_session")
        assert client.get("/api/status").status_code == 200

    def test_wrong_password_is_refused(self, client):
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        assert client.post("/api/auth/login", json={"password": "wrong"}).status_code == 401
        assert client.get("/api/status").status_code == 401

    def test_login_without_a_password_configured_says_so(self, client):
        """A 401 here would read as "wrong password" on a gateway that has none.

        Only reachable by clearing it explicitly now that first start seeds one.
        """
        from agentica.gateway import accounts

        accounts.store().clear_password(ADMIN_ID)
        resp = client.post("/api/auth/login", json={"password": "anything"})
        assert resp.status_code == 409
        assert "--set-password" in resp.json()["detail"]

    def test_repeated_failures_are_throttled(self, client):
        """The password is guessable at machine speed, and this endpoint is
        reachable from the LAN in the deployment that required a password."""
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        codes = [
            client.post("/api/auth/login", json={"password": "no"}).status_code
            for _ in range(7)
        ]
        assert codes[:5] == [401] * 5
        assert 429 in codes[5:]
        # And a throttled window says how long, rather than looking broken.
        resp = client.post("/api/auth/login", json={"password": "no"})
        assert resp.status_code == 429
        assert int(resp.headers["retry-after"]) >= 1

    def test_a_session_survives_a_restart(self, client, monkeypatch, tmp_path):
        """The whole reason sessions are on disk: the machine token is per
        process, so a cookie holding it sent the user back to the terminal
        after every restart."""
        from agentica.gateway import accounts, auth, deps, runtime
        from agentica.gateway.main import app

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        client.post("/api/auth/login", json={"password": "correct horse battery"})
        cookie = client.cookies.get("agentica_session")

        # A "restart": a new process token, same auth.json.
        monkeypatch.delenv("AGENTICA_GATEWAY_TOKEN", raising=False)
        auth.reset_token_for_tests()
        monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "restarted.json")
        svc = MagicMock()
        svc._ensure_initialized = AsyncMock()
        svc.list_sessions = MagicMock(return_value=[])
        with TestClient(app, raise_server_exceptions=False) as fresh:
            deps.agent_service = svc
            fresh.cookies.set("agentica_session", cookie)
            assert fresh.get("/api/status").status_code == 200

    def test_logout_kills_the_session_server_side(self, client):
        """Clearing the cookie is not enough: a copied cookie must stop too."""
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        client.post("/api/auth/login", json={"password": "correct horse battery"})
        stolen = client.cookies.get("agentica_session")
        client.post("/api/auth/logout")

        client.cookies.set("agentica_session", stolen)
        assert client.get("/api/status").status_code == 401

    def test_shell_route_sends_a_signed_out_browser_to_the_login_page(self, client):
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        resp = client.get("/chat", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login?next=/chat"

    def test_the_startup_notice_carries_the_password_and_not_the_token(self, client):
        """The banner used to print `/chat?token=…`, which changed on every
        restart and was the only way in. A password is what a person can be
        told once."""
        from agentica.gateway.main import _sign_in_notice

        notice = _sign_in_notice(ADMIN_ID, "abcd-efgh")
        assert "admin / abcd-efgh" in notice
        assert TOKEN not in notice
        assert "token" not in notice.lower()
        # The frame lines up even though one line is Chinese (CJK glyphs are two
        # terminal columns wide, so len() would leave it ragged).
        assert len({len(line.encode("utf-8")) for line in notice.splitlines()}) > 1
        widths = {sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in line)
                  for line in notice.splitlines()}
        assert len(widths) == 1

    def test_changing_the_password_signs_other_browsers_out(self, client):
        """A password change is what a user does after "somebody may have my
        cookie"; keeping those sessions alive would defeat the point."""
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        client.post("/api/auth/login", json={"password": "correct horse battery"})
        other = accounts.store().open_session(ADMIN_ID, "password")

        assert client.post(
            "/api/auth/password",
            json={"old_password": "correct horse battery", "password": "a new long one"},
        ).status_code == 200
        assert accounts.store().read_session(other) is None
        # The browser that made the change keeps working.
        assert client.get("/api/status").status_code == 200

    def test_a_token_holder_may_change_it_without_the_old_one(self, client):
        """They already proved they can read a 0600 file on this machine — which
        is where the generated password is kept anyway. This is the desktop
        shell's path: it signs itself in with the token and shows no terminal,
        so requiring the printed password would strand it."""
        assert client.post(
            "/api/auth/password", headers=_auth(), json={"password": "first password"}
        ).status_code == 200
        assert client.post(
            "/api/auth/login", json={"password": "first password"}
        ).status_code == 200

    def test_a_password_session_needs_the_old_one(self, client):
        from agentica.gateway import accounts

        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        client.post("/api/auth/login", json={"password": "correct horse battery"})
        resp = client.post(
            "/api/auth/password", json={"password": "another long one", "old_password": "nope"}
        )
        assert resp.status_code == 400

    def test_a_short_password_is_refused(self, client):
        """Six, because scrypt and the login throttle are the real defences and a
        longer minimum only pushes people toward a password they already use."""
        resp = client.post("/api/auth/password", headers=_auth(), json={"password": "12345"})
        assert resp.status_code == 400
        assert "6" in resp.json()["detail"]
        assert client.post(
            "/api/auth/password", headers=_auth(), json={"password": "123456"}
        ).status_code == 200


class TestAccountStore:
    """The store on its own — no HTTP, so a failure here is unambiguous."""

    def store(self, tmp_path):
        from agentica.gateway.accounts import AccountStore

        return AccountStore(tmp_path / "auth.json")

    def test_the_file_holds_a_password_hash_so_it_is_0600(self, tmp_path):
        st = self.store(tmp_path)
        st.set_password(ADMIN_ID, "correct horse battery")
        assert stat.S_IMODE(st.path.stat().st_mode) == 0o600
        assert not list(st.path.parent.glob("*.tmp"))

    def test_the_plaintext_is_never_written(self, tmp_path):
        st = self.store(tmp_path)
        st.set_password(ADMIN_ID, "correct horse battery")
        assert "correct horse battery" not in st.path.read_text()

    def test_the_session_token_is_never_written(self, tmp_path):
        """Only its sha256 goes to disk, so a readable auth.json cannot be
        replayed as a session."""
        st = self.store(tmp_path)
        token = st.open_session(ADMIN_ID, "password")
        assert token not in st.path.read_text()
        assert st.read_session(token) is not None

    def test_an_expired_session_is_rejected_and_swept(self, tmp_path):
        import json
        from datetime import datetime, timedelta, timezone

        st = self.store(tmp_path)
        token = st.open_session(ADMIN_ID, "password")
        data = json.loads(st.path.read_text())
        (digest,) = data["sessions"].keys()
        data["sessions"][digest]["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        st.path.write_text(json.dumps(data))

        assert st.read_session(token) is None
        assert json.loads(st.path.read_text())["sessions"] == {}

    def test_a_session_close_to_expiry_is_renewed(self, tmp_path):
        """A browser used daily is never signed out; one left for a week is."""
        import json
        from datetime import datetime, timedelta, timezone

        st = self.store(tmp_path)
        token = st.open_session(ADMIN_ID, "password")
        data = json.loads(st.path.read_text())
        (digest,) = data["sessions"].keys()
        soon = datetime.now(timezone.utc) + timedelta(hours=2)
        data["sessions"][digest]["expires_at"] = soon.isoformat()
        st.path.write_text(json.dumps(data))

        assert st.read_session(token) is not None
        renewed = json.loads(st.path.read_text())["sessions"][digest]["expires_at"]
        assert datetime.fromisoformat(renewed) > soon + timedelta(days=1)

    def test_a_hand_edited_hash_is_a_failed_login_not_a_crash(self, tmp_path):
        from agentica.gateway.accounts import verify_password

        assert verify_password("x", "garbage") is False
        assert verify_password("x", "") is False
        assert verify_password("x", "scrypt$a$b$c$d$e") is False

    def test_a_missing_file_reads_as_no_accounts(self, tmp_path):
        st = self.store(tmp_path)
        assert st.has_password() is False
        assert st.read_session("whatever") is None

    def test_throttling_does_not_leak_which_ids_exist(self, tmp_path):
        from agentica.gateway.accounts import LoginThrottled

        st = self.store(tmp_path)
        st.set_password(ADMIN_ID, "correct horse battery")
        for _ in range(5):
            assert st.check_password("ghost", "x") is False
        with pytest.raises(LoginThrottled):
            st.check_password("ghost", "x")


class TestCsrf:
    """SameSite=Lax is the real defence; this is the second line."""

    def test_a_form_content_type_cannot_ride_the_cookie(self, client):
        client.get(f"/chat?token={TOKEN}")
        resp = client.post(
            "/api/profile/switch",
            content="name=x",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 415

    def test_multipart_upload_passes_with_the_client_header(self, client):
        """/api/upload genuinely needs multipart, so the header is what a form
        cannot forge — not the body type."""
        client.get(f"/chat?token={TOKEN}")
        resp = client.post(
            "/api/upload",
            files={"file": ("a.txt", b"hi", "text/plain")},
            headers={"X-Agentica-Client": "web"},
        )
        assert resp.status_code != 415

    def test_a_header_token_is_never_held_to_form_rules(self, client):
        """`curl -d` sends form-urlencoded by default, and a form cannot set
        Authorization — so a script must not be caught by this."""
        resp = client.post(
            "/api/profile/switch",
            content="name=x",
            headers={**_auth(), "Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code != 415


class TestOpenBindGuard:
    def test_loopback_needs_no_password(self):
        from agentica.gateway.main import _refuse_open_bind

        assert _refuse_open_bind("127.0.0.1") is None

    def test_lan_bind_without_a_password_is_refused(self, monkeypatch):
        """The token is printed in a terminal and cannot be rotated without a
        restart; a password is the credential a person can change."""
        from agentica.gateway.main import _refuse_open_bind

        monkeypatch.setenv("GATEWAY_AUTH", "true")
        assert "set-password" in _refuse_open_bind("0.0.0.0")

    def test_lan_bind_with_only_the_generated_password_is_refused(self, monkeypatch):
        """Seeding must not quietly satisfy this guard. The generated password
        is printed to a log and kept in plaintext so a locked-out owner can get
        in on loopback — neither is true of a credential facing a network."""
        from agentica.gateway import accounts
        from agentica.gateway.main import _refuse_open_bind

        monkeypatch.setenv("GATEWAY_AUTH", "true")
        accounts.store().seed_admin()
        assert accounts.store().has_password() is True
        assert "generated" in _refuse_open_bind("0.0.0.0")

    def test_lan_bind_with_a_chosen_password_is_allowed(self, monkeypatch):
        from agentica.gateway import accounts
        from agentica.gateway.main import _refuse_open_bind

        monkeypatch.setenv("GATEWAY_AUTH", "true")
        accounts.store().seed_admin()
        accounts.store().set_password(ADMIN_ID, "correct horse battery")
        assert _refuse_open_bind("0.0.0.0") is None

    def test_gate_off_is_an_explicit_instruction_and_is_obeyed(self, monkeypatch):
        """An explicit argument is intent. It gets a loud warning, not a veto."""
        from agentica.gateway.main import _refuse_open_bind

        monkeypatch.setenv("GATEWAY_AUTH", "false")
        assert _refuse_open_bind("0.0.0.0") is None


class TestDesktopShutdown:
    def test_a_session_cannot_stop_the_process(self, client):
        """A browser tab must not be able to kill the process behind it."""
        client.get(f"/chat?token={TOKEN}")
        assert client.post("/api/desktop/shutdown").status_code == 401

    def test_the_token_holder_gets_a_graceful_stop(self, client, monkeypatch):
        """This exists for Windows, where killing a child is a hard
        TerminateProcess and the channels would never disconnect."""
        from agentica.gateway import main as gw_main

        server = MagicMock()
        server.should_exit = False
        monkeypatch.setattr(gw_main, "_server", server)
        assert client.post("/api/desktop/shutdown", headers=_auth()).status_code == 200
        assert server.should_exit is True

    def test_without_a_server_object_it_declines(self, client, monkeypatch):
        from agentica.gateway import main as gw_main

        monkeypatch.setattr(gw_main, "_server", None)
        assert client.post("/api/desktop/shutdown", headers=_auth()).status_code == 503


class TestGateOff:
    def test_gateway_auth_false_opens_everything(self, monkeypatch, tmp_path):
        from agentica.gateway import deps, runtime
        from agentica.gateway.main import app

        monkeypatch.setenv("GATEWAY_AUTH", "false")
        monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "runtime.json")
        svc = MagicMock()
        svc._ensure_initialized = AsyncMock()
        svc.list_sessions = MagicMock(return_value=[])
        with TestClient(app, raise_server_exceptions=False) as c:
            original = deps.agent_service
            deps.agent_service = svc
            assert c.get("/api/status").status_code == 200
            deps.agent_service = original

    def test_the_record_carries_no_token_when_the_gate_is_off(self, monkeypatch, tmp_path):
        """Writing a token nobody checks would suggest the API is guarded."""
        from agentica.gateway import deps, runtime
        from agentica.gateway.main import app

        monkeypatch.setenv("GATEWAY_AUTH", "false")
        path = tmp_path / "runtime.json"
        monkeypatch.setattr(runtime, "RUNTIME_FILE", path)
        svc = MagicMock()
        svc._ensure_initialized = AsyncMock()
        with TestClient(app, raise_server_exceptions=False):
            deps.agent_service = svc
            assert json.loads(path.read_text())["token"] == ""


class TestWebSocketGate:
    def test_handshake_without_a_token_is_closed(self, client):
        """`method: "agent"` on this socket runs the agent, so it is the same
        authority as /api — and the HTTP middleware never sees a ws scope."""
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "req", "id": "1", "method": "connect", "params": {}})
                ws.receive_json()
        assert excinfo.value.code == 4401

    def test_token_in_the_connect_frame_is_accepted(self, client):
        """The protocol has always documented params.auth.token; now it means
        something."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "req", "id": "1", "method": "connect",
                "params": {"auth": {"token": TOKEN}, "client": {"id": "c1"}},
            })
            assert ws.receive_json()["payload"]["type"] == "hello-ok"

    def test_token_in_the_handshake_query_is_accepted(self, client):
        with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
            ws.send_json({
                "type": "req", "id": "1", "method": "connect",
                "params": {"client": {"id": "c1"}},
            })
            assert ws.receive_json()["ok"] is True


class TestRuntimeRecord:
    def test_publish_then_read_round_trips(self, monkeypatch, tmp_path):
        from agentica.gateway import runtime

        monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "gateway" / "runtime.json")
        rec = runtime.GatewayRuntime(pid=os.getpid(), host="127.0.0.1", port=54321,
                                     token="t", version="1.2.3")
        path = runtime.publish(rec)
        back = runtime.read()
        assert (back.pid, back.port, back.token) == (os.getpid(), 54321, "t")
        # It holds a credential.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert not list(path.parent.glob("*.tmp"))

    def test_url_never_hands_out_the_bind_address(self):
        from agentica.gateway.runtime import GatewayRuntime

        wildcard = GatewayRuntime(pid=1, host="0.0.0.0", port=8881, token="", version="")
        assert wildcard.url == "http://127.0.0.1:8881"
        v6 = GatewayRuntime(pid=1, host="::", port=8881, token="", version="")
        assert v6.url == "http://127.0.0.1:8881"

    def test_unpublish_only_removes_your_own_record(self, monkeypatch, tmp_path):
        """A desktop-spawned gateway exiting must not unpublish the terminal
        one that has since taken the file."""
        from agentica.gateway import runtime

        path = tmp_path / "runtime.json"
        monkeypatch.setattr(runtime, "RUNTIME_FILE", path)
        runtime.publish(runtime.GatewayRuntime(pid=4242, host="127.0.0.1", port=1,
                                               token="", version=""))
        runtime.unpublish(9999)
        assert path.exists()
        runtime.unpublish(4242)
        assert not path.exists()

    def test_a_corrupt_record_reads_as_no_gateway(self, monkeypatch, tmp_path):
        """Half a file means "start one", which is also right for junk."""
        from agentica.gateway import runtime

        path = tmp_path / "runtime.json"
        monkeypatch.setattr(runtime, "RUNTIME_FILE", path)
        assert runtime.read() is None
        path.write_text('{"pid": 12', encoding="utf-8")
        assert runtime.read() is None
        path.write_text('{"host": "127.0.0.1"}', encoding="utf-8")
        assert runtime.read() is None

    def test_pid_liveness(self):
        from agentica.gateway.runtime import is_pid_alive

        assert is_pid_alive(os.getpid()) is True
        assert is_pid_alive(0) is False
        assert is_pid_alive(-1) is False


class TestPortReporting:
    def test_port_zero_binds_a_real_port(self):
        """--port 0 must be resolved by whoever holds the socket; handing the
        number to uvicorn instead re-opens the race the zero avoids."""
        from agentica.gateway.main import _bind

        sock = _bind("127.0.0.1", 0)
        try:
            port = sock.getsockname()[1]
            assert port > 0
        finally:
            sock.close()

    def test_explicit_port_is_honoured(self):
        from agentica.gateway.main import _bind

        probe = _bind("127.0.0.1", 0)
        wanted = probe.getsockname()[1]
        probe.close()
        sock = _bind("127.0.0.1", wanted)
        try:
            assert sock.getsockname()[1] == wanted
        finally:
            sock.close()

    def test_the_published_record_carries_the_bound_port(self, monkeypatch, tmp_path):
        from agentica.gateway import deps, runtime
        from agentica.gateway.config import settings
        from agentica.gateway.main import app

        monkeypatch.setenv("GATEWAY_AUTH", "true")
        monkeypatch.setenv("AGENTICA_GATEWAY_TOKEN", TOKEN)
        path = tmp_path / "runtime.json"
        monkeypatch.setattr(runtime, "RUNTIME_FILE", path)
        monkeypatch.setattr(settings, "port", 45678)

        svc = MagicMock()
        svc._ensure_initialized = AsyncMock()
        with TestClient(app, raise_server_exceptions=False):
            deps.agent_service = svc
            saved = json.loads(path.read_text())
        assert saved["port"] == 45678
        assert saved["token"] == TOKEN
        assert saved["url"] == "http://127.0.0.1:45678"
        # Removed on the way out, so a stale record cannot outlive the process.
        assert not path.exists()


class TestDieWithParent:
    def test_exits_once_the_parent_is_gone(self, monkeypatch, tmp_path):
        from agentica.gateway import main as gw_main
        from agentica.gateway import runtime

        monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "runtime.json")
        monkeypatch.setattr(gw_main, "PARENT_POLL_SECONDS", 0.01)
        calls = []

        def fake_exit(code):
            calls.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(os, "_exit", fake_exit)
        with pytest.raises(SystemExit):
            asyncio.run(gw_main._exit_with_parent(parent_pid=999_999))
        assert calls == [0]

    def test_stays_up_while_the_parent_lives(self, monkeypatch, tmp_path):
        from agentica.gateway import main as gw_main
        from agentica.gateway import runtime

        monkeypatch.setattr(runtime, "RUNTIME_FILE", tmp_path / "runtime.json")
        monkeypatch.setattr(gw_main, "PARENT_POLL_SECONDS", 0.01)
        monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("exited early"))

        async def run_briefly():
            task = asyncio.create_task(gw_main._exit_with_parent(os.getpid()))
            await asyncio.sleep(0.05)
            task.cancel()

        asyncio.run(run_briefly())

    def test_no_watchdog_without_parent_pid(self, monkeypatch, tmp_path):
        """Nobody passed a parent, so nothing polls — a plain `agentica-gateway`
        in a terminal must not exit because of this feature."""
        from agentica.gateway.config import Settings

        with monkeypatch.context() as m:
            m.delenv("AGENTICA_GATEWAY_PARENT_PID", raising=False)
            assert Settings.from_env().parent_pid == 0

    def test_parent_pid_comes_from_env_or_flag(self, monkeypatch):
        from agentica.gateway.config import Settings
        from agentica.gateway.main import _parse_args

        monkeypatch.setenv("AGENTICA_GATEWAY_PARENT_PID", "4242")
        assert Settings.from_env().parent_pid == 4242
        assert _parse_args(["--parent-pid", "77"]).parent_pid == 77
