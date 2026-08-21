"""Unit tests for WeChatChannel (inline WxBotClient is mocked)."""
import asyncio
import json
import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


def test_channel_type_enum_has_wechat():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.WECHAT.value == "wechat"


def test_wxbotclient_extract_text_concatenates():
    from agentica.gateway.channels.wechat import WxBotClient
    msg = {
        "item_list": [
            {"type": 1, "text_item": {"text": "hello"}},
            {"type": 1, "text_item": {"text": "world"}},
            {"type": 2, "image_item": {}},
        ]
    }
    assert WxBotClient.extract_text(msg) == "hello\nworld"


def test_wxbotclient_is_user_msg():
    from agentica.gateway.channels.wechat import WxBotClient
    assert WxBotClient.is_user_msg({"message_type": 1}) is True
    assert WxBotClient.is_user_msg({"message_type": 2}) is False


@pytest.mark.asyncio
async def test_on_native_message_emits_unified_message(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel
    from agentica.gateway.channels.base import ChannelType

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()

    received = []

    async def handler(msg):
        received.append(msg)

    ch.set_handler(handler)

    bot = MagicMock()
    msg_payload = {
        "message_id": "wx-1",
        "message_type": 1,
        "from_user_id": "wx-user-1",
        "context_token": "ctx-token-1",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }
    bot.extract_text.return_value = "hi"

    ch._on_native_message(bot, msg_payload)

    # call_soon_threadsafe schedules on the loop; yield to let it run
    await asyncio.sleep(0.05)

    assert len(received) == 1
    m = received[0]
    assert m.channel == ChannelType.WECHAT
    assert m.channel_id == "wx-user-1"
    assert m.content == "hi"
    assert m.metadata["context_token"] == "ctx-token-1"
    assert ch._user_ctx["wx-user-1"] == "ctx-token-1"


@pytest.mark.asyncio
async def test_on_native_message_dedupes(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = "hi"
    msg_payload = {
        "message_id": "dup",
        "message_type": 1,
        "from_user_id": "u",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }

    ch._on_native_message(bot, msg_payload)
    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_on_native_message_allowlist_blocks(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(
        token_file=str(tmp_path / "tok.json"),
        allowed_users=["someone-else"],
    )
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = "hi"
    msg_payload = {
        "message_id": "m",
        "message_type": 1,
        "from_user_id": "blocked",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }

    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert received == []


@pytest.mark.asyncio
async def test_send_uses_cached_context_and_splits(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._bot = MagicMock()
    ch._user_ctx["u1"] = "ctx-A"

    text = "z" * (WeChatChannel.SPLIT_LIMIT * 2 + 1)
    ok = await ch.send("u1", text)

    assert ok is True
    assert ch._bot.send_text.call_count == 3
    # All three calls should pass the cached context_token
    for call in ch._bot.send_text.call_args_list:
        args, _ = call
        assert args[0] == "u1"
        assert args[2] == "ctx-A"


@pytest.mark.asyncio
async def test_send_when_not_connected_returns_false(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    assert await ch.send("u1", "hi") is False


def test_base_info_carries_bot_agent_and_version():
    from agentica.gateway.channels.wechat import WxBotClient, _BOT_AGENT, _VERSION

    bi = WxBotClient(token="t")._base_info()
    assert bi["channel_version"] == _VERSION
    assert bi["bot_agent"] == _BOT_AGENT
    assert bi["bot_agent"].startswith("Agentica/")


def test_aes_ecb_roundtrip():
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    key = os.urandom(16)
    plain = b"wechat clawbot media payload " * 7
    cipher = bot._aes_ecb_encrypt(plain, key)
    assert len(cipher) % 16 == 0
    assert cipher != plain
    assert bot._aes_ecb_decrypt(cipher, key) == plain


def test_parse_aes_key_accepts_raw16_b64_hex_b64_and_bare_hex():
    """Inbound CDNMedia.aes_key is not one encoding.

    Images typically send base64(raw 16 bytes). Voice/file/video send
    base64(32 hex chars). ImageItem.aeskey is a bare 32-char hex string.
    Treating the hex form as a 32-byte AES-256 key decrypts garbage and
    surfaces as pycryptodome 'Padding is incorrect.'
    """
    import base64

    from agentica.gateway.channels.wechat import WxBotClient

    key = os.urandom(16)
    assert WxBotClient._parse_aes_key(base64.b64encode(key).decode()) == key
    hex_b64 = base64.b64encode(key.hex().encode("ascii")).decode()
    assert WxBotClient._parse_aes_key(hex_b64) == key
    assert WxBotClient._parse_aes_key(key.hex()) == key


def test_hex_encoded_aes_key_decrypts_voice_payload():
    """The encoding that inbound voice/file actually uses must round-trip.

    Naive ``AES.new(b64decode(aes_key), ECB)`` on a hex-wrapped key is
    AES-256 and fails unpad with 'Padding is incorrect.'
    """
    import base64

    pytest.importorskip("Crypto.Cipher", reason="pycryptodome required for AES tests")
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad

    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    key = os.urandom(16)
    plain = b"#!SILK_V3" + b"\x00" * 80
    cipher = bot._aes_ecb_encrypt(plain, key)
    hex_b64 = base64.b64encode(key.hex().encode("ascii")).decode()

    naive = base64.b64decode(hex_b64)
    assert len(naive) == 32
    # pycryptodome reports the same "this key is wrong" outcome two ways
    # depending on the garbage it decrypts to — "Padding is incorrect." when the
    # length byte is out of range, "PKCS#7 padding is incorrect." when the
    # padding bytes disagree. With a random key either can come out, so matching
    # the capitalised spelling alone was a ~1-in-15 flake (measured 2/30).
    with pytest.raises(ValueError, match="(?i)padding is incorrect"):
        unpad(AES.new(naive, AES.MODE_ECB).decrypt(cipher), 16)

    assert bot._aes_ecb_decrypt(cipher, bot._parse_aes_key(hex_b64)) == plain


def test_download_media_uses_parsed_key_and_default_cdn_base(monkeypatch):
    import base64

    from agentica.gateway.channels.wechat import WxBotClient, _CDN_BASE_URL

    bot = WxBotClient(token="t")
    key = os.urandom(16)
    plain = b"\xff\xd8\xff\xe0" + b"jpeg-body" * 8
    cipher = bot._aes_ecb_encrypt(plain, key)
    hex_b64 = base64.b64encode(key.hex().encode("ascii")).decode()
    captured = {}

    class _Resp:
        content = cipher

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=30):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(
        "agentica.gateway.channels.wechat.requests.get", fake_get
    )
    out = bot.download_media({
        "encrypt_query_param": "enc-q",
        "aes_key": hex_b64,
    })
    assert out == plain
    assert captured["url"].startswith(_CDN_BASE_URL)
    assert "encrypted_query_param=enc-q" in captured["url"]


def test_download_media_skips_decrypt_when_encrypt_type_zero(monkeypatch):
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    raw = b"\xff\xd8\xff\xe0plaintext-jpeg"

    class _Resp:
        content = raw

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "agentica.gateway.channels.wechat.requests.get",
        lambda *a, **k: _Resp(),
    )
    out = bot.download_media({
        "encrypt_query_param": "enc-q",
        "full_url": "https://cdn.example/file",
        "encrypt_type": 0,
    })
    assert out == raw


def test_send_typing_uses_status_field(monkeypatch):
    from agentica.gateway.channels.wechat import WxBotClient, _TYPING, _CANCEL

    bot = WxBotClient(token="t")
    captured = {}

    def fake_post(ep, body, timeout=10):
        captured["ep"] = ep
        captured["body"] = body
        return {}

    monkeypatch.setattr(bot, "_post", fake_post)
    bot.send_typing("u1", typing_ticket="TICK", cancel=False)
    assert captured["ep"] == "ilink/bot/sendtyping"
    assert captured["body"]["ilink_user_id"] == "u1"
    assert captured["body"]["typing_ticket"] == "TICK"
    assert captured["body"]["status"] == _TYPING
    assert "typing_status" not in captured["body"]

    bot.send_typing("u1", typing_ticket="TICK", cancel=True)
    assert captured["body"]["status"] == _CANCEL


def test_send_typing_fetches_ticket_from_getconfig(monkeypatch):
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    captured = {}
    eps = []

    def fake_post(ep, body, timeout=10):
        eps.append(ep)
        if ep == "ilink/bot/getconfig":
            return {"typing_ticket": "GOT"}
        captured["body"] = body
        return {}

    monkeypatch.setattr(bot, "_post", fake_post)
    bot.send_typing("u2")
    assert "ilink/bot/getconfig" in eps
    assert captured["body"]["typing_ticket"] == "GOT"


def test_extract_media_pulls_cdn_refs():
    from agentica.gateway.channels.wechat import WxBotClient

    msg = {
        "item_list": [
            {"type": 2, "image_item": {"media": {"encrypt_query_param": "e1", "aes_key": "k1"}}},
            {"type": 4, "file_item": {"media": {"encrypt_query_param": "e2", "aes_key": "k2"}}},
            {"type": 1, "text_item": {"text": "hi"}},
        ]
    }
    media = WxBotClient.extract_media(msg)
    assert len(media) == 2
    assert media[0]["encrypt_query_param"] == "e1"
    assert media[1]["aes_key"] == "k2"


@pytest.mark.asyncio
async def test_on_native_message_attaches_media(tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = ""
    msg_payload = {
        "message_id": "m-media",
        "message_type": 1,
        "from_user_id": "u",
        "item_list": [
            {"type": 2, "image_item": {"media": {"encrypt_query_param": "e", "aes_key": "k"}}},
        ],
    }
    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert len(received) == 1
    ref = received[0].metadata["media"][0]
    assert ref["kind"] == "image"
    assert ref["media"]["encrypt_query_param"] == "e"


def test_extract_media_typed_kinds_and_skips_video_thumb():
    from agentica.gateway.channels.wechat import WxBotClient

    msg = {
        "item_list": [
            {"type": 2, "image_item": {
                "aeskey": "ab" * 16,
                "media": {"encrypt_query_param": "e1"},
            }},
            {"type": 3, "voice_item": {"media": {"encrypt_query_param": "e2"}}},
            {"type": 5, "video_item": {
                "media": {"encrypt_query_param": "e3"},
                "thumb_media": {"encrypt_query_param": "t3"},
            }},
            {"type": 4, "file_item": {"media": {"encrypt_query_param": "e4"}}},
            {"type": 1, "text_item": {"text": "hi"}},
        ]
    }
    typed = WxBotClient.extract_media_typed(msg)
    assert [(kind, m["encrypt_query_param"]) for kind, m in typed] == [
        ("image", "e1"),
        ("voice", "e2"),
        ("video", "e3"),
        ("file", "e4"),
    ]
    # ImageItem.aeskey (bare hex) must ride on the CDNMedia dict — dropping
    # it is the other half of the inbound decrypt failure.
    assert typed[0][1]["aes_key"] == "ab" * 16


@pytest.mark.asyncio
async def test_fetch_media_downloads_typed_payloads(tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel
    from agentica.gateway.channels.base import ChannelType, Message

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._bot = MagicMock()
    payloads = {"e1": b"\xff\xd8\xff\xe0jpeg", "e2": b"#!SILK_V3xx", "e3": b"mp4data"}
    ch._bot.download_media.side_effect = (
        lambda ref, cdn_base_url=None: payloads[ref["encrypt_query_param"]]
    )
    message = Message(
        channel=ChannelType.WECHAT,
        channel_id="u",
        sender_id="u",
        sender_name="u",
        content="",
        message_id="m",
        metadata={"media": [
            {"kind": "image", "media": {"encrypt_query_param": "e1"}},
            {"kind": "voice", "media": {"encrypt_query_param": "e2"}},
            {"kind": "video", "media": {"encrypt_query_param": "e3"}},
        ]},
    )

    out = await ch.fetch_media(message)

    assert [(m.kind, m.data) for m in out] == [
        ("image", b"\xff\xd8\xff\xe0jpeg"),
        ("voice", b"#!SILK_V3xx"),
        ("video", b"mp4data"),
    ]


@pytest.mark.asyncio
async def test_fetch_media_default_channel_returns_empty():
    from agentica.gateway.channels.base import Channel, ChannelType, Message

    class _TextOnly(Channel):
        @property
        def channel_type(self):
            return ChannelType.WEB

        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def send(self, channel_id, content, **kwargs):
            return True

    msg = Message(
        channel=ChannelType.WEB, channel_id="c", sender_id="s", sender_name="s",
        content="hi", message_id="1", metadata={"media": [{"kind": "image", "media": {}}]},
    )
    assert await _TextOnly().fetch_media(msg) == []


def test_get_updates_session_expired_clears_credentials_and_raises(tmp_path):
    """errcode -14: the token is dead, so both it and the cursor must be
    dropped from the persisted state (a plain restart then goes straight to
    QR login), and the caller is told via SessionExpiredError."""
    from agentica.gateway.channels.wechat import WxBotClient, SessionExpiredError

    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"bot_token": "dead", "ilink_bot_id": "b", "updates_buf": "cur"}))
    bot = WxBotClient(token_file=str(tf))
    assert bot.token == "dead"
    bot._post = lambda *a, **k: {"errcode": -14, "errmsg": "session timeout"}

    with pytest.raises(SessionExpiredError):
        bot.get_updates()

    saved = json.loads(tf.read_text())
    assert saved["bot_token"] == ""
    assert saved["updates_buf"] == ""
    assert bot.token == ""


def test_get_updates_ret_minus_14_also_means_session_expired(tmp_path):
    """The server reports the expiry in either `errcode` or `ret`."""
    from agentica.gateway.channels.wechat import WxBotClient, SessionExpiredError

    bot = WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    bot._post = lambda *a, **k: {"ret": -14, "errmsg": "session timeout"}
    with pytest.raises(SessionExpiredError):
        bot.get_updates()


def test_get_updates_other_errcode_raises_so_the_loop_backs_off(tmp_path):
    """A persistent non--14 error must not be warn-and-return either — that
    was the hot loop: the server answers instantly, the loop retries at once."""
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    bot._post = lambda *a, **k: {"errcode": -2, "errmsg": "bad param"}
    with pytest.raises(RuntimeError, match="err -2"):
        bot.get_updates()


def test_get_updates_success_persists_new_cursor(tmp_path):
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    bot._post = lambda *a, **k: {"ret": 0, "get_updates_buf": "nb", "msgs": [{"message_id": 1}]}
    assert bot.get_updates() == [{"message_id": 1}]
    assert bot._buf == "nb"
    assert json.loads((tmp_path / "t.json").read_text())["updates_buf"] == "nb"


def test_run_loop_backs_off_2s_then_30s_on_persistent_errors(tmp_path, monkeypatch):
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    polls = []

    def fake_get_updates(timeout=30):
        polls.append(1)
        if len(polls) > 3:
            raise KeyboardInterrupt
        raise RuntimeError("boom")

    sleeps = []
    monkeypatch.setattr(bot, "get_updates", fake_get_updates)
    monkeypatch.setattr(wechat.time, "sleep", lambda s: sleeps.append(s))

    bot.run_loop(lambda c, m: None)

    assert sleeps == [2, 2, 30]


def test_run_loop_logs_request_poll_errors_as_warning(tmp_path, monkeypatch):
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    polls = []

    def fake_get_updates(timeout=30):
        polls.append(1)
        if len(polls) > 1:
            raise KeyboardInterrupt
        raise wechat.requests.exceptions.ConnectionError("remote closed")

    warnings = []
    errors = []
    sleeps = []
    monkeypatch.setattr(bot, "get_updates", fake_get_updates)
    monkeypatch.setattr(wechat.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(wechat.logger, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(wechat.time, "sleep", lambda s: sleeps.append(s))

    bot.run_loop(lambda c, m: None)

    assert sleeps == [2]
    assert len(warnings) == 1
    assert "WeChat: loop error: remote closed, retry in 2s" in warnings[0]
    assert errors == []


def test_run_loop_relogs_in_via_qr_on_session_expired(tmp_path, monkeypatch):
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token="t", token_file=str(tmp_path / "t.json"))
    polls = []

    def fake_get_updates(timeout=30):
        polls.append(1)
        if len(polls) > 1:
            raise KeyboardInterrupt
        raise wechat.SessionExpiredError("getUpdates err -14 session timeout")

    logins = []
    monkeypatch.setattr(bot, "get_updates", fake_get_updates)
    monkeypatch.setattr(bot, "login_qr", lambda: logins.append(1))

    bot.run_loop(lambda c, m: None)

    assert logins == [1]


def test_run_loop_stops_after_bounded_relogin_attempts(tmp_path, monkeypatch):
    """An unattended gateway must not mint QR codes forever: after the capped
    attempts the loop stops (the error log names the manual fix)."""
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token="t", token_file=str(tmp_path / "t.json"))

    def always_expired(timeout=30):
        raise wechat.SessionExpiredError("getUpdates err -14 session timeout")

    logins = []
    sleeps = []

    def failed_login():
        logins.append(1)
        raise RuntimeError("qr expired unscanned")

    monkeypatch.setattr(bot, "get_updates", always_expired)
    monkeypatch.setattr(bot, "login_qr", failed_login)
    monkeypatch.setattr(wechat.time, "sleep", lambda s: sleeps.append(s))

    bot.run_loop(lambda c, m: None)

    assert len(logins) == wechat._RELOGIN_MAX_ATTEMPTS
    assert sleeps == [wechat._RELOGIN_RETRY_DELAY] * wechat._RELOGIN_MAX_ATTEMPTS


def test_create_login_qr_returns_png(tmp_path, monkeypatch):
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token_file=str(tmp_path / "t.json"))

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"qrcode": "qid-1", "qrcode_img_content": "https://wx.example/qr"}

    class FakeImg:
        def save(self, buf, format=None):
            buf.write(b"\x89PNG")

    class FakeQR:
        def add_data(self, url):
            assert url.startswith("https://")

        def make(self, fit=True):
            return None

        def make_image(self):
            return FakeImg()

    monkeypatch.setattr(wechat, "_ensure_qrcode", lambda: None)
    wechat.qrcode = MagicMock()
    wechat.qrcode.QRCode = lambda **kw: FakeQR()
    monkeypatch.setattr(wechat.requests, "get", lambda *a, **k: FakeResp())

    info = bot.create_login_qr()
    assert info["qrcode"] == "qid-1"
    assert info["png"]
    assert info["expires_in"] == wechat.QR_TTL_SECONDS


def test_poll_login_qr_persists_token_on_confirmed(tmp_path, monkeypatch):
    from agentica.gateway.channels import wechat

    bot = wechat.WxBotClient(token_file=str(tmp_path / "t.json"))

    class FakeResp:
        def json(self):
            return {
                "status": "confirmed",
                "bot_token": "tok-1",
                "ilink_bot_id": "bot-1",
            }

    monkeypatch.setattr(wechat.requests, "get", lambda *a, **k: FakeResp())
    s = bot.poll_login_qr("qid-1")
    assert s["status"] == "confirmed"
    assert bot.token == "tok-1"
    saved = json.loads((tmp_path / "t.json").read_text())
    assert saved["bot_token"] == "tok-1"


@pytest.mark.asyncio
async def test_connect_without_token_does_not_open_qr(tmp_path, monkeypatch):
    from agentica.gateway.channels.wechat import WeChatChannel, WxBotClient

    called = []
    monkeypatch.setattr(WxBotClient, "login_qr", lambda self: called.append(1))
    ch = WeChatChannel(token_file=str(tmp_path / "missing.json"))
    ok = await ch.connect()
    assert ok is False
    assert called == []
    assert ch.is_connected is False


@pytest.mark.asyncio
async def test_start_web_qr_then_poll_confirmed_starts_loop(tmp_path, monkeypatch):
    from agentica.gateway.channels.wechat import WeChatChannel, WxBotClient

    ch = WeChatChannel(token_file=str(tmp_path / "t.json"))
    monkeypatch.setattr(
        WxBotClient,
        "create_login_qr",
        lambda self: {"qrcode": "qid", "url": "u", "png": "aaa", "expires_in": 120},
    )

    started = await ch.start_web_qr(owner="llli")
    assert started["status"] == "pending"
    assert started["png"] == "aaa"
    assert ch.gateway_user_id == "llli"
    assert json.loads((tmp_path / "t.json").read_text())["gateway_user_id"] == "llli"

    monkeypatch.setattr(
        WxBotClient,
        "poll_login_qr",
        lambda self, qr_id: {"status": "confirmed", "bot_token": "t", "ilink_bot_id": "b"},
    )
    monkeypatch.setattr(ch, "_begin_polling", AsyncMock(return_value=True))
    polled = await ch.poll_web_qr("qid")
    assert polled["status"] == "confirmed"
    ch._begin_polling.assert_awaited_once()


def test_bind_owner_persists_across_new_client(tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel, WxBotClient

    tf = str(tmp_path / "t.json")
    ch = WeChatChannel(token_file=tf)
    ch.bind_owner("llli")
    assert ch.gateway_user_id == "llli"
    again = WxBotClient(token_file=tf)
    assert again.gateway_user_id == "llli"
