# -*- coding: utf-8 -*-
"""Tests for central sensitive text redaction."""


def test_high_confidence_masks_unambiguous_secrets():
    """Default ``high_confidence`` level masks vendor-prefixed keys, JWTs,
    private-key blocks, auth headers, DB connstr passwords, URL query tokens
    and JSON secret fields — but leaves plain ``key: value`` pairs alone so
    edit_file round-trips don't break."""
    from agentica.security.redact import redact_sensitive_text

    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "super-secret-key-material\n"
        "-----END PRIVATE KEY-----"
    )
    text = "\n".join([
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
        "Bearer abcdefghijklmnopqrstuvwxyz123456",
        "https://api.example.com/v1?api_key=super_secret_123&other=ok",
        '{"api_key": "super_secret_in_json_field"}',
        "db=postgresql://alice:db_password_123@example.com/app",
        private_key,
    ])

    redacted = redact_sensitive_text(text)

    # Vendor-prefixed key & AWS key
    assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    # Authorization header & bearer token
    assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz123456" not in redacted
    # URL query token & JSON secret field
    assert "super_secret_123" not in redacted
    assert "super_secret_in_json_field" not in redacted
    # DB connstr password
    assert "db_password_123" not in redacted
    # Private-key block
    assert "super-secret-key-material" not in redacted
    # Non-secret traffic preserved
    assert "other=ok" in redacted
    assert "REDACTED" in redacted


def test_high_confidence_preserves_source_code_identifiers():
    """High-confidence mode must NOT rewrite plain ``api_key=existing_key``
    style source code — that's what broke edit_file round-trips."""
    from agentica.security.redact import redact_sensitive_text

    text = (
        'api_key=existing_key\n'
        'TOKEN: my_var\n'
        'PASSWORD: $(env)\n'
        'self.api_key = config.get("api_key")\n'
    )
    redacted = redact_sensitive_text(text)
    # Default level leaves these untouched so edit_file matches survive.
    assert redacted == text


def test_strict_level_masks_env_and_keyvalue_forms():
    """Opt-in ``strict`` level still catches env-assignment and key:value
    forms for log-sink scenarios where round-trip safety doesn't matter."""
    from agentica.security.redact import redact_sensitive_text

    text = "OPENAI_API_KEY=sk_real_secret_value_xyz\npassword: plain_text_password_123"
    redacted = redact_sensitive_text(text, level="strict")
    assert "sk_real_secret_value_xyz" not in redacted
    assert "plain_text_password_123" not in redacted
    assert redacted.count("REDACTED") >= 2


def test_tools_safety_reexports_central_redactor():
    from agentica.security.redact import redact_sensitive_text as central_redact
    from agentica.tools.safety import redact_sensitive_text as safety_redact

    text = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    assert safety_redact(text) == central_redact(text)


# ---------------------------------------------------------------------------
# Model-loop integration: redaction is OFF by default; opt-in via env vars.
# ---------------------------------------------------------------------------


def test_tool_outputs_pass_through_by_default(monkeypatch):
    """With redaction toggles off (the default), tool result text reaches the
    LLM verbatim — so read_file output that the LLM later feeds back into
    edit_file as old_string will match byte-for-byte."""
    import asyncio

    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import Function, FunctionCall

    monkeypatch.delenv("AGENTICA_REDACT_TOOL_OUTPUTS", raising=False)
    monkeypatch.delenv("AGENTICA_REDACT_STREAMED_TEXT", raising=False)

    payload = "key=value\napi_key=existing_key\nTOKEN: my_var"

    def emit() -> str:
        """Emit source-code-like text."""
        return payload

    function = Function.from_callable(emit)
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    results = []

    async def run_tool() -> None:
        async for _ in model.run_function_calls(
            [FunctionCall(function=function, arguments={})],
            results,
        ):
            pass

    asyncio.run(run_tool())

    # Default: source-code-shaped text is delivered verbatim.
    assert results[0].content == payload


def test_tool_outputs_redacted_when_enabled(monkeypatch):
    """Setting AGENTICA_REDACT_TOOL_OUTPUTS=1 brings back the strict mask."""
    import asyncio

    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import Function, FunctionCall

    monkeypatch.setenv("AGENTICA_REDACT_TOOL_OUTPUTS", "1")

    secret_prefix = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"

    def emit() -> str:
        """Emit a real-shape secret."""
        return f"output={secret_prefix}"

    function = Function.from_callable(emit)
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    results = []

    async def run_tool() -> None:
        async for _ in model.run_function_calls(
            [FunctionCall(function=function, arguments={})],
            results,
        ):
            pass

    asyncio.run(run_tool())

    assert secret_prefix not in results[0].content
    assert "REDACTED" in results[0].content


def test_streamed_show_result_passthrough_by_default(monkeypatch):
    """Generator tool chunks stream verbatim when stream redaction is off."""
    import asyncio

    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import Function, FunctionCall

    monkeypatch.delenv("AGENTICA_REDACT_TOOL_OUTPUTS", raising=False)
    monkeypatch.delenv("AGENTICA_REDACT_STREAMED_TEXT", raising=False)

    def emit_chunks():
        """Emit ordinary source-code chunks."""
        yield "line1: api_key=existing_key\n"
        yield "line2: more_text\n"

    function = Function.from_callable(emit_chunks)
    function.show_result = True
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    yielded = []
    results = []

    async def run_tool() -> None:
        async for chunk in model.run_function_calls(
            [FunctionCall(function=function, arguments={})],
            results,
        ):
            yielded.append(chunk.content)

    asyncio.run(run_tool())

    streamed = "".join(yielded)
    assert "line1: api_key=existing_key\n" in streamed
    assert "line2: more_text\n" in streamed


def test_streamed_show_result_redacts_when_enabled(monkeypatch):
    """With AGENTICA_REDACT_STREAMED_TEXT=1, streamed chunks get scrubbed
    (strict regex catches secrets even when split across chunks)."""
    import asyncio

    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import Function, FunctionCall

    monkeypatch.setenv("AGENTICA_REDACT_STREAMED_TEXT", "1")

    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"

    def emit_secret_chunks():
        """Emit a fake secret in chunks."""
        yield "progress 1\n"
        yield f"token={secret}\n"
        yield "progress 2\n"

    function = Function.from_callable(emit_secret_chunks)
    function.show_result = True
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    yielded = []
    results = []

    async def run_tool() -> None:
        async for chunk in model.run_function_calls(
            [FunctionCall(function=function, arguments={})],
            results,
        ):
            yielded.append(chunk.content)

    asyncio.run(run_tool())

    streamed = "".join(str(c) for c in yielded)
    assert secret not in streamed
    assert "REDACTED" in streamed


def test_unterminated_private_key_block_always_redacted(monkeypatch):
    """The PEM-private-key safety floor runs even with redaction toggled off."""
    import asyncio

    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import Function, FunctionCall

    monkeypatch.delenv("AGENTICA_REDACT_TOOL_OUTPUTS", raising=False)
    monkeypatch.delenv("AGENTICA_REDACT_STREAMED_TEXT", raising=False)

    def emit_unterminated():
        """Emit a fake unterminated PEM block."""
        yield "progress before\n"
        yield "-----BEGIN PRIVATE KEY-----\n"
        yield "super-secret-key-material\n"

    function = Function.from_callable(emit_unterminated)
    function.show_result = True
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    results = []

    async def run_tool() -> None:
        async for _ in model.run_function_calls(
            [FunctionCall(function=function, arguments={})],
            results,
        ):
            pass

    asyncio.run(run_tool())

    # Final tool message must not carry the unterminated key body.
    assert "super-secret-key-material" not in results[0].content
    assert "REDACTED_PRIVATE_KEY" in results[0].content
