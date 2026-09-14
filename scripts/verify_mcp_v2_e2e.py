# -*- coding: utf-8 -*-
"""End-to-end verification of agentica's MCP client against mcp 2.2.0.

Self-contained: starts the example server on both HTTP transports, checks the
four things the 1.x -> 2.x migration moved, then shuts them down again.

    python scripts/verify_mcp_v2_e2e.py

What it pins, and why each one is worth a run:

- **StreamableHTTP client construction.** 2.x dropped `headers` / `timeout` /
  `sse_read_timeout` from `streamable_http_client()` and moved them onto an
  `httpx2.AsyncClient`. Passing the old keywords is a TypeError, so this fails
  loudly rather than silently.
- **`ClientSession` timeout units.** The third positional argument is now
  seconds, not a `timedelta`.
- **`Tool.input_schema`.** v1's `inputSchema` is gone from attribute access, and
  reading it returns nothing rather than raising — so this asserts the *contents*
  of the parameter schema, not merely that a dict came back.
- **`CallToolResult.is_error`.** The v1 spelling raises AttributeError, exercised
  here through a tool that genuinely fails.

Not run in CI: it needs free ports and a subprocess it can kill.
"""
import asyncio
import os
import socket
import subprocess
import sys
import time

# scripts/ sits two levels below the repo root; put that root ahead of the
# editable install so this runs the checkout it lives in, not the main one.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import agentica

_AGENTICA_MODULE = str(agentica.__file__)
assert os.path.dirname(_AGENTICA_MODULE).startswith(_ROOT), (
    f"imported agentica from {_AGENTICA_MODULE}, expected it under {_ROOT}"
)

from agentica.mcp.client import MCPClient
from agentica.mcp.server import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp
from agentica.tools.mcp_tool import McpTool

SERVER = os.path.join(_ROOT, "examples", "mcp", "calc_server.py")
HTTP_URL = "http://localhost:8000/mcp"
SSE_URL = "http://localhost:8081/sse"

FAILURES = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(label)


def _probe_status(port, path):
    """Status of a raw GET to `path`, or None while nothing useful is answering.

    A bound socket is not readiness: uvicorn accepts connections before the
    lifespan startup has finished, and a request landing in that window fails
    in a way that reads like "this endpoint does not exist here".

    Readiness is "the route is mounted", and the server says so itself: an
    unmounted path is 404, while a mounted one answers with whatever its own
    protocol wants (`/sse` opens a stream, `GET /mcp` is a 400 because it is
    not a valid StreamableHTTP request). So any non-404 status counts.
    """
    request = (
        f"GET {path} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as s:
            s.sendall(request)
            first = s.recv(64)
    except OSError:
        return None
    if not first.startswith(b"HTTP/"):
        return None
    try:
        return int(first.split(b" ")[1])
    except (IndexError, ValueError):
        return None


def _log_path(transport):
    return f"/tmp/agentica-mcp-verify-{transport}.log"


def start_server(transport, port, path):
    log_path = _log_path(transport)
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, SERVER, "--transport", transport],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{transport} server exited with {proc.returncode}; see {log_path}"
            )
        status = _probe_status(port, path)
        if status is not None and status != 404:
            return proc
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError(f"{transport} server never served {path} on :{port}; see {log_path}")


def stop(proc):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), 9)


async def low_level(server, label):
    """MCPClient path: schema contents, a real call, and the is_error branch."""
    print(f"\n[{label}] MCPClient")
    async with server:
        async with MCPClient(server=server) as client:
            tools = await client.list_tools()
            names = [t.name for t in tools]
            check(f"{label}: lists tools", "add" in names, f"{len(names)} tools")

            add = next(t for t in tools if t.name == "add")
            props = add.input_schema.get("properties", {})
            check(f"{label}: add schema has a & b", set(props) == {"a", "b"}, str(sorted(props)))

            res = await client.call_tool("add", {"a": 2, "b": 3})
            text = client.extract_result_text(res)
            check(f"{label}: add(2,3) == 5", "5" in text, text.strip())

            # divide by zero -> server raises -> is_error=True -> extract_result_text
            # must take the error branch (this is the `result.isError` rename).
            res = await client.call_tool("divide", {"a": 1, "b": 0})
            text = client.extract_result_text(res)
            check(f"{label}: divide by zero reports an error", text.startswith("Error:"), text[:60])


async def via_mcp_tool(url, label):
    """McpTool path: the parameter schema must reach Function.parameters."""
    print(f"\n[{label}] McpTool")
    tool = McpTool(url=url, sse_timeout=5.0, sse_read_timeout=60.0)
    async with tool:
        check(f"{label}: tools registered", "add" in tool.functions, f"{len(tool.functions)} tools")
        props = tool.functions["add"].parameters.get("properties", {})
        check(f"{label}: add exposes parameter 'a'", "a" in props, str(sorted(props)))
        check(f"{label}: parameter 'a' kept its type",
              props.get("a", {}).get("type") == "number", str(props.get("a")))

        out = tool.functions["multiply"].entrypoint(a=6, b=7)
        check(f"{label}: multiply(6,7) == 42", "42" in str(out), str(out).strip())


async def main():
    await low_level(
        MCPServerStreamableHttp(
            name="http",
            params={"url": HTTP_URL, "timeout": 5.0,
                    "sse_read_timeout": 60.0, "terminate_on_close": True},
        ),
        "streamable-http",
    )
    await low_level(MCPServerSse(name="sse", params={"url": SSE_URL}), "sse")
    await low_level(
        MCPServerStdio(
            name="stdio",
            params={"command": sys.executable, "args": [SERVER], "env": {**os.environ}},
        ),
        "stdio",
    )

    await via_mcp_tool(HTTP_URL, "McpTool/streamable-http")
    await via_mcp_tool(SSE_URL, "McpTool/sse")


def run():
    servers = []
    try:
        servers.append(start_server("http", 8000, "/mcp"))
        servers.append(start_server("sse", 8081, "/sse"))
        for transport in ("http", "sse"):
            print(f"--- {transport} server log ({_log_path(transport)}) ---")
            print("".join(open(_log_path(transport)).readlines()[-4:]).rstrip())
        asyncio.run(main())
    finally:
        for proc in servers:
            stop(proc)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run())
