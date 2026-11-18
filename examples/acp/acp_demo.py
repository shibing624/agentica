#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP (Agent Client Protocol) demo without an IDE or an LLM call.

Usage:
    python examples/acp/acp_demo.py
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.acp.protocol import ACPProtocolHandler
from agentica.acp.types import (
    ACP_PROTOCOL_VERSION,
    ACPErrorCode,
    ACPRequest,
    ACPResponse,
)
from agentica.acp.handlers import ACPHandlers
from agentica.acp.session import SessionManager, SessionStatus


def demo_acp_types():
    """Show the JSON frames that travel on stdio (one object per line)."""
    request = ACPRequest(
        id=1,
        method="initialize",
        params={"protocolVersion": ACP_PROTOCOL_VERSION},
    )
    print(f"[OK] Request frame: {json.dumps(request.to_dict(), ensure_ascii=False)}")

    success = ACPResponse.create_success(id=1, result={"status": "ok"})
    print(f"[OK] Success frame: {json.dumps(success.to_dict(), ensure_ascii=False)}")

    error = ACPResponse.create_error(
        id=2, code=ACPErrorCode.METHOD_NOT_FOUND, message="Unknown method"
    )
    print(f"[OK] Error frame: {json.dumps(error.to_dict(), ensure_ascii=False)}")


def demo_session_management(storage_dir):
    """Show session metadata persistence and restore in another manager."""
    manager = SessionManager(storage_dir=storage_dir)
    session = manager.create_session(mode="default", cwd=".")
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi")
    session.update_status(SessionStatus.COMPLETED)
    manager.save_session(session)

    restored = SessionManager(storage_dir=storage_dir).restore_session(session.id)
    print(f"[OK] Restored session {restored.id} with {len(restored.messages)} messages")
    manager.delete_session(session.id)


async def demo_acp_handlers():
    """Walk through the v1 handshake and the session methods."""
    handlers = ACPHandlers(protocol=ACPProtocolHandler())

    init = handlers.handle_initialize({
        "protocolVersion": ACP_PROTOCOL_VERSION,
        "clientInfo": {"name": "demo-client", "version": "1.0.0"},
    })
    print(
        f"[OK] initialize: {init['agentInfo']['name']} v{init['agentInfo']['version']}, "
        f"protocol v{init['protocolVersion']}, "
        f"loadSession={init['agentCapabilities']['loadSession']}"
    )

    tools = handlers.handle_tools_list({})["tools"]
    print(f"[OK] tools/list: {len(tools)} tools")

    session_id = handlers.handle_session_new({"cwd": ".", "mcpServers": []})["sessionId"]
    print(f"[OK] session/new: {session_id}")

    call = await handlers.handle_tools_call({
        "name": "glob",
        "arguments": {"pattern": "*.md", "path": "."},
    })
    print(f"[OK] tools/call glob: isError={call['isError']}")

    loaded = handlers.handle_session_load({"sessionId": session_id})
    print(f"[OK] session/load: {loaded['sessionId']}")
    handlers.handle_session_delete({"sessionId": session_id})

    ping = handlers.handle_ping({})
    print(f"[OK] ping: {ping['status']} (initialized={ping['initialized']})")


async def main():
    """Run all demos."""
    print("Agentica ACP v1 demo")
    demo_acp_types()
    with tempfile.TemporaryDirectory(prefix="agentica-acp-demo-") as storage_dir:
        demo_session_management(storage_dir)
    await demo_acp_handlers()

    print("\nTo use ACP with an IDE:")
    print("  1. Run: agentica acp")
    print("  2. Configure your IDE to start that command over stdio")
    print("  3. See docs/advanced/acp.md for the NDJSON frames")


if __name__ == "__main__":
    asyncio.run(main())
