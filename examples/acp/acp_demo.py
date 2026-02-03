#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP (Agent Client Protocol) Demo

Demonstrates the basic functionality of the ACP server
without requiring an actual IDE connection.

Usage:
    python examples/acp_demo/acp_demo.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.acp.types import (
    ACPRequest, 
    ACPResponse, 
    ACPTool, 
    ACPToolCall,
    ACPErrorCode,
)
from agentica.acp.handlers import ACPHandlers
from agentica.acp.session import SessionManager, SessionStatus


def demo_acp_types():
    """Demo ACP data types"""
    print("=" * 60)
    print("Demo 1: ACP Types")
    print("=" * 60)
    
    # Create request
    request = ACPRequest(
        id=1,
        method="initialize",
        params={"protocolVersion": "2024-11-05"}
    )
    print(f"\n[OK] Created request: {request.method}")
    print(f"   JSON: {json.dumps(request.to_dict(), indent=2)[:200]}...")
    
    # Create success response
    response = ACPResponse.create_success(
        id=1,
        result={"status": "ok", "tools": ["read_file", "write_file"]}
    )
    print(f"\n[OK] Created success response")
    print(f"   Has result: {response.result is not None}")
    
    # Create error response
    error = ACPResponse.create_error(
        id=2,
        code=ACPErrorCode.METHOD_NOT_FOUND,
        message="Unknown method"
    )
    print(f"\n[OK] Created error response")
    print(f"   Error code: {error.error['code']}")


def demo_session_management():
    """Demo session management"""
    print("\n" + "=" * 60)
    print("Demo 2: Session Management")
    print("=" * 60)
    
    manager = SessionManager()
    
    # Create sessions
    session1 = manager.create_session(mode="default")
    session2 = manager.create_session(mode="chat")
    print(f"\n[OK] Created 2 sessions: {session1.id}, {session2.id}")
    
    # Add messages
    session1.add_message("user", "Hello, can you help me?")
    session1.add_message("assistant", "Of course! What do you need?")
    print(f"[OK] Added messages to session {session1.id}")
    print(f"   Message count: {len(session1.messages)}")
    
    # Update status
    session1.update_status(SessionStatus.RUNNING)
    print(f"[OK] Updated status to: {session1.status.value}")
    
    # List sessions
    sessions = manager.list_sessions()
    print(f"\n[OK] Active sessions: {len(sessions)}")
    for s in sessions:
        print(f"   - {s['id']}: {s['mode']} ({s['status']})")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\n[OK] Session stats: {stats}")
    
    # Delete session
    manager.delete_session(session2.id)
    print(f"[OK] Deleted session {session2.id}")


def demo_acp_handlers():
    """Demo ACP method handlers"""
    print("\n" + "=" * 60)
    print("Demo 3: ACP Handlers")
    print("=" * 60)
    
    handlers = ACPHandlers()
    
    # Initialize
    print("\n1. Initialize")
    result = handlers.handle_initialize({
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "demo-client", "version": "1.0.0"}
    })
    print(f"   Server: {result['serverInfo']['name']} v{result['serverInfo']['version']}")
    print(f"   Capabilities: {list(result['capabilities'].keys())}")
    
    # List tools
    print("\n2. List Tools")
    tools_result = handlers.handle_tools_list({})
    tools = tools_result['tools']
    print(f"   Available tools: {len(tools)}")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description'][:50]}...")
    
    # Create session
    print("\n3. Create Session")
    session_result = handlers.handle_session_new({
        "mode": "default",
        "context": {"project": "demo"}
    })
    session_id = session_result['sessionId']
    print(f"   Session ID: {session_id}")
    print(f"   Status: {session_result['status']}")
    
    # List sessions
    print("\n4. List Sessions")
    list_result = handlers.handle_session_list({})
    print(f"   Sessions: {len(list_result['sessions'])}")
    print(f"   Stats: {list_result['stats']}")
    
    # Call tool
    print("\n5. Call Tool (ls)")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    call_result = handlers.handle_tools_call({
        "name": "ls",
        "arguments": {"directory": project_root}
    })
    print(f"   Success: {not call_result['isError']}")
    print(f"   Content preview: {call_result['content'][:100]}...")


def demo_full_workflow():
    """Demo a complete workflow"""
    print("\n" + "=" * 60)
    print("Demo 4: Complete Workflow")
    print("=" * 60)
    
    handlers = ACPHandlers()
    
    # Step 1: Initialize connection
    print("\n[1/5] Initializing connection...")
    init_result = handlers.handle_initialize({
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "IDE", "version": "1.0"}
    })
    print(f"      [OK] Connected to {init_result['serverInfo']['name']}")
    
    # Step 2: Get available tools
    print("\n[2/5] Getting available tools...")
    tools_result = handlers.handle_tools_list({})
    print(f"      [OK] Found {len(tools_result['tools'])} tools")
    
    # Step 3: Create a session
    print("\n[3/5] Creating session...")
    session_result = handlers.handle_session_new({
        "mode": "chat",
        "context": {"file": "main.py"}
    })
    session_id = session_result['sessionId']
    print(f"      [OK] Session created: {session_id}")
    
    # Step 4: Use a tool (read file)
    print("\n[4/5] Using tool (read_file)...")
    call_result = handlers.handle_tools_call({
        "name": "read_file",
        "arguments": {
            "file_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "README.md"),
            "limit": 10
        }
    })
    if not call_result['isError']:
        lines = call_result['content'].split('\n')[:5]
        print(f"      [OK] File content (first 5 lines):")
        for line in lines:
            print(f"        {line}")
    
    # Step 5: Cleanup
    print("\n[5/5] Cleaning up...")
    handlers.handle_session_delete({"sessionId": session_id})
    print(f"      [OK] Session deleted")
    
    print("\n" + "=" * 60)
    print("[OK] Workflow completed successfully!")
    print("=" * 60)


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("Agentica ACP Demo")
    print("=" * 60)
    
    try:
        demo_acp_types()
        demo_session_management()
        demo_acp_handlers()
        demo_full_workflow()
        
        print("\n" + "=" * 60)
        print("[OK] All demos completed successfully!")
        print("=" * 60)
        print("\nTo use ACP with an IDE:")
        print("  1. Run: agentica acp")
        print("  2. Configure your IDE to connect to it")
        print("  3. See docs/acp.md for details")
        print()
        return 0
        
    except Exception as e:
        print(f"\n[ERR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
