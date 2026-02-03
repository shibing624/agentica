# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Session Management

Manages ACP sessions for IDE integration, including:
- Session creation and lifecycle
- Message history
- Context management
- Streaming output support
- AbortController for cancellation
"""

from __future__ import annotations

import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

from agentica.utils.log import logger


class SessionStatus(str, Enum):
    """Session status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class SessionMessage:
    """Message in ACP session (renamed to avoid conflict with protocol ACPMessage)"""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ACPSession:
    """
    ACP Session for managing a conversation with IDE.
    
    Each session has:
    - Unique ID (sessionId for ACP, maps to sessionKey for backend)
    - Message history
    - Status tracking
    - Context data
    - Mode (default, plan, edit, etc.)
    - AbortController for cancellation support
    - Active run tracking with idempotency key
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_key: Optional[str] = None  # Gateway session key mapping
    cwd: str = "."  # Working directory
    status: SessionStatus = field(default=SessionStatus.CREATED)
    messages: List[SessionMessage] = field(default_factory=list)
    mode: str = "default"  # default, plan, edit, chat
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    # For cancellation support
    abort_event: Optional[asyncio.Event] = field(default=None, repr=False)
    active_run_id: Optional[str] = None  # Idempotency key for current run
    
    def add_message(self, role: Literal["user", "assistant", "system", "tool"], 
                    content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to the session"""
        message = SessionMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        
    def get_history(self, limit: int = None) -> List[SessionMessage]:
        """Get message history, optionally limited to last N messages"""
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:].copy()
    
    def update_status(self, status: SessionStatus) -> None:
        """Update session status"""
        self.status = status
        self.updated_at = datetime.now()
        logger.debug(f"Session {self.id} status: {status.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "sessionKey": self.session_key,
            "cwd": self.cwd,
            "status": self.status.value,
            "mode": self.mode,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SessionManager:
    """
    Manages multiple ACP sessions.
    
    Provides:
    - Session CRUD operations
    - Session lookup by ID or run_id
    - Active run management with AbortController
    - Cleanup of old sessions
    """
    
    def __init__(self, max_sessions: int = 100):
        self._sessions: Dict[str, ACPSession] = {}
        self._run_id_to_session: Dict[str, str] = {}  # run_id -> session_id mapping
        self._max_sessions = max_sessions
        
    def create_session(self, mode: str = "default", 
                       initial_context: Dict[str, Any] = None,
                       cwd: str = ".",
                       session_key: str = None) -> ACPSession:
        """
        Create a new session.
        
        Args:
            mode: Session mode (default, plan, edit, chat)
            initial_context: Initial context data
            cwd: Working directory
            session_key: Optional backend session key
            
        Returns:
            New ACPSession instance
        """
        # Cleanup if too many sessions
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_old_sessions()
            
        session = ACPSession(
            mode=mode,
            context=initial_context or {},
            cwd=cwd,
            session_key=session_key or f"acp:{uuid.uuid4().hex[:8]}"
        )
        self._sessions[session.id] = session
        
        logger.info(f"Created ACP session: {session.id} (mode: {mode}, cwd: {cwd})")
        return session
    
    def get_session(self, session_id: str) -> Optional[ACPSession]:
        """Get a session by ID"""
        return self._sessions.get(session_id)
    
    def get_session_by_run_id(self, run_id: str) -> Optional[ACPSession]:
        """Get a session by active run ID"""
        session_id = self._run_id_to_session.get(run_id)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    def set_active_run(self, session_id: str, run_id: str, abort_event: asyncio.Event = None) -> bool:
        """
        Set the active run for a session.
        
        Args:
            session_id: Session ID
            run_id: Unique run/idempotency ID
            abort_event: Optional asyncio.Event for cancellation
            
        Returns:
            True if successful
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Clear previous run if any
        if session.active_run_id:
            self._run_id_to_session.pop(session.active_run_id, None)
        
        session.active_run_id = run_id
        session.abort_event = abort_event or asyncio.Event()
        self._run_id_to_session[run_id] = session_id
        
        return True
    
    def clear_active_run(self, session_id: str) -> None:
        """Clear the active run for a session"""
        session = self._sessions.get(session_id)
        if session:
            if session.active_run_id:
                self._run_id_to_session.pop(session.active_run_id, None)
            session.active_run_id = None
            session.abort_event = None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [session.to_dict() for session in self._sessions.values()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        session = self._sessions.get(session_id)
        if session:
            # Clean up run mapping
            if session.active_run_id:
                self._run_id_to_session.pop(session.active_run_id, None)
            del self._sessions[session_id]
            logger.info(f"Deleted ACP session: {session_id}")
            return True
        return False
    
    def cancel_session(self, session_id: str) -> bool:
        """
        Cancel a running session.
        
        Sets the abort_event to signal cancellation.
        """
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.RUNNING:
            # Signal abort
            if session.abort_event:
                session.abort_event.set()
            session.update_status(SessionStatus.CANCELLED)
            self.clear_active_run(session_id)
            logger.info(f"Cancelled ACP session: {session_id}")
            return True
        return False
    
    def _cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Cleanup old completed sessions.
        
        Args:
            max_age_hours: Maximum age in hours for completed sessions
            
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self._sessions.items():
            # Remove completed/error/cancelled sessions older than max_age
            if session.status in (SessionStatus.COMPLETED, SessionStatus.ERROR, SessionStatus.CANCELLED):
                age = (now - session.updated_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            session = self._sessions[session_id]
            if session.active_run_id:
                self._run_id_to_session.pop(session.active_run_id, None)
            del self._sessions[session_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old ACP sessions")
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, int]:
        """Get session statistics"""
        stats = {
            "total": len(self._sessions),
            "created": 0,
            "running": 0,
            "completed": 0,
            "error": 0,
            "cancelled": 0,
        }
        for session in self._sessions.values():
            stats[session.status.value] += 1
        return stats
