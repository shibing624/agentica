# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session and Storage management for Agent

This module contains methods for session management, storage operations,
and memory persistence.
"""
from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from agentica.utils.log import logger
from agentica.model.message import Message
from agentica.memory import AgentMemory, SessionSummary, Memory
from agentica.agent_session import AgentSession
from agentica.db.base import SessionRow

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class SessionMixin:
    """Mixin class containing session and storage methods for Agent."""

    def get_agent_data(self: "Agent") -> Dict[str, Any]:
        """Get agent data to store in database"""
        agent_data: Dict[str, Any] = {}
        if self.name is not None:
            agent_data["name"] = self.name
        if self.agent_data is not None:
            agent_data.update(self.agent_data)
        return agent_data

    def get_session_data(self: "Agent") -> Dict[str, Any]:
        """Get session data to store in database"""
        session_data: Dict[str, Any] = {}
        if self.session_name is not None:
            session_data["session_name"] = self.session_name
        if self.session_state is not None:
            session_data["session_state"] = self.session_state
        if self.images is not None:
            session_data["images"] = [img.model_dump() for img in self.images]
        if self.videos is not None:
            session_data["videos"] = [vid.model_dump() for vid in self.videos]
        if self.session_data is not None:
            session_data.update(self.session_data)
        return session_data

    def get_agent_session(self: "Agent") -> AgentSession:
        """Get the AgentSession object containing the agent state"""
        return AgentSession(
            session_id=self.session_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            memory=self.memory.to_dict() if self.memory else None,
            agent_data=self.get_agent_data(),
            user_data=self.user_data,
            session_data=self.get_session_data(),
        )

    @classmethod
    def from_agent_session(cls, session: AgentSession) -> "Agent":
        """Create an Agent instance from an AgentSession"""
        return cls(
            session_id=session.session_id,
            agent_id=session.agent_id,
            user_id=session.user_id,
            memory=AgentMemory.from_dict(session.memory) if session.memory else AgentMemory(),
        )

    def read_from_storage(self: "Agent") -> Optional[AgentSession]:
        """Read and load session from storage"""
        if self.db is None:
            return None

        try:
            # Read session from storage
            session_row: Optional[SessionRow] = self.db.read_session(session_id=self.session_id)
            if session_row is not None:
                # Load memory from session_row
                if session_row.memory is not None:
                    try:
                        self.memory = AgentMemory.from_dict(session_row.memory)
                    except Exception as e:
                        logger.warning(f"Failed to load memory from storage: {e}")

                # Load agent_data from session_row
                if session_row.agent_data is not None:
                    try:
                        if "name" in session_row.agent_data:
                            self.name = session_row.agent_data.get("name")
                    except Exception as e:
                        logger.warning(f"Failed to load agent_data from storage: {e}")

                # Load user_data from session_row
                if session_row.user_data is not None:
                    try:
                        if self.user_data is None:
                            self.user_data = session_row.user_data
                        else:
                            self.user_data.update(session_row.user_data)
                    except Exception as e:
                        logger.warning(f"Failed to load user_data from storage: {e}")

                # Load session_data from session_row
                if session_row.session_data is not None:
                    try:
                        if "session_name" in session_row.session_data:
                            self.session_name = session_row.session_data.get("session_name")
                        if "session_state" in session_row.session_data:
                            session_state_from_storage = session_row.session_data.get("session_state")
                            if session_state_from_storage is not None:
                                # Update session_state with values from storage
                                for key, value in session_state_from_storage.items():
                                    self.session_state[key] = value
                    except Exception as e:
                        logger.warning(f"Failed to load session_data from storage: {e}")

                # Create an AgentSession object
                self._agent_session = AgentSession(
                    session_id=session_row.session_id,
                    agent_id=session_row.agent_id,
                    user_id=session_row.user_id,
                    memory=session_row.memory,
                    agent_data=session_row.agent_data,
                    user_data=session_row.user_data,
                    session_data=session_row.session_data,
                )
                return self._agent_session
        except Exception as e:
            logger.warning(f"Failed to read from storage: {e}")
        return None

    def write_to_storage(self: "Agent") -> Optional[AgentSession]:
        """Write the agent session to storage"""
        if self.db is None:
            return None

        try:
            agent_session: AgentSession = self.get_agent_session()
            self.db.upsert_session(
                session_row=SessionRow(
                    session_id=agent_session.session_id,
                    agent_id=agent_session.agent_id,
                    user_id=agent_session.user_id,
                    memory=agent_session.memory,
                    agent_data=agent_session.agent_data,
                    user_data=agent_session.user_data,
                    session_data=agent_session.session_data,
                )
            )
            return agent_session
        except Exception as e:
            logger.warning(f"Failed to write to storage: {e}")
        return None

    def add_introduction(self: "Agent", introduction: str) -> None:
        """Add an introduction message to memory"""
        if introduction is None:
            return

        # Check if introduction is already in memory
        for message in self.memory.messages:
            if message.role == "assistant" and message.content == introduction:
                return

        # Add introduction to memory
        self.memory.add_message(Message(role="assistant", content=introduction))

    def load_session(self: "Agent", session_id: Optional[str] = None, force: bool = False) -> Optional[str]:
        """Load a session from storage
        
        Args:
            session_id: The session_id to load. If None, will try to load the current session_id.
            force: If True, load even if the session_id is already loaded.
            
        Returns:
            The session_id that was loaded, or None if no session was loaded.
        """
        # If session_id is None, use the current session_id
        _session_id_to_load = session_id or self.session_id

        # Don't load if already loaded
        if not force and _session_id_to_load == self.session_id and self._agent_session is not None:
            return None

        logger.info(f"Loading session: {_session_id_to_load}")

        # Update session_id if provided
        if session_id is not None:
            self.session_id = session_id

        # Clear the existing agent session
        self._agent_session = None

        # Reset memory if force is True
        if force:
            self.memory = AgentMemory()

        # Load from storage
        self.read_from_storage()

        return _session_id_to_load

    def create_session(self: "Agent", session_id: Optional[str] = None) -> str:
        """Create a new session
        
        Args:
            session_id: The session_id to create. If None, a new UUID will be generated.
            
        Returns:
            The session_id that was created.
        """
        from uuid import uuid4

        # Generate new session_id if not provided
        _new_session_id = session_id or str(uuid4())
        logger.info(f"Creating new session: {_new_session_id}")

        # Update session_id
        self.session_id = _new_session_id

        # Clear the existing agent session
        self._agent_session = None

        # Reset memory
        self.memory = AgentMemory()

        return _new_session_id

    def new_session(self: "Agent", session_id: Optional[str] = None) -> str:
        """Alias for create_session"""
        return self.create_session(session_id=session_id)

    def reset(self: "Agent") -> None:
        """Reset the agent state - creates a new session"""
        from uuid import uuid4

        # Create new session
        self.session_id = str(uuid4())
        self._agent_session = None
        self.memory = AgentMemory()
        logger.info(f"Agent reset. New session_id: {self.session_id}")

    def load_user_memories(self: "Agent") -> None:
        """Load user memories from the database"""
        if self.memory is None:
            return

        logger.warning("load_user_memories is deprecated. Use Workspace.read_memory() instead.")
        try:
            self.memory.load_user_memories()
        except Exception as e:
            logger.warning(f"Failed to load user memories: {e}")

    def get_user_memories(self: "Agent") -> Optional[List[Memory]]:
        """Get user memories from memory"""
        if self.memory is None:
            return None

        logger.warning("get_user_memories is deprecated. Use Workspace.read_memory() instead.")
        return self.memory.memories

    def clear_user_memories(self: "Agent") -> None:
        """Clear user memories from memory and database"""
        if self.memory is None:
            return

        logger.warning("clear_user_memories is deprecated. Memories are now managed via Workspace.")
        try:
            self.memory.clear_memories()
        except Exception as e:
            logger.warning(f"Failed to clear user memories: {e}")

    def rename(self: "Agent", name: str) -> None:
        """Rename the Agent and save to storage"""
        self.read_from_storage()
        self.name = name
        self.write_to_storage()

    def rename_session(self: "Agent", session_name: str) -> None:
        """Rename the current session and save to storage"""
        self.read_from_storage()
        self.session_name = session_name
        self.write_to_storage()

    def generate_session_name(self: "Agent") -> str:
        """Generate a name for the session using the first 6 messages from memory"""
        if self.model is None:
            raise Exception("Model not set")

        gen_session_name_prompt = "Conversation\n"
        messages_for_generating_session_name = []
        try:
            message_pars = self.memory.get_message_pairs()
            for message_pair in message_pars[:3]:
                messages_for_generating_session_name.append(message_pair[0])
                messages_for_generating_session_name.append(message_pair[1])
        except Exception as e:
            logger.warning(f"Failed to generate name: {e}")

        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f"{message.role.upper()}: {message.content}\n"

        gen_session_name_prompt += "\n\nConversation Name: "

        system_message = Message(
            role=self.system_message_role,
            content="Please provide a suitable name for this conversation in maximum 5 words. "
                    "Remember, do not exceed 5 words.",
        )
        user_message = Message(role=self.user_message_role, content=gen_session_name_prompt)
        generate_name_messages = [system_message, user_message]
        generated_name = self.model.response(messages=generate_name_messages)
        content = generated_name.content
        if content is None:
            logger.error("Generated name is None. Trying again.")
            return self.generate_session_name()
        if len(content.split()) > 15:
            logger.error("Generated name is too long. Trying again.")
            return self.generate_session_name()
        return content.replace('"', "").strip()

    def auto_rename_session(self: "Agent") -> None:
        """Automatically rename the session and save to storage"""
        self.read_from_storage()
        generated_session_name = self.generate_session_name()
        logger.debug(f"Generated Session Name: {generated_session_name}")
        self.session_name = generated_session_name
        self.write_to_storage()

    def delete_session(self: "Agent", session_id: str) -> None:
        """Delete the session from database"""
        if self.db is None:
            return
        self.db.delete_session(session_id=session_id)
