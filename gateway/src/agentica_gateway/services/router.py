# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Message router for dispatching incoming channel messages to agents.
"""
from dataclasses import dataclass
from typing import List, Optional

from ..channels.base import Message, ChannelType


@dataclass
class RoutingRule:
    """A single routing rule that maps messages to a target agent.

    Attributes:
        agent_id: The ID of the agent to route matching messages to.
        channel: If set, the message must originate from this channel type.
        channel_id: If set, the message must come from this specific
                    conversation/chat within the channel.
        sender_id: If set, the message must come from this specific sender.
        priority: Higher values are evaluated first (descending order).
    """
    agent_id: str
    channel: Optional[ChannelType] = None
    channel_id: Optional[str] = None
    sender_id: Optional[str] = None
    priority: int = 0


class MessageRouter:
    """Routes incoming channel messages to the appropriate agent.

    Rules are evaluated in descending priority order. The first rule whose
    constraints all match wins. If no rule matches, the ``default_agent``
    is used.
    """

    def __init__(self, default_agent: str = "main"):
        """
        Args:
            default_agent: The agent ID to use when no routing rule matches.
        """
        self.default_agent = default_agent
        self.rules: List[RoutingRule] = []

    def add_rule(self, rule: RoutingRule):
        """Add a routing rule and re-sort by descending priority.

        Args:
            rule: The routing rule to add.
        """
        self.rules.append(rule)
        # Keep rules sorted by descending priority (highest first)
        self.rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, agent_id: str, channel: Optional[ChannelType] = None):
        """Remove routing rules matching the given agent and optional channel.

        Args:
            agent_id: Remove rules targeting this agent.
            channel: If provided, only remove rules for this channel type.
                     If ``None``, all rules for the agent are removed.
        """
        self.rules = [
            r for r in self.rules
            if not (r.agent_id == agent_id and (channel is None or r.channel == channel))
        ]

    def route(self, message: Message) -> str:
        """Determine which agent should handle the given message.

        Matching priority (in order of rule priority):
            1. Exact ``sender_id`` match
            2. Channel type + ``channel_id`` match
            3. Channel type match only
            4. Default agent (fallback)

        Args:
            message: The incoming message to route.

        Returns:
            The agent ID that should handle the message.
        """
        for rule in self.rules:
            if self._match(message, rule):
                return rule.agent_id
        return self.default_agent

    def _match(self, message: Message, rule: RoutingRule) -> bool:
        """Check whether a message satisfies all constraints of a routing rule.

        All non-None fields in the rule must match the corresponding message
        fields. A ``None`` field in the rule is treated as a wildcard.

        Args:
            message: The incoming message.
            rule: The routing rule to test.

        Returns:
            True if the message matches all rule constraints, False otherwise.
        """
        # sender_id must match exactly if specified
        if rule.sender_id and message.sender_id != rule.sender_id:
            return False

        # channel type must match if specified
        if rule.channel and message.channel != rule.channel:
            return False

        # channel_id (conversation) must match if specified
        if rule.channel_id and message.channel_id != rule.channel_id:
            return False

        return True

    def get_session_id(self, message: Message, agent_id: str) -> str:
        """Generate a deterministic session ID from the message and agent.

        Format: ``agent:{agent_id}:{channel}:{channel_id}``

        This ensures that messages from the same conversation on the same
        channel always map to the same agent session.

        Args:
            message: The incoming message.
            agent_id: The agent that will handle the message.

        Returns:
            A unique session ID string.
        """
        return f"agent:{agent_id}:{message.channel.value}:{message.channel_id}"

    def list_rules(self) -> List[dict]:
        """Serialize all routing rules to a list of dicts (for API responses).

        Returns:
            A list of dicts, each containing ``agent_id``, ``channel``,
            ``channel_id``, ``sender_id``, and ``priority``.
        """
        return [
            {
                "agent_id": r.agent_id,
                "channel": r.channel.value if r.channel else None,
                "channel_id": r.channel_id,
                "sender_id": r.sender_id,
                "priority": r.priority,
            }
            for r in self.rules
        ]
