# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: What is actually sitting in the model's context window.

One measurement path shared by the status bar and ``/usage`` so the headline
number and the breakdown can never disagree.

Everything here is a LOCAL tiktoken estimate of the context the next main-agent
request will carry. Provider usage remains authoritative for billing, but it is
not a session-context state model: one run can contain retries, tool loops, and
auxiliary LLM calls whose prompt tokens must stay separate from this figure.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.model.message import Message
from agentica.utils.tokens import count_tokens, count_tool_tokens

# Marker written by CompressionManager.auto_compact() and the CLI /compact
# fallback in front of the summary that replaced the compacted turns.
COMPACT_SUMMARY_PREFIX = "[Context compressed]"


@dataclass
class ContextBreakdown:
    """Per-section token estimate of the next request's prompt."""

    sections: List[tuple] = field(default_factory=list)  # (label, tokens)
    window: int = 0

    @property
    def total(self) -> int:
        return sum(tokens for _, tokens in self.sections)

    @property
    def percent_full(self) -> float:
        return (self.total / self.window * 100) if self.window > 0 else 0.0

    def visible_sections(self) -> List[tuple]:
        """Sections worth printing — empty ones are noise, not information."""
        return [(label, tokens) for label, tokens in self.sections if tokens > 0]


def _split_tools_by_origin(agent) -> tuple:
    """Partition the API tool schemas into (local, mcp).

    ``model.tools`` is the wire format and carries no provenance, so the split
    goes through ``model.functions[name].origin``, which does.
    """
    tools = agent.model.tools or []
    functions: Dict[str, Any] = agent.model.functions or {}
    mcp_names = {
        name for name, fn in functions.items()
        if fn.origin is not None and fn.origin.type == "mcp"
    }
    if not mcp_names:
        return tools, []

    local, mcp = [], []
    for tool in tools:
        name = ""
        if isinstance(tool, dict):
            name = (tool.get("function") or {}).get("name", "") or tool.get("name", "")
        (mcp if name in mcp_names else local).append(tool)
    return local, mcp


def _history_for_next_run(agent) -> List[Any]:
    """The conversation messages the next prompt would replay."""
    if not agent.add_history_to_context:
        return []
    return agent.working_memory.get_messages_from_last_n_runs(
        last_n=agent.num_history_turns,
        skip_role=agent.prompt_config.system_message_role,
    )


async def measure_context(agent) -> ContextBreakdown:
    """Estimate what the next request will carry, split by origin.

    The system prompt is assembled into a single string, so its parts cannot be
    recovered from the finished blob. They are instead re-measured from the same
    sources that fed it, and whatever is left over stays in the ``System prompt``
    row — that keeps the rows summing to the real total instead of inventing a
    precision the assembly does not support.
    """
    breakdown = ContextBreakdown(window=agent.model.context_window or 0)
    model_id = agent.model.id

    # Tool schemas are attached by the runner, not at construction time.
    agent.update_model()

    system_message = await agent.get_system_message()
    system_total = count_tokens([system_message] if system_message else [], None, model_id)

    workspace = await agent.get_workspace_context_prompt()
    workspace_tokens = count_tokens([_as_message(workspace)], None, model_id) if workspace else 0
    # The rendered block, not the source list: once the session is frozen the
    # two diverge (a mid-session skill upgrade rewrites the list only).
    skills_block = agent._get_session_guidance_block()
    skills_tokens = _count_prompt_list([skills_block] if skills_block else [], model_id)
    tool_guide_tokens = _count_prompt_list(agent._tool_policy_prompts, model_id)

    attributed = workspace_tokens + skills_tokens + tool_guide_tokens
    base_tokens = max(system_total - attributed, 0)

    local_tools, mcp_tools = _split_tools_by_origin(agent)

    history = _history_for_next_run(agent)
    summary_msgs = [m for m in history if _is_compact_summary(m)]
    plain_msgs = [m for m in history if not _is_compact_summary(m)]

    breakdown.sections = [
        ("System prompt", base_tokens),
        ("Rules & workspace", workspace_tokens),
        ("Skills", skills_tokens),
        ("Tool guide & subagents", tool_guide_tokens),
        ("Tool definitions", count_tool_tokens(local_tools, model_id) if local_tools else 0),
        ("MCP tools", count_tool_tokens(mcp_tools, model_id) if mcp_tools else 0),
        ("Summarized conversation", count_tokens(summary_msgs, None, model_id)),
        ("Conversation", count_tokens(plain_msgs, None, model_id)),
    ]
    return breakdown


def _is_compact_summary(message) -> bool:
    content = message.content
    return isinstance(content, str) and content.startswith(COMPACT_SUMMARY_PREFIX)


def _as_message(text: str) -> Message:
    return Message(role="system", content=text)


def _count_prompt_list(prompts: Optional[List[str]], model_id: str) -> int:
    if not prompts:
        return 0
    return count_tokens([_as_message("\n".join(prompts))], None, model_id)
