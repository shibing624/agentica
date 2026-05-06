# -*- coding: utf-8 -*-
"""Critic protocol + refine() composer for actor-critic patterns.

Design philosophy
=================

Critic in agentica is **a protocol, not a capability**. As base models get
stronger (GPT-5+ already self-critique internally), reimplementing a generic
LLM critic in user-space wastes tokens. What stays valuable is the *contract*
that lets developers:

- Inject **business constraints** (red-line rules, schema requirements,
  retrieval-grounding checks) that base models can never know about.
- **Audit** the verdict trail (every critic verdict lands in the CHAT log).
- **Replace / compose** critics — heterogeneous actor (cheap model) + critic
  (strong model) is a cost-quality sweet spot only the SDK can express.

Therefore agentica provides:

- ``Critic`` Protocol — the duck-typed contract any critic must satisfy.
- ``CritiqueResult`` — structured outcome (approved + issues + critic_name).
- ``refine()`` — actor-critic composer with parallel critic execution.
- Two batteries-included adapters:
    - ``SchemaCritic`` — Pydantic schema validation (program-grade verifier;
      always beats any LLM critic, ~zero cost).
    - ``AgentCritic`` — thin wrapper that turns any ``Agent`` into a critic.

agentica deliberately does NOT ship ``ExecCritic`` (run tests/SQL) or
``RetrievalCritic`` (verify against RAG) — these are too business-specific.
Implement your own in <20 lines by matching the ``Critic`` Protocol.

When NOT to use
===============

If your base model is GPT-5+ class and the task has no business-specific
constraints, refine() will burn tokens for marginal gain. Use it when:

1. Output must satisfy a programmatic schema (use ``SchemaCritic``).
2. You want a heterogeneous actor + critic (cheap actor, strong critic).
3. You have business red-lines a generic self-critique cannot enforce.

Example
=======

::

    from pydantic import BaseModel
    from agentica import Agent
    from agentica.critic import SchemaCritic, AgentCritic, refine

    class Reply(BaseModel):
        intent: str
        confidence: float

    actor = Agent(name="writer", model=cheap_model)
    reviewer = Agent(name="reviewer", model=strong_model,
                     instructions="Check correctness; reply APPROVED or list issues.")

    final = await refine(
        actor,
        task="Classify: 'when does the bus leave?'",
        critics=[SchemaCritic(Reply), AgentCritic(reviewer)],
        max_iter=2,
    )
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Type

from pydantic import BaseModel, ValidationError

from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent


_APPROVAL_TOKEN = "APPROVED"

_CRITIC_PROMPT_TEMPLATE = (
    "Critique the following draft produced for the task.\n\n"
    "Task: {task}\n\n"
    "Draft:\n{draft}\n\n"
    f"If the draft fully addresses the task with no material issues, reply "
    f"with the single word {_APPROVAL_TOKEN}. Otherwise, list specific "
    "actionable issues — one per line, no preamble."
)

_REVISE_PROMPT_TEMPLATE = (
    "Revise your previous draft to address the critic feedback.\n\n"
    "Original task: {task}\n\n"
    "Previous draft:\n{draft}\n\n"
    "Critic feedback:\n{feedback}\n\n"
    "Return only the revised draft."
)

_DRAFT_PREVIEW_CHARS = 200


def _log_draft(actor_name: str, label: str, draft: str) -> None:
    """Emit the actor's draft as a CHAT line so the dialogue is auditable.

    Long drafts are truncated and newlines collapsed to a glyph for one-line
    log readability; full drafts remain in the actor's RunResponse.
    """
    preview = draft[:_DRAFT_PREVIEW_CHARS]
    if len(draft) > _DRAFT_PREVIEW_CHARS:
        preview += "..."
    preview = preview.replace("\n", " \u23ce ")
    # stacklevel=3 skips _log_draft itself + the chat helper, so the log
    # record points at refine() — the meaningful call site.
    logger.chat(f"[actor:{actor_name}] {label}: {preview}", stacklevel=3)


@dataclass
class CritiqueResult:
    """Outcome of a single critic check.

    Attributes:
        approved: True if the critic accepts the draft.
        issues: Free-form description of problems (empty when approved).
        critic_name: Identifier of which critic produced this verdict
            (used in trace / logs).
    """
    approved: bool
    issues: str = ""
    critic_name: str = ""


class Critic(Protocol):
    """Duck-typed contract for a critic.

    Any object exposing ``name: str`` and an async ``__call__(task, answer)``
    returning :class:`CritiqueResult` satisfies this Protocol.

    Custom critics can be implemented in <20 lines, e.g.::

        class RegexCritic:
            name = "regex"
            def __init__(self, pattern: str):
                self.pattern = re.compile(pattern)
            async def __call__(self, task, answer):
                if self.pattern.search(answer):
                    return CritiqueResult(approved=True, critic_name=self.name)
                return CritiqueResult(
                    approved=False,
                    issues=f"output must match /{self.pattern.pattern}/",
                    critic_name=self.name,
                )
    """
    name: str

    async def __call__(self, task: str, answer: str) -> CritiqueResult: ...


class SchemaCritic:
    """Pydantic-schema-based critic. Approves iff answer parses as the model.

    Program-grade verifier: zero LLM cost, deterministic, always beats any
    LLM critic on schema conformance.
    """

    def __init__(self, schema: Type[BaseModel], name: str = "schema") -> None:
        self.schema = schema
        self.name = name

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        try:
            self.schema.model_validate_json(answer)
            return CritiqueResult(approved=True, critic_name=self.name)
        except ValidationError as e:
            return CritiqueResult(
                approved=False,
                issues=str(e),
                critic_name=self.name,
            )
        except (ValueError, TypeError) as e:
            return CritiqueResult(
                approved=False,
                issues=f"unparseable: {e}",
                critic_name=self.name,
            )


class AgentCritic:
    """Thin wrapper that turns any ``Agent`` into a Critic.

    The wrapped agent should be configured (via system prompt) to either
    reply with the literal token ``APPROVED`` or list actionable issues.
    The wrapper handles the prompting; the user controls the critic agent's
    model, temperature, and rubric.
    """

    def __init__(self, agent: "Agent", name: str = "agent_critic") -> None:
        self.agent = agent
        self.name = name

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        prompt = _CRITIC_PROMPT_TEMPLATE.format(task=task, draft=answer)
        resp = await self.agent.run(prompt)
        text = resp.content or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if _APPROVAL_TOKEN in text.upper():
            return CritiqueResult(approved=True, critic_name=self.name)
        return CritiqueResult(
            approved=False,
            issues=text,
            critic_name=self.name,
        )


async def refine(
    actor: "Agent",
    task: str,
    critics: List[Critic],
    max_iter: int = 1,
) -> str:
    """Actor-critic composer with parallel critic execution.

    Runs ``actor`` once, then up to ``max_iter`` revision rounds. Each round:

    1. All critics evaluate the current draft **in parallel** (asyncio.gather).
    2. Each verdict is emitted to ``logger.chat()`` for trace.
    3. If every critic approves, return the draft.
    4. Otherwise, concatenate all rejecter feedback and ask the actor to revise.

    With ``max_iter=0`` or ``critics=[]`` the function degenerates to a single
    actor call and returns immediately — no critic-induced cost.

    Args:
        actor: The agent producing the draft.
        task: User task / question.
        critics: List of objects matching the ``Critic`` Protocol.
        max_iter: Max revision rounds. ``0`` skips the critic loop.

    Returns:
        The final draft string (after revisions, or unmodified if no critics).
    """
    actor_name = actor.name or "actor"

    draft_resp = await actor.run(task)
    draft = draft_resp.content or ""
    if not isinstance(draft, str):
        draft = str(draft)
    _log_draft(actor_name, "draft", draft)

    if not critics or max_iter <= 0:
        return draft

    for _ in range(max_iter):
        verdicts = await asyncio.gather(*[c(task, draft) for c in critics])
        for v in verdicts:
            status = "APPROVED" if v.approved else "REJECTED"
            issue_preview = (v.issues[:120] + "...") if len(v.issues) > 120 else v.issues
            logger.chat(f"[critic:{v.critic_name}] {status}{(' - ' + issue_preview) if issue_preview else ''}")

        if all(v.approved for v in verdicts):
            break

        all_issues = "\n".join(
            f"- ({v.critic_name}) {v.issues}"
            for v in verdicts
            if not v.approved
        )
        revise_resp = await actor.run(
            _REVISE_PROMPT_TEMPLATE.format(
                task=task, draft=draft, feedback=all_issues
            )
        )
        revised = revise_resp.content or ""
        if not isinstance(revised, str):
            revised = str(revised)
        draft = revised
        _log_draft(actor_name, "revised", draft)

    return draft
