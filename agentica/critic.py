# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 

Critic protocol + refine() composer for actor-critic patterns.

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
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Protocol, Type, Union

from pydantic import BaseModel, ValidationError

from agentica.prompts.critic import CRITIC_PROMPT_TEMPLATE, REVISE_PROMPT_TEMPLATE
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent


_APPROVAL_TOKEN = "APPROVED"


class CritiqueStyle(str, Enum):
    """Reviewing temperament for an LLM-based critic.

    The CarePilot paper (arXiv:2603.24157) found that strict / neutral /
    lenient styles materially change downstream task accuracy and that
    *neutral* is the safest default — strict critics over-reject good
    drafts and burn revision tokens, lenient critics under-catch issues.
    """

    STRICT = "strict"
    NEUTRAL = "neutral"
    LENIENT = "lenient"


_STYLE_GUIDANCE = {
    CritiqueStyle.STRICT: (
        "Apply a strict standard. Demand evidence, precise definitions, and "
        "completeness. Reject any meaningful gap; never approve out of politeness."
    ),
    CritiqueStyle.NEUTRAL: (
        "Apply a balanced standard. Reject only material gaps that would "
        "mislead a reasonable reader; minor wording issues alone do not warrant rejection."
    ),
    CritiqueStyle.LENIENT: (
        "Apply a lenient standard. Approve drafts that broadly address the task "
        "even if incomplete in detail; reject only on serious factual or relevance failures."
    ),
}

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


VerifyFnReturn = Union[bool, "CritiqueResult", Awaitable[Union[bool, "CritiqueResult"]]]
VerifyFn = Callable[[str, str], VerifyFnReturn]


class ExecCritic:
    """Behavioural verifier — runs a user-supplied predicate against the answer.

    Where :class:`SchemaCritic` checks structural validity (does the answer
    parse as the schema?) and :class:`AgentCritic` asks an LLM for a verbal
    judgement, ``ExecCritic`` verifies *actual behaviour*: execute the
    candidate output (e.g. generated code, a candidate skill, a tool plan)
    in a sandbox and observe whether it produces the expected result.

    The verifier supplied by the user owns the sandboxing strategy — running
    in a subprocess, container, restricted exec, or network call — so the
    SDK stays neutral about isolation policy. The Critic Protocol contract
    is what the SDK provides; the user plugs in any verification logic.

    The predicate is given ``(task, answer)`` and may return:

    * ``bool`` — ``True`` approves; ``False`` rejects with a generic message.
    * :class:`CritiqueResult` — full verdict; ``critic_name`` is overridden
      to ``self.name`` so the trail stays consistent.
    * any of the above wrapped in a coroutine (async predicates are awaited).

    Any exception raised by the predicate is caught and surfaced as a
    rejection with the exception type and message, so a flaky sandbox
    cannot crash the refine loop.

    Example::

        def run_in_sandbox(task: str, code: str) -> bool:
            namespace: dict = {}
            try:
                exec(code, namespace)
                return namespace.get("solve")(*task_inputs) == expected
            except Exception:
                return False

        critic = ExecCritic(run_in_sandbox, name="sandbox_replay")
    """

    def __init__(self, verify_fn: VerifyFn, name: str = "exec") -> None:
        self.verify_fn = verify_fn
        self.name = name

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        try:
            result: Any = self.verify_fn(task, answer)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            return CritiqueResult(
                approved=False,
                issues=f"verify_fn raised {type(e).__name__}: {e}",
                critic_name=self.name,
            )

        if isinstance(result, CritiqueResult):
            return CritiqueResult(
                approved=result.approved,
                issues=result.issues,
                critic_name=self.name,
            )
        if isinstance(result, bool):
            return CritiqueResult(
                approved=result,
                issues="" if result else "verify_fn returned False",
                critic_name=self.name,
            )
        return CritiqueResult(
            approved=False,
            issues=f"verify_fn returned non-bool, non-CritiqueResult value: {result!r}",
            critic_name=self.name,
        )


class AgentCritic:
    """Thin wrapper that turns any ``Agent`` into a Critic.

    The wrapped agent should be configured (via system prompt) to either
    reply with the literal token ``APPROVED`` or list actionable issues.
    The wrapper handles the prompting; the user controls the critic agent's
    model, temperature, and rubric.

    Args:
        agent: The Agent to wrap as a critic.
        name: Identifier surfaced in logs and CritiqueResult.
        style: :class:`CritiqueStyle` controlling reviewing temperament
            (strict / neutral / lenient). Defaults to NEUTRAL — see
            :class:`CritiqueStyle` docstring for rationale.
    """

    def __init__(
        self,
        agent: "Agent",
        name: str = "agent_critic",
        style: CritiqueStyle = CritiqueStyle.NEUTRAL,
    ) -> None:
        self.agent = agent
        self.name = name
        self.style = style

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            style_guidance=_STYLE_GUIDANCE[self.style],
            task=task,
            draft=answer,
            approval_token=_APPROVAL_TOKEN,
        )
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


@dataclass
class RefineRound:
    """One round of the actor-critic loop.

    Captures the draft submitted to critics and every verdict produced.
    The first round records the initial draft (no revision yet); subsequent
    rounds record the post-revision draft.
    """

    draft: str
    verdicts: List[CritiqueResult] = field(default_factory=list)


@dataclass
class RefineResult:
    """Structured outcome of :func:`refine`.

    Attributes:
        final_draft: The actor's last produced draft.
        approved: True iff every critic approved on the final round.
        iterations: Number of critique rounds actually executed
            (initial review + revisions). Always ``len(history)``.
        stopped_reason: One of ``"approved"``, ``"max_iter"``, ``"loop_detected"``,
            or ``"no_critics"`` (no critics or max_iter=0; trivial pass-through).
        history: Full trail — one :class:`RefineRound` per critique round.
    """

    final_draft: str
    approved: bool
    iterations: int
    stopped_reason: str
    history: List[RefineRound] = field(default_factory=list)


def _verdicts_signature(verdicts: List[CritiqueResult]) -> tuple:
    """Stable hash key for detecting that critics gave the same verdict twice.

    A round is considered identical to the previous one if every critic
    returned the same approval flag and the same issues text. When that
    happens twice in a row, revision is unlikely to make further progress
    and we stop early.
    """
    return tuple((v.critic_name, v.approved, v.issues.strip()) for v in verdicts)


async def refine(
    actor: "Agent",
    task: str,
    critics: List[Critic],
    max_iter: int = 1,
) -> RefineResult:
    """Actor-critic composer with parallel critics and loop early-stop.

    Each round runs all critics in parallel; if every critic approves, the
    loop stops. Otherwise the actor revises against concatenated feedback.
    Inspired by the CarePilot paper (arXiv:2603.24157), the loop also exits
    early when two consecutive rounds produce identical verdicts — a signal
    that further revisions are unlikely to help and would only burn tokens.

    With ``max_iter=0`` or ``critics=[]`` the function degenerates to a single
    actor call and returns ``stopped_reason="no_critics"``.

    Args:
        actor: The agent producing the draft.
        task: User task / question.
        critics: List of objects matching the ``Critic`` Protocol.
        max_iter: Max revision rounds. ``0`` skips the critic loop.

    Returns:
        :class:`RefineResult` with the final draft, approval flag, iteration
        count, stop reason, and full per-round history.
    """
    actor_name = actor.name or "actor"

    draft_resp = await actor.run(task)
    draft = draft_resp.content or ""
    if not isinstance(draft, str):
        draft = str(draft)
    _log_draft(actor_name, "draft", draft)

    if not critics or max_iter <= 0:
        return RefineResult(
            final_draft=draft,
            approved=True,
            iterations=0,
            stopped_reason="no_critics",
            history=[],
        )

    history: List[RefineRound] = []
    last_signature: tuple | None = None
    final_approved = False
    stopped_reason = "max_iter"

    for _ in range(max_iter):
        verdicts = await asyncio.gather(*[c(task, draft) for c in critics])
        history.append(RefineRound(draft=draft, verdicts=list(verdicts)))
        for v in verdicts:
            status = "APPROVED" if v.approved else "REJECTED"
            issue_preview = (v.issues[:120] + "...") if len(v.issues) > 120 else v.issues
            logger.chat(f"[critic:{v.critic_name}] {status}{(' - ' + issue_preview) if issue_preview else ''}")

        if all(v.approved for v in verdicts):
            final_approved = True
            stopped_reason = "approved"
            break

        signature = _verdicts_signature(verdicts)
        if last_signature is not None and signature == last_signature:
            stopped_reason = "loop_detected"
            logger.chat(
                f"[refine] loop detected — verdicts unchanged for 2 rounds, stopping early"
            )
            break
        last_signature = signature

        all_issues = "\n".join(
            f"- ({v.critic_name}) {v.issues}"
            for v in verdicts
            if not v.approved
        )
        revise_resp = await actor.run(
            REVISE_PROMPT_TEMPLATE.format(
                task=task, draft=draft, feedback=all_issues
            )
        )
        revised = revise_resp.content or ""
        if not isinstance(revised, str):
            revised = str(revised)
        draft = revised
        _log_draft(actor_name, "revised", draft)

    return RefineResult(
        final_draft=draft,
        approved=final_approved,
        iterations=len(history),
        stopped_reason=stopped_reason,
        history=history,
    )
