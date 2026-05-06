# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Actor-Critic refine() demo — multi-critic refinement loop
              combining a deterministic Pydantic schema critic and an
              LLM-based reviewer.

The canonical use of :func:`agentica.critic.refine`:

* Actor (cheap model) produces a JSON intent classification.
* :class:`SchemaCritic` performs program-grade Pydantic validation
  at zero LLM cost.
* :class:`AgentCritic` (strong model, ``STRICT`` style) performs
  semantic plausibility review.
* Both critics run in parallel; ``refine()`` iterates until every
  critic approves, ``max_iter`` is exhausted, or the loop detector
  triggers an early stop.

What this teaches
=================

``SchemaCritic`` catches typos / out-of-enum / out-of-range errors
that no LLM critic reliably catches — exactly the "1% fail rate"
that bites in production. The SDK provides protocol-level scaffolding
so you can mix deterministic verifiers with LLM verifiers, audit
every verdict, and revise iteratively — none of which a single
in-model self-critique prompt can guarantee.

Run with OPENAI_API_KEY set::

    python examples/agent_patterns/13_actor_critic_refine.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat
from agentica.critic import SchemaCritic, AgentCritic, CritiqueStyle, refine
from agentica.utils.log import logger, CHAT_LEVEL


class IntentReply(BaseModel):
    """Customer-message intent classification with confidence score."""
    intent: Literal["question", "command", "complaint", "smalltalk"]
    confidence: float = Field(ge=0.0, le=1.0)


def _enable_chat_logging() -> None:
    """Surface only inter-agent CHAT events; hide framework INFO chatter."""
    logger.setLevel(CHAT_LEVEL)
    for h in logger.handlers:
        h.setLevel(CHAT_LEVEL)


async def main():
    _enable_chat_logging()

    actor = Agent(
        name="classifier",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=(
            "Classify the user message as ONE of: question, command, complaint, smalltalk. "
            'Output JSON only: {"intent": "<one of the four>", "confidence": <float in 0..1>}. '
            "No prose, no markdown code fences."
        ),
    )
    reviewer = Agent(
        name="reviewer",
        model=OpenAIChat(id="gpt-4o"),
        instructions=(
            "Verify whether the classification is reasonable for the given user message. "
            "Reply with the single token APPROVED if so, otherwise list specific issues."
        ),
    )

    message = "My phone keeps freezing, please refund me!"
    print("=" * 60)
    print(f"User message: {message}")
    print("=" * 60)

    result = await refine(
        actor,
        task=f'Classify the following user message:\n"{message}"',
        critics=[
            SchemaCritic(IntentReply, name="schema"),
            AgentCritic(reviewer, name="reviewer", style=CritiqueStyle.STRICT),
        ],
        max_iter=3,
    )

    print("\n--- RefineResult ---")
    print(f"final_draft   : {result.final_draft}")
    print(f"approved      : {result.approved}")
    print(f"stopped_reason: {result.stopped_reason}")
    print(f"iterations    : {result.iterations}")
    print("\n--- Round-by-round audit trail ---")
    for i, rnd in enumerate(result.history, 1):
        print(f"  round {i}: draft={rnd.draft!r}")
        for v in rnd.verdicts:
            status = "APPROVED" if v.approved else "REJECTED"
            issue_head = (v.issues.splitlines()[0][:80] + " ...") if v.issues else ""
            print(f"           [{v.critic_name}] {status} {issue_head}")


if __name__ == "__main__":
    asyncio.run(main())
