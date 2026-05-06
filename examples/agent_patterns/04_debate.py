# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-agent debate using ``refine()`` actor-critic loop.

This rewrite demonstrates how :mod:`agentica.critic` maps onto a classic
debate scenario:

* Each debater takes a position on the topic.
* The opposing debater is wrapped as a ``Critic`` — its rebuttals surface
  as the ``issues`` field of :class:`CritiqueResult`.
* ``refine()`` iterates: position -> opponent's rebuttal -> revised position,
  for ``max_iter`` rounds.
* A moderator agent synthesises both refined positions into a balanced
  summary.

The CHAT log channel surfaces every rebuttal/approval verdict, so the
dialectic is visible without scrolling through framework chatter.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio

from agentica import Agent, OpenAIChat
from agentica.critic import AgentCritic, refine
from agentica.utils.log import logger, CHAT_LEVEL


def _enable_chat_only_logging() -> None:
    """Show CHAT-level events only, hiding routine framework INFO lines."""
    logger.setLevel(CHAT_LEVEL)
    for h in logger.handlers:
        h.setLevel(CHAT_LEVEL)


async def main():
    _enable_chat_only_logging()

    optimist = Agent(
        name="Optimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions=(
            "你是乐观主义者，看到AI技术积极的一面。在阐述立场时，"
            "用具体案例支持你的观点，保持坚定但理性。\n"
            "当作为评审者评估对方观点时，请始终给出有力反驳——"
            "不要回复 APPROVED，逐条列出对方论点的薄弱之处。"
        ),
    )

    pessimist = Agent(
        name="Pessimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions=(
            "你是谨慎的批评者，关注AI技术潜在的风险和问题。在阐述立场时，"
            "用具体案例支持你的观点，保持建设性。\n"
            "当作为评审者评估对方观点时，请始终给出有力反驳——"
            "不要回复 APPROVED，逐条列出对方论点的盲区与风险。"
        ),
    )

    moderator = Agent(
        name="Moderator",
        model=OpenAIChat(id="gpt-4o"),
        instructions="你是辩论主持人。综合双方立场，给出平衡客观的总结。",
    )

    topic = "人工智能是否会取代大部分人类工作"

    print("=" * 60)
    print(f"辩论主题: {topic}")
    print("=" * 60)
    print(
        "\n[本 demo 使用 refine() 让每位辩手的立场被对方挑战后再修订。\n"
        " CHAT 行展示每轮反驳，最终输出经过对抗精炼的立场。]\n"
    )

    optimist_position = await refine(
        actor=optimist,
        task=f"就'{topic}'阐述你的立场（约200字）。",
        critics=[AgentCritic(pessimist, name="pessimist_rebuttal")],
        max_iter=2,
    )

    pessimist_position = await refine(
        actor=pessimist,
        task=f"就'{topic}'阐述你的立场（约200字）。",
        critics=[AgentCritic(optimist, name="optimist_rebuttal")],
        max_iter=2,
    )

    print("\n【乐观派的最终立场（经过 2 轮反驳后精炼）】")
    print("-" * 40)
    print(optimist_position)

    print("\n【谨慎派的最终立场（经过 2 轮反驳后精炼）】")
    print("-" * 40)
    print(pessimist_position)

    summary = await moderator.run(
        f"""请综合以下两位辩手在'{topic}'议题上的立场，给出平衡的总结（约300字）：

【乐观派立场】
{optimist_position}

【谨慎派立场】
{pessimist_position}
"""
    )

    print("\n【主持人总结】")
    print("-" * 40)
    print(summary.content)


if __name__ == "__main__":
    asyncio.run(main())
