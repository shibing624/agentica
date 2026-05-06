# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-agent debate — explicit 3-round pipeline with full output.

Design
======

Debate is fundamentally **two-way adversarial dialogue**, not the one-way
``actor -> critic -> revise`` flow that ``refine()`` models. So this demo
hand-rolls the pipeline (so every utterance is visible to the user) while
still leveraging :class:`agentica.critic.AgentCritic` to wrap each side as
the other's critic and extract their rebuttal as a structured
:class:`CritiqueResult`.

Three rounds:

1. **Opening** — both debaters state their position independently.
2. **Cross-examination** — each side acts as ``AgentCritic`` over the
   opposing position; the rebuttal lands in ``CritiqueResult.issues``.
3. **Refinement** — each side revises their position in light of the
   rebuttal received.

A moderator agent then synthesises both refined positions into a balanced
summary. All content is printed in full (no log truncation).
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio

from agentica import Agent, OpenAIChat
from agentica.critic import AgentCritic


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _utterance(speaker: str, content: str) -> None:
    print(f"\n--- {speaker} ---")
    print(content.strip())


async def main():
    optimist = Agent(
        name="Optimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions=(
            "你是乐观主义者，看到 AI 技术积极的一面。"
            "在阐述立场时用具体案例支持，保持坚定但理性。\n"
            "当作为评审者反驳对方观点时，请逐条列出对方论点的薄弱之处和忽视的好处，"
            "不要回复 APPROVED。"
        ),
    )

    pessimist = Agent(
        name="Pessimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions=(
            "你是谨慎的批评者，关注 AI 技术潜在的风险与代价。"
            "在阐述立场时用具体案例支持，保持建设性。\n"
            "当作为评审者反驳对方观点时，请逐条列出对方论点的盲区与风险，"
            "不要回复 APPROVED。"
        ),
    )

    moderator = Agent(
        name="Moderator",
        model=OpenAIChat(id="gpt-4o"),
        instructions="你是辩论主持人。综合双方立场，给出平衡客观的总结。",
    )

    topic = "人工智能是否会取代大部分人类工作"

    _section(f"辩论主题: {topic}")

    # Round 1: opening positions, run in parallel since they are independent.
    _section("第一轮 · 开场陈述")
    opening_prompt = f"就'{topic}'阐述你的立场（约200字）。"
    opt_open, pes_open = await asyncio.gather(
        optimist.run(opening_prompt),
        pessimist.run(opening_prompt),
    )
    _utterance("Optimist 开场", opt_open.content)
    _utterance("Pessimist 开场", pes_open.content)

    # Round 2: each side critiques the opposing opening. AgentCritic returns
    # a CritiqueResult whose `.issues` field is the structured rebuttal.
    _section("第二轮 · 交叉反驳")
    pes_as_critic = AgentCritic(pessimist, name="pessimist_rebuttal")
    opt_as_critic = AgentCritic(optimist, name="optimist_rebuttal")
    pes_rebut, opt_rebut = await asyncio.gather(
        pes_as_critic(topic, opt_open.content),
        opt_as_critic(topic, pes_open.content),
    )
    _utterance("Pessimist 反驳 Optimist", pes_rebut.issues)
    _utterance("Optimist 反驳 Pessimist", opt_rebut.issues)

    # Round 3: each side refines their position in light of rebuttal received.
    _section("第三轮 · 精炼立场")
    refine_template = (
        "对方对你前述立场的反驳如下：\n{rebuttal}\n\n"
        "请基于这些反驳，重新阐述你的立场（约250字），"
        "正面回应对方关切，但坚持你的核心观点。"
    )
    opt_final, pes_final = await asyncio.gather(
        optimist.run(refine_template.format(rebuttal=pes_rebut.issues)),
        pessimist.run(refine_template.format(rebuttal=opt_rebut.issues)),
    )
    _utterance("Optimist 精炼立场", opt_final.content)
    _utterance("Pessimist 精炼立场", pes_final.content)

    # Moderator synthesises the final two refined positions.
    _section("主持人总结")
    summary = await moderator.run(
        f"""综合以下两位辩手在'{topic}'议题上经过交叉反驳后的精炼立场，"""
        f"""给出平衡客观的总结（约300字）：\n\n"""
        f"""【乐观派立场】\n{opt_final.content}\n\n"""
        f"""【谨慎派立场】\n{pes_final.content}\n"""
    )
    print(summary.content.strip())


if __name__ == "__main__":
    asyncio.run(main())
