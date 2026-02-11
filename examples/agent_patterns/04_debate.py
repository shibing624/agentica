# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Debate demo - Demonstrates multi-agent debate pattern

This example shows how to create agents that debate with each other,
demonstrating different perspectives on a topic.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


def main():
    # Create debater agents with different perspectives
    optimist = Agent(
        name="Optimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions="""你是一个乐观主义者，总是看到事物积极的一面。
在辩论中，你需要：
1. 强调AI技术带来的好处和机遇
2. 用具体例子支持你的观点
3. 保持礼貌但坚定的态度
4. 回应对方的观点并提出反驳""",
    )
    
    pessimist = Agent(
        name="Pessimist",
        model=OpenAIChat(id="gpt-4o"),
        instructions="""你是一个谨慎的批评者，关注潜在的风险和问题。
在辩论中，你需要：
1. 指出AI技术可能带来的风险和挑战
2. 用具体例子支持你的观点
3. 保持理性和建设性的态度
4. 回应对方的观点并提出质疑""",
    )
    
    moderator = Agent(
        name="Moderator",
        model=OpenAIChat(id="gpt-4o"),
        instructions="""你是辩论的主持人，负责：
1. 引导辩论的进行
2. 总结双方观点
3. 提出新的讨论角度
4. 最后给出平衡的总结""",
    )

    topic = "人工智能是否会取代大部分人类工作"
    
    print("=" * 60)
    print(f"辩论主题: {topic}")
    print("=" * 60)
    
    # Opening statements
    print("\n【开场陈述】")
    print("-" * 40)
    
    print("\n乐观派观点:")
    optimist_opening = optimist.run_sync(f"请就'{topic}'这个话题发表你的开场陈述（约200字）")
    print(optimist_opening)
    
    print("\n谨慎派观点:")
    pessimist_opening = pessimist.run_sync(f"请就'{topic}'这个话题发表你的开场陈述（约200字）")
    print(pessimist_opening)
    
    # Rebuttal round
    print("\n【反驳环节】")
    print("-" * 40)
    
    print("\n乐观派反驳:")
    optimist_rebuttal = optimist.run_sync(
        f"对方的观点是：{pessimist_opening.content}\n请进行反驳（约150字）"
    )
    print(optimist_rebuttal)
    
    print("\n谨慎派反驳:")
    pessimist_rebuttal = pessimist.run_sync(
        f"对方的观点是：{optimist_opening.content}\n请进行反驳（约150字）"
    )
    print(pessimist_rebuttal)
    
    # Moderator summary
    print("\n【主持人总结】")
    print("-" * 40)
    
    summary = moderator.run_sync(
        f"""请总结这场关于'{topic}'的辩论：
        
乐观派开场：{optimist_opening.content}
谨慎派开场：{pessimist_opening.content}
乐观派反驳：{optimist_rebuttal.content}
谨慎派反驳：{pessimist_rebuttal.content}

请给出平衡的总结和你的看法（约300字）"""
    )
    print(summary)


if __name__ == "__main__":
    main()
