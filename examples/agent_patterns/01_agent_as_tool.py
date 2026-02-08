# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent as Tool demo - Demonstrates how to use an Agent as a tool for another Agent

This pattern differs from Team (transfer):
- Team: Sub-agent receives task description + expected output + additional info (full context)
- as_tool: Sub-agent only receives input_text, treated as a simple function call

Use cases:
- Specialist agents that perform specific tasks (translation, summarization, etc.)
- Modular agent composition where sub-agents are black-box tools
- When you don't need full conversation context passed to sub-agents
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


def main():
    """Main function demonstrating Agent as Tool pattern."""
    
    # ============================================================================
    # Define specialist agents
    # ============================================================================
    
    # Chinese translator agent
    chinese_translator = Agent(
        name="Chinese Translator",
        model=OpenAIChat(id='gpt-4o-mini'),
        instructions="你是一个专业的中文翻译。请将输入的文本准确翻译成中文。",
        description="将文本翻译成中文",
    )
    
    # French translator agent
    french_translator = Agent(
        name="French Translator", 
        model=OpenAIChat(id='gpt-4o-mini'),
        instructions="You are a professional French translator. Translate the input text to French accurately.",
        description="Translate text to French",
    )
    
    # Text summarizer agent
    summarizer = Agent(
        name="Text Summarizer",
        model=OpenAIChat(id='gpt-4o-mini'),
        instructions="你是一个专业的文本摘要专家。请用1-2句话简洁地总结输入的文本。",
        description="简洁地总结文本",
    )
    
    # ============================================================================
    # Create orchestrator agent with sub-agents as tools
    # ============================================================================
    
    orchestrator = Agent(
        name="Orchestrator",
        model=OpenAIChat(id='gpt-4o'),
        instructions="""\
你是一个协调器Agent，可以将任务委派给专家Agent。
你可以使用以下工具：
- translate_to_chinese: 将文本翻译成中文
- translate_to_french: 将文本翻译成法语
- summarize_text: 简洁地总结文本

当用户要求翻译或摘要时，请使用相应的工具。
你可以链式调用多个工具（例如，先摘要再翻译）。
请用中文回复用户。
""",
        tools=[
            chinese_translator.as_tool(
                tool_name="translate_to_chinese",
                tool_description="将给定的文本翻译成中文",
            ),
            french_translator.as_tool(
                tool_name="translate_to_french",
                tool_description="将给定的文本翻译成法语",
            ),
            summarizer.as_tool(
                tool_name="summarize_text",
                tool_description="用1-2句话简洁地总结给定的文本",
            ),
        ],
    )
    
    # ============================================================================
    # Test the orchestrator
    # ============================================================================
    
    print("=" * 60)
    print("示例1: 简单翻译成中文")
    print("=" * 60)
    orchestrator.print_response("请将 'Hello, how are you today?' 翻译成中文")
    
    print("\n" + "=" * 60)
    print("示例2: 翻译成法语")
    print("=" * 60)
    orchestrator.print_response("请将 'The weather is beautiful today' 翻译成法语")
    
    print("\n" + "=" * 60)
    print("示例3: 文本摘要")
    print("=" * 60)
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    """
    orchestrator.print_response(f"请总结以下文本：\n{long_text}")
    
    print("\n" + "=" * 60)
    print("示例4: 链式操作 - 先摘要再翻译成中文")
    print("=" * 60)
    orchestrator.print_response(
        f"请先总结这段文本，然后将摘要翻译成中文：\n{long_text}"
    )


if __name__ == "__main__":
    main()
