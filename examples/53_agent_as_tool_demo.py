# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent as Tool demo - demonstrates how to use an Agent as a tool for another Agent.

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat


def main():
    """Main function demonstrating Agent as Tool pattern."""
    
    # ============================================================================
    # Define specialist agents
    # ============================================================================
    
    # Spanish translator agent
    spanish_translator = Agent(
        name="Spanish Translator",
        model=OpenAIChat(id='gpt-4o-mini'),
        instructions="You are a professional Spanish translator. Translate the input text to Spanish accurately.",
        description="Translate text to Spanish",
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
        instructions="You are a professional text summarizer. Summarize the input text concisely in 1-2 sentences.",
        description="Summarize text concisely",
    )
    
    # ============================================================================
    # Create orchestrator agent with sub-agents as tools
    # ============================================================================
    
    orchestrator = Agent(
        name="Orchestrator",
        model=OpenAIChat(id='gpt-4o'),
        instructions="""\
You are an orchestrator agent that can delegate tasks to specialist agents.
You have access to the following tools:
- translate_to_spanish: Translate text to Spanish
- translate_to_french: Translate text to French  
- summarize_text: Summarize text concisely

When the user asks for translation or summarization, use the appropriate tool.
You can chain multiple tools if needed (e.g., summarize then translate).
""",
        tools=[
            spanish_translator.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the given text to Spanish",
            ),
            french_translator.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the given text to French",
            ),
            summarizer.as_tool(
                tool_name="summarize_text",
                tool_description="Summarize the given text concisely in 1-2 sentences",
            ),
        ],
        debug_mode=True,
    )
    
    # ============================================================================
    # Test the orchestrator
    # ============================================================================
    
    print("=" * 60)
    print("Example 1: Simple translation to Spanish")
    print("=" * 60)
    orchestrator.print_response("Translate 'Hello, how are you today?' to Spanish")
    
    print("\n" + "=" * 60)
    print("Example 2: Translation to French")
    print("=" * 60)
    orchestrator.print_response("Translate 'The weather is beautiful today' to French")
    
    print("\n" + "=" * 60)
    print("Example 3: Summarization")
    print("=" * 60)
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" 
    cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    """
    orchestrator.print_response(f"Please summarize the following text:\n{long_text}")
    
    print("\n" + "=" * 60)
    print("Example 4: Chained operations - Summarize then translate")
    print("=" * 60)
    orchestrator.print_response(
        f"First summarize this text, then translate the summary to Spanish:\n{long_text}"
    )


if __name__ == "__main__":
    main()
