# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Research Bot - A comprehensive research assistant application

This application demonstrates a full-featured research bot that can:
1. Search the web for information
2. Analyze and synthesize findings
3. Generate comprehensive reports
4. Save research to files

Usage:
    python main.py
"""
import sys
import os
from textwrap import dedent
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import Agent, OpenAIChat, BaiduSearchTool, DeepSeek, SkillTool
from agentica.tools.file_tool import FileTool


def create_research_bot():
    """Create a research bot agent."""
    return Agent(
        model=DeepSeek(),
        name="ResearchBot",
        description="A comprehensive research assistant",
        instructions=[
            "You are an expert research assistant.",
            "When given a research topic:",
            "1. Search for relevant information from multiple sources",
            "2. Analyze and cross-reference the findings",
            "3. Synthesize the information into a coherent report",
            "4. Provide citations and references",
            "5. Highlight key insights and conclusions",
            "",
            "Always be thorough, accurate, and objective in your research.",
            "Use markdown formatting for better readability. 中文回答。",
        ],
        tools=[BaiduSearchTool(), FileTool(), SkillTool()],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        enable_multi_round=True,
    )


def research_topic(bot: Agent, topic: str, save_to_file: bool = True) -> str:
    """Conduct research on a topic.
    
    Args:
        bot: The research bot agent
        topic: The topic to research
        save_to_file: Whether to save the report to a file
        
    Returns:
        The research report
    """
    print(f"\n{'='*60}")
    print(f"Researching: {topic}")
    print(f"{'='*60}\n")

    prompt = dedent(f"""
    Please conduct comprehensive research on the following topic:
    
    **Topic**: {topic}
    
    Your research should include:
    1. **Overview**: Brief introduction to the topic
    2. **Key Findings**: Main points and discoveries
    3. **Analysis**: Your analysis of the information
    4. **Current Trends**: Recent developments (if applicable)
    5. **Conclusion**: Summary and key takeaways
    6. **References**: Sources used
    
    Please search for at least 3 different sources to ensure comprehensive coverage. 中文回答。
    """)

    response = bot.run(prompt)
    report = response.content

    if save_to_file:
        # Create output directory if it doesn't exist
        output_dir = "outputs/research"
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:50]
        filename = f"{output_dir}/research_{safe_topic}_{timestamp}.md"

        # Save report
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Research Report: {topic}\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(report)

        print(f"\nReport saved to: {filename}")

    return report


def interactive_mode(bot: Agent):
    """Run the research bot in interactive mode."""
    print("\n" + "="*60)
    print("Research Bot - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  /research <topic> - Research a topic")
    print("  /quick <question> - Quick question (no file save)")
    print("  /exit - Exit the program")
    print()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("Goodbye!")
                break

            if user_input.startswith("/research "):
                topic = user_input[10:].strip()
                if topic:
                    report = research_topic(bot, topic, save_to_file=True)
                    print(f"\n{report}")
                else:
                    print("Please provide a topic to research.")

            elif user_input.startswith("/quick "):
                question = user_input[7:].strip()
                if question:
                    response = bot.run(question)
                    print(f"\nBot: {response.content}")
                else:
                    print("Please provide a question.")

            else:
                # Default: treat as a research topic
                report = research_topic(bot, user_input, save_to_file=True)
                print(f"\n{report}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    """Main entry point."""
    print("="*60)
    print("Research Bot Application")
    print("="*60)

    # Create the research bot
    bot = create_research_bot()

    # Example: Research a topic
    topic = "AI Agent框架的发展趋势"
    report = research_topic(bot, topic)
    print(f"\n{report}")

    # Uncomment to run in interactive mode
    # interactive_mode(bot)


if __name__ == "__main__":
    main()
