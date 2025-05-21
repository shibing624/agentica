# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica
"""
import argparse
import os
import sys
import time
from rich.console import Console
from rich.text import Text
import importlib
from typing import List, Optional
from agentica import Agent, OpenAIChat, Moonshot, AzureOpenAIChat, Yi, ZhipuAI, DeepSeek, PythonAgent


console = Console()

# Color constants
class TermColor:
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

# Tool mapping dictionary - maps tool names to their import paths
TOOL_MAP = {
    'airflow': 'agentica.tools.airflow_tool.AirflowTool',
    'analyze_image': 'agentica.tools.analyze_image_tool.AnalyzeImageTool',
    'arxiv': 'agentica.tools.arxiv_tool.ArxivTool',
    'baidu_search': 'agentica.tools.baidu_search_tool.BaiduSearchTool',
    'calculator': 'agentica.tools.calculator_tool.CalculatorTool',
    'code': 'agentica.tools.code_tool.CodeTool',
    'cogview': 'agentica.tools.cogview_tool.CogViewTool',
    'cogvideo': 'agentica.tools.cogvideo_tool.CogVideoTool',
    'dalle': 'agentica.tools.dalle_tool.DalleTool',
    'duckduckgo': 'agentica.tools.duckduckgo_tool.DuckDuckGoTool',
    'edit': 'agentica.tools.edit_tool.EditTool',
    'file': 'agentica.tools.file_tool.FileTool',
    'hackernews': 'agentica.tools.hackernews_tool.HackerNewsTool',
    'jina': 'agentica.tools.jina_tool.JinaTool',
    'mcp': 'agentica.tools.mcp_tool.MCPTool',
    'newspaper': 'agentica.tools.newspaper_tool.NewspaperTool',
    'ocr': 'agentica.tools.ocr_tool.OcrTool',
    'run_python_code': 'agentica.tools.run_python_code_tool.RunPythonCodeTool',
    'search_exa': 'agentica.tools.search_exa_tool.SearchExaTool',
    'search_serper': 'agentica.tools.search_serper_tool.SearchSerperTool',
    'shell': 'agentica.tools.shell_tool.ShellTool',
    'string': 'agentica.tools.string_tool.StringTool',
    'text_analysis': 'agentica.tools.text_analysis_tool.TextAnalysisTool',
    'url_crawler': 'agentica.tools.url_crawler_tool.UrlCrawlerTool',
    'volc_tts': 'agentica.tools.volc_tts_tool.VolcTtsTool',
    'weather': 'agentica.tools.weather_tool.WeatherTool',
    'web_search_pro': 'agentica.tools.web_search_pro_tool.WebSearchProTool',
    'wikipedia': 'agentica.tools.wikipedia_tool.WikipediaTool',
    'workspace': 'agentica.tools.workspace_tool.WorkspaceTool',
    'yfinance': 'agentica.tools.yfinance_tool.YFinanceTool',
}


def parse_args():
    parser = argparse.ArgumentParser(description='CLI for agentica')
    parser.add_argument('--query', type=str, help='Question to ask the LLM', default=None)
    parser.add_argument('--model_provider', type=str,
                        choices=['openai', 'azure', 'moonshot', 'zhipuai', 'deepseek', 'yi'],
                        help='LLM model provider', default='openai')
    parser.add_argument('--model_name', type=str,
                        help='LLM model name to use, can be gpt-4o/glm-4-flash/deepseek-chat/yi-lightning/...',
                        default='gpt-4o-mini')
    parser.add_argument('--api_base', type=str, help='API base URL for the LLM')
    parser.add_argument('--api_key', type=str, help='API key for the LLM')
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens for the LLM')
    parser.add_argument('--temperature', type=float, help='Temperature for the LLM')
    parser.add_argument('--verbose', type=int, help='enable verbose mode', default=0)
    parser.add_argument('--tools', nargs='*',
                        choices=list(TOOL_MAP.keys()),
                        help='Tools to enable')
    return parser.parse_args()


def get_model(model_provider, model_name, api_base=None, api_key=None, max_tokens=None, temperature=None):
    params = {"id": model_name}
    if api_base is not None:
        params["api_base"] = api_base
    if api_key is not None:
        params["api_key"] = api_key
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if model_provider == 'openai':
        model = OpenAIChat(**params)
    elif model_provider == 'azure':
        model = AzureOpenAIChat(**params)
    elif model_provider == 'moonshot':
        model = Moonshot(**params)
    elif model_provider == 'zhipuai':
        model = ZhipuAI(**params)
    elif model_provider == 'deepseek':
        model = DeepSeek(**params)
    elif model_provider == 'yi':
        model = Yi(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def configure_tools(tool_names: Optional[List[str]] = None) -> List:
    """
    Configure and instantiate tools based on their names.
    Uses dynamic import to load tools only when requested.

    Args:
        tool_names: List of tool names to enable. Must be keys in TOOL_MAP.

    Returns:
        List of instantiated tool objects.
    """
    if not tool_names:
        return []

    tools = []
    for name in tool_names:
        if name not in TOOL_MAP:
            console.print(f"[yellow]Warning: Tool '{name}' not recognized. Skipping.[/yellow]")
            continue

        try:
            # Dynamically import and instantiate the tool
            module_path, class_name = TOOL_MAP[name].rsplit('.', 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            tools.append(tool_instance)
            console.print(f"[green]Successfully loaded tool: {name}[/green]")
        except ImportError as e:
            console.print(f"[red]Error: Could not import tool '{name}'. Missing dependencies? {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize tool '{name}': {str(e)}[/red]")

    return tools


def print_header(model_provider: str, model_name: str, tools: Optional[List[str]] = None):
    """Print the application header with version and model information"""
    header_color = TermColor.BRIGHT_CYAN
    accent_color = TermColor.BRIGHT_GREEN
    reset = TermColor.RESET

    # Create a boxed header
    box_width = 60
    border_top = f"{header_color}╭{'─' * (box_width - 1)}╮{reset}"
    border_bottom = f"{header_color}╰{'─' * (box_width - 1)}╯{reset}"

    # Print header
    print(border_top)

    # App name and version
    app_name = f"{header_color}│{reset} {accent_color}Agentica CLI{reset} - Interactive AI Assistant"
    padding = box_width - len(app_name) + len(header_color) + len(reset) + len(accent_color) + len(reset)
    print(f"{app_name}{' ' * padding}{header_color}│{reset}")

    # Model info
    model_info = f"{header_color}│{reset} Model: {accent_color}{model_provider}/{model_name}{reset}"
    padding = box_width - len(model_info) + len(header_color) + len(reset) + len(accent_color) + len(reset)
    print(f"{model_info}{' ' * padding}{header_color}│{reset}")

    # Tools info
    if tools:
        tools_str = ", ".join(tools)
        tools_line = f"{header_color}│{reset} Tools: {accent_color}{tools_str}{reset}"
        if len(tools_line) - len(header_color) - len(reset) - len(accent_color) - len(reset) > box_width - 4:
            tools_str = tools_str[:(box_width - 24)] + "..."
            tools_line = f"{header_color}│{reset} Tools: {accent_color}{tools_str}{reset}"
        padding = box_width - len(tools_line) + len(header_color) + len(reset) + len(accent_color) + len(reset)
        print(f"{tools_line}{' ' * padding}{header_color}│{reset}")

    # Working directory
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    cwd_line = f"{header_color}│{reset} Working Directory: {cwd}"
    if len(cwd_line) - len(header_color) - len(reset) > box_width - 4:
        display_cwd = "..." + cwd[-(box_width - 24):]
        cwd_line = f"{header_color}│{reset} Working Directory: {display_cwd}"
    padding = box_width - len(cwd_line) + len(header_color) + len(reset)
    print(f"{cwd_line}{' ' * padding}{header_color}│{reset}")

    # Input instructions
    input_line = f"{header_color}│{reset} Input with {accent_color}>{reset} prompt, press {accent_color}Enter twice{reset} to submit"
    padding = box_width - len(input_line) + len(header_color) + len(reset) + 2 * len(accent_color) + 2 * len(reset)
    print(f"{input_line}{' ' * padding}{header_color}│{reset}")

    print(border_bottom)
    print()


def run_interactive(agent):
    """Run the interactive CLI with multi-line input support"""
    thinking_start_time = 0
    loading = False

    def show_thinking_indicator():
        """Show a thinking indicator with elapsed time while waiting for a response"""
        if loading:
            elapsed = int(time.time() - thinking_start_time)
            sys.stdout.write(f"\rThinking... ({elapsed}s)")
            sys.stdout.flush()

    def clear_thinking_indicator():
        """Clear the thinking indicator from the terminal"""
        sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the line
        sys.stdout.flush()

    def get_user_input():
        """Get multi-line user input. Input ends when user enters a single empty line."""
        try:
            console.print(Text("> ", style="green"), end="")
            lines = []
            while True:
                try:
                    line = input()
                    if not line.strip():  # Empty line terminates input
                        break
                    lines.append(line)
                except EOFError:
                    return None

            # If no valid input, return None
            if not lines or not any(line.strip() for line in lines):
                return None

            user_input = '\n'.join(lines)
            user_input = user_input.strip()

            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', '\\q']:
                return None

            return user_input
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
            return None

    while True:
        try:
            query = get_user_input()
            if query is None:
                break

            # Set loading state and start thinking indicator
            loading = True
            thinking_start_time = time.time()

            # Start a timer to update the thinking indicator
            import threading
            def update_indicator():
                while loading:
                    show_thinking_indicator()
                    time.sleep(1)  # Update every second

            indicator_thread = threading.Thread(target=update_indicator)
            indicator_thread.daemon = True
            indicator_thread.start()

            # Get response from agent
            response = agent.run(query, stream=True)
            first_chunk = True

            # Print response
            for chunk in response:
                if first_chunk:
                    loading = False  # Stop the thinking indicator
                    indicator_thread.join(timeout=0.1)  # Wait for indicator thread to finish
                    clear_thinking_indicator()  # Clear the indicator
                    first_chunk = False
                console.print(chunk.content, end="")
            console.print("\n")

        except KeyboardInterrupt:
            loading = False
            print("\nOperation interrupted.")
            continue
        except Exception as e:
            loading = False
            console.print(f"[red]Error: {str(e)}[/red]")
            continue

    console.print("\nThank you for using Agentica CLI. Goodbye!")


def main():
    args = parse_args()
    model = get_model(args.model_provider, args.model_name, api_base=args.api_base,
                      api_key=args.api_key, max_tokens=args.max_tokens)
    tools = configure_tools(args.tools) if args.tools else None
    debug_mode = args.verbose > 0
    if args.tools and 'python' in args.tools:
        agent = PythonAgent(model=model, add_datetime_to_instructions=True, add_history_to_messages=True,
                            tools=tools, debug_mode=debug_mode)
    else:
        agent = Agent(model=model, add_datetime_to_instructions=True, add_history_to_messages=True,
                      tools=tools, debug_mode=debug_mode)

    if args.query:
        response = agent.run(args.query, stream=True)
        for chunk in response:
            console.print(chunk.content, end="")
    else:
        print_header(args.model_provider, args.model_name, args.tools)
        run_interactive(agent)


if __name__ == "__main__":
    main()
