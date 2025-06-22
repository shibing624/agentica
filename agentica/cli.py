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
from agentica.config import AGENTICA_HOME

console = Console()
history_file = os.path.join(AGENTICA_HOME, "cli_history.txt")


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


def _create_agent(agent_config: dict, active_tool_names: List[str]):
    """Helper to create or recreate an agent with current config and tools."""
    model = get_model(
        model_provider=agent_config["model_provider"],
        model_name=agent_config["model_name"],
        api_base=agent_config.get("api_base"),
        api_key=agent_config.get("api_key"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
    )
    configured_tools = configure_tools(active_tool_names)

    # Determine agent type
    agent_class = Agent
    if 'run_python_code' in active_tool_names:  # or other criteria for PythonAgent
        agent_class = PythonAgent

    new_agent = agent_class(
        model=model,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        tools=configured_tools,
        debug_mode=agent_config["debug_mode"]
    )
    return new_agent


def print_header(model_provider: str, model_name: str, tools: Optional[List[str]] = None,
                 active_tools: Optional[List[str]] = None):
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

    # Initial Tools info (from command line)
    if tools:
        tools_str = ", ".join(tools)
        init_tools_line = f"{header_color}│{reset} Initial Tools: {accent_color}{tools_str}{reset}"
        if len(init_tools_line) - len(header_color) - len(reset) - len(accent_color) - len(reset) > box_width - 4:
            tools_str = tools_str[:(box_width - 30)] + "..."  # Adjusted for "Initial Tools: "
            init_tools_line = f"{header_color}│{reset} Initial Tools: {accent_color}{tools_str}{reset}"
        padding = box_width - len(init_tools_line) + len(header_color) + len(reset) + len(accent_color) + len(reset)
        print(f"{init_tools_line}{' ' * padding}{header_color}│{reset}")

    # Active Tools info
    active_tools_display = ", ".join(active_tools) if active_tools else "None"
    active_tools_line = f"{header_color}│{reset} Active Tools: {accent_color}{active_tools_display}{reset}"
    if len(active_tools_line) - len(header_color) - len(reset) - len(accent_color) - len(reset) > box_width - 4:
        active_tools_display = active_tools_display[:(box_width - 28)] + "..."  # Adjusted for "Active Tools: "
        active_tools_line = f"{header_color}│{reset} Active Tools: {accent_color}{active_tools_display}{reset}"
    padding = box_width - len(active_tools_line) + len(header_color) + len(reset) + len(accent_color) + len(reset)
    print(f"{active_tools_line}{' ' * padding}{header_color}│{reset}")

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

    raw_input_instruction = f"Type /help for available commands. Use /m for multi-line."
    colored_input_instruction = f"{header_color}│{reset} {raw_input_instruction.replace(
        '/m', f"{accent_color}/m{reset}").replace('/help', f"{accent_color}/help{reset}")} "
    visible_length = len(f"│ {raw_input_instruction} │")
    padding_needed = box_width - visible_length
    print(f"{colored_input_instruction}{' ' * padding_needed}{header_color}│{reset}")

    print(border_bottom)
    print()


def get_user_input_revised(in_multiline_mode_flag: List[bool], multiline_buffer: List[str]):
    """
    Handles user input. 
    Returns (input_string, input_type).
    input_type can be: "query", "command", "multiline_start", "multiline_content", 
                         "multiline_end", "exit", "interrupt"
    """
    try:
        if in_multiline_mode_flag[0]:
            console.print(Text("... ", style="blue"), end="")
        else:
            console.print(Text("> ", style="green"), end="")
        sys.stdout.flush()
        line = input()

        if in_multiline_mode_flag[0]:
            stripped_line = line.strip()
            if stripped_line.lower() in ["/em", "/endmultiline", "/end"]:
                in_multiline_mode_flag[0] = False
                return "", "multiline_end"  # Explicit end command
            multiline_buffer.append(line)  # Add current line (could be empty)
            return line, "multiline_content"
        else:  # Not in multi-line mode
            stripped_line = line.strip()
            if stripped_line.lower() in ['exit', 'quit', '\\q', '/exit', '/quit']:
                return None, "exit"

            if stripped_line.startswith("/"):
                if stripped_line.lower() == "/m":
                    in_multiline_mode_flag[0] = True
                    multiline_buffer.clear()  # Clear buffer when entering multi-line mode
                    console.print("[info]Entered multi-line input mode. Type /em to submit.[/info]")
                    return "", "multiline_start"
                return stripped_line, "command"
            return line, "query"

    except KeyboardInterrupt:
        if in_multiline_mode_flag[0]:
            in_multiline_mode_flag[0] = False
            console.print("\nMulti-line input cancelled.", style="yellow")
            return "", "interrupt"
        # If not in multiline, let the main loop's KeyboardInterrupt handle it
        console.print("\nExiting...", style="yellow")  # Or handle as exit
        return None, "exit"
    except EOFError:
        console.print("\nExiting...", style="yellow")
        return None, "exit"


def run_interactive(agent_config: dict, initial_active_tool_names: List[str]):
    """Run the interactive CLI with multi-line input support and command handling."""

    current_agent = _create_agent(agent_config, initial_active_tool_names)
    active_tool_names = list(initial_active_tool_names)  # Make a mutable copy

    in_multiline_mode = [False]  # Use a list for mutable closure behavior
    multiline_buffer = []

    # For thinking indicator
    thinking_start_time = 0
    loading = False
    indicator_thread = None

    def show_thinking_indicator():
        if loading:
            elapsed = int(time.time() - thinking_start_time)
            sys.stdout.write(f"\rThinking... ({elapsed}s)")
            sys.stdout.flush()

    def stop_thinking_indicator():
        nonlocal loading, indicator_thread
        if loading:
            loading = False  # Set loading to false FIRST
            if indicator_thread and indicator_thread.is_alive():
                indicator_thread.join(timeout=0.1)  # Attempt to join the thread
            # Clear the line regardless of thread state, as it might have printed one last time
            sys.stdout.write("\r" + " " * 40 + "\r")  # Use a slightly wider clear space
            sys.stdout.flush()

    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        tools=agent_config.get("initial_tools_arg"),  # Pass original CLI tools arg for header
        active_tools=active_tool_names
    )

    while True:
        try:
            raw_input_line, input_type = get_user_input_revised(in_multiline_mode, multiline_buffer)

            if input_type == "exit":
                break

            if input_type == "multiline_start":
                # multiline_buffer is already cleared in get_user_input_revised
                continue
            elif input_type == "multiline_content":
                # multiline_buffer is appended in get_user_input_revised
                continue
            elif input_type in ["multiline_end", "interrupt"]:
                query_to_send = None
                if input_type == "multiline_end" and multiline_buffer:  # Explicit /em
                    query_to_send = "\n".join(multiline_buffer)

                multiline_buffer.clear()  # Clear buffer after processing
                in_multiline_mode[0] = False  # Ensure mode is exited

                if query_to_send:
                    pass
                else:
                    if input_type == "interrupt":
                        console.print("[info]Multi-line input cancelled.[/info]")
                    continue
                raw_input_line = query_to_send
                input_type = "query"

            # Handle Commands or Query
            if input_type == "command":
                command_parts = raw_input_line.lower().split()
                command = command_parts[0]

                if command == "/help":
                    console.print(Text(
                        "Available commands:\n"
                        "  /help                         Show this help message.\n"
                        "  /m                            Enter multi-line input mode.\n"
                        "  /em                           Exit multi-line input mode and submit.\n"
                        "  /tools list                   List all available tools and their status.\n"
                        "  /tools add <tool1> [t2..]     Add tool(s) to the current session.\n"
                        "  /tools remove <tool1> [t2..]  Remove tool(s) from the current session.\n"
                        "  /tools current                Show currently active tools.\n"
                        "  /tools clear                  Remove all active tools.\n"
                        "  /exit or /quit                Exit the Agentica CLI.",
                        style="yellow"
                    ))
                elif command == "/tools":
                    if len(command_parts) > 1:
                        sub_command = command_parts[1]
                        if sub_command == "list":
                            console.print("Available tools (active marked with [*]):", style="cyan")
                            all_tool_names = sorted(list(TOOL_MAP.keys()))
                            for tool_name in all_tool_names:
                                marker = " [*]" if tool_name in active_tool_names else ""
                                console.print(f"  - {tool_name}{marker}")
                        elif sub_command == "add" and len(command_parts) > 2:
                            tools_to_process = command_parts[2:]
                            added_any = False
                            for tool_to_add in tools_to_process:
                                if tool_to_add in TOOL_MAP:
                                    if tool_to_add not in active_tool_names:
                                        active_tool_names.append(tool_to_add)
                                        console.print(f"Tool '{tool_to_add}' added.", style="green")
                                        added_any = True
                                    else:
                                        console.print(f"Tool '{tool_to_add}' is already active.", style="yellow")
                                else:
                                    console.print(
                                        f"Unknown tool: {tool_to_add}. Available: {', '.join(TOOL_MAP.keys())}",
                                        style="red")
                            if added_any:
                                current_agent = _create_agent(agent_config, active_tool_names)
                                console.print("Agent recreated with updated tools.", style="green")
                                print_header(agent_config["model_provider"], agent_config["model_name"],
                                             agent_config.get("initial_tools_arg"), active_tool_names)
                        elif sub_command == "remove" and len(command_parts) > 2:
                            tools_to_process = command_parts[2:]
                            removed_any = False
                            for tool_to_remove in tools_to_process:
                                if tool_to_remove in active_tool_names:
                                    active_tool_names.remove(tool_to_remove)
                                    console.print(f"Tool '{tool_to_remove}' removed.", style="green")
                                    removed_any = True
                                else:
                                    console.print(f"Tool '{tool_to_remove}' is not active or unknown.", style="yellow")
                            if removed_any:
                                current_agent = _create_agent(agent_config, active_tool_names)
                                console.print("Agent recreated with updated tools.", style="green")
                                print_header(agent_config["model_provider"], agent_config["model_name"],
                                             agent_config.get("initial_tools_arg"), active_tool_names)
                        elif sub_command == "current":
                            if active_tool_names:
                                console.print("Active tools: " + ", ".join(active_tool_names), style="cyan")
                            else:
                                console.print("No tools are currently active.", style="cyan")
                        elif sub_command == "clear":
                            active_tool_names = []
                            current_agent = _create_agent(agent_config, active_tool_names)
                            console.print("All tools cleared. Recreating agent.", style="green")
                            print_header(agent_config["model_provider"], agent_config["model_name"],
                                         agent_config.get("initial_tools_arg"), active_tool_names)
                        else:
                            console.print(f"Unknown /tools subcommand: {sub_command}. Try /help.", style="red")
                    else:
                        console.print("Usage: /tools [list|add|remove|current|clear]. Try /help.", style="yellow")
                elif raw_input_line.lower() not in ["/m", "/em"]:  # Avoid processing these as unknown
                    console.print(f"Unknown command: {raw_input_line}. Try /help.", style="red")
                continue  # After command, loop for new input

            # Process Query
            if input_type == "query" and raw_input_line.strip():  # Ensure there's a query to send
                query = raw_input_line  # This is now either single line or assembled multi-line

                try:
                    import readline
                    readline.add_history(query.split('\n')[0])  # Add first line of query to history
                except Exception as e:
                    pass

                loading = True
                thinking_start_time = time.time()

                import threading
                indicator_thread = threading.Thread(target=show_thinking_indicator)  # Recreate thread object
                indicator_thread.daemon = True

                # Need a wrapper for show_thinking_indicator to loop
                def _update_indicator_loop():
                    while loading:  # Loop depends on loading flag
                        show_thinking_indicator()
                        time.sleep(0.2)  # Update frequently

                indicator_thread = threading.Thread(target=_update_indicator_loop)
                indicator_thread.daemon = True
                indicator_thread.start()

                try:
                    response_stream = current_agent.run(query, stream=True)

                    first_chunk = True
                    for chunk in response_stream:
                        if first_chunk:
                            stop_thinking_indicator()  # Stop and clear before first output
                            first_chunk = False
                        if chunk and chunk.content:  # Ensure chunk and content exist
                            console.print(chunk.content, end="")
                    if not first_chunk:  # if we printed anything
                        console.print()  # Final newline
                    else:  # No content from agent
                        stop_thinking_indicator()  # Ensure indicator is stopped
                        console.print("[info]Agent returned no content.[/info]")

                except Exception as e:
                    stop_thinking_indicator()
                    console.print(f"\n[bold red]Error during agent execution: {str(e)}[/bold red]")
                finally:
                    stop_thinking_indicator()  # Ensure it's always stopped
                    try:
                        import readline
                        readline.write_history_file(history_file)  # Save history after each query attempt
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not save history: {e}[/yellow]")
                console.print()  # Add a blank line after agent response or error

            elif input_type == "query" and not raw_input_line.strip():
                # Empty single line query, just loop.
                continue

        except KeyboardInterrupt:
            stop_thinking_indicator()
            console.print("\nOperation interrupted by user. Type /exit to quit.", style="yellow")
            in_multiline_mode[0] = False  # Ensure exit from multiline mode
            multiline_buffer = []
            continue
        except Exception as e:
            stop_thinking_indicator()
            console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
            continue

    try:
        import readline
        readline.write_history_file(history_file)
        console.print("[info]Session history saved.[/info]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save final history: {e}[/yellow]")
    console.print("\nThank you for using Agentica CLI. Goodbye!", style="bold green")


def main():
    args = parse_args()
    # Store agent configuration parameters
    agent_config = {
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "debug_mode": args.verbose > 0,
        "initial_tools_arg": list(args.tools) if args.tools else []  # For header
    }
    initial_active_tools = list(args.tools) if args.tools else []

    if args.query:
        # Non-interactive mode (existing logic)
        console.print(f"Running query: {args.query}", style="cyan")
        console.print(
            f"Model: {agent_config['model_provider']}/{agent_config['model_name']}, "
            f"Tools: {initial_active_tools if initial_active_tools else 'None'}",
            style="magenta")

        agent_instance = _create_agent(agent_config, initial_active_tools)
        response = agent_instance.run(args.query, stream=True)
        for chunk in response:
            console.print(chunk.content, end="")
        console.print()  # final newline
    else:
        # Interactive mode
        try:
            import readline
            if os.path.dirname(history_file):
                os.makedirs(os.path.dirname(history_file), exist_ok=True)
            readline.read_history_file(history_file)
            readline.parse_and_bind("tab: complete")
        except FileNotFoundError:
            console.print("[info]No history file found. A new one will be created.[/info]")
        except Exception as e:
            console.print(f"[yellow]Could not initialize/load readline history: {e}[/yellow]")

        run_interactive(agent_config, initial_active_tools)


if __name__ == "__main__":
    main()
