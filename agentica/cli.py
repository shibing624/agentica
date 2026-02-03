# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica

Data Isolation Architecture:
  - user_id: Workspace memory isolation (each user has independent memories)
  - session_id: Conversation history isolation (each session is independent)

Interactive Features:
  Enter           Submit your message
  Alt+Enter       Insert newline for multi-line (Option+Enter or ESC then Enter)
  Ctrl+J          Insert newline (alternative)
  @filename       Type @ to auto-complete files and inject content
  /command        Type / to see available commands (auto-completes)

Interactive Commands:
  /help           Show available commands and features
  /clear          Clear screen and reset conversation
  /newchat        Start a new chat session (new session_id)
  /user [id]      Show current user or switch to another user
  /users          List all registered users
  /skills         List available skills and triggers
  /memory         Show recent memory
  /save <text>    Save to daily memory
  /workspace      Show workspace status
"""
import argparse
import importlib
import json
import os
import sys
import re
import glob as glob_module
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from rich.console import Console
from rich.text import Text

from agentica import DeepAgent, OpenAIChat, Moonshot, AzureOpenAIChat, Yi, ZhipuAI, DeepSeek
from agentica.config import AGENTICA_HOME
from agentica.utils.log import suppress_console_logging
from agentica.workspace import Workspace
from agentica.skills import load_skills, get_skill_registry, get_available_skills

console = Console()
history_file = os.path.join(AGENTICA_HOME, "cli_history.txt")

# Tool icons for CLI display
TOOL_ICONS = {
    # File tools
    "ls": "📁",
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "✂️",
    "glob": "🔍",
    "grep": "🔎",
    # Execution
    "execute": "⚡",
    # Web tools
    "web_search": "🌐",
    "fetch_url": "🔗",
    # Task management
    "write_todos": "📋",
    "read_todos": "📋",
    # Subagent
    "task": "🤖",
    # Default
    "default": "🔧",
}

# Tool registry - maps tool names to (module_name, class_name)
# Format: 'tool_name': ('module_name', 'ClassName')
# Module path: agentica.tools.{module_name}_tool.{ClassName}
TOOL_REGISTRY = {
    # AI/ML Tools
    'cogvideo':        ('cogvideo',        'CogVideoTool'),
    'cogview':         ('cogview',         'CogViewTool'),
    'dalle':           ('dalle',           'DalleTool'),
    'image_analysis':  ('image_analysis',  'ImageAnalysisTool'),
    'ocr':             ('ocr',             'OcrTool'),
    'video_analysis':  ('video_analysis',  'VideoAnalysisTool'),
    'volc_tts':        ('volc_tts',        'VolcTtsTool'),

    # Search Tools
    'arxiv':           ('arxiv',           'ArxivTool'),
    'baidu_search':    ('baidu_search',    'BaiduSearchTool'),
    'dblp':            ('dblp',            'DblpTool'),
    'duckduckgo':      ('duckduckgo',      'DuckDuckGoTool'),
    'search_bocha':    ('search_bocha',    'SearchBochaTool'),
    'search_exa':      ('search_exa',      'SearchExaTool'),
    'search_serper':   ('search_serper',   'SearchSerperTool'),
    'web_search_pro':  ('web_search_pro',  'WebSearchProTool'),
    'wikipedia':       ('wikipedia',       'WikipediaTool'),

    # Web/Network Tools
    'browser':         ('browser',         'BrowserTool'),
    'jina':            ('jina',            'JinaTool'),
    'newspaper':       ('newspaper',       'NewspaperTool'),
    'url_crawler':     ('url_crawler',     'UrlCrawlerTool'),

    # File/Code Tools
    'calculator':      ('calculator',      'CalculatorTool'),
    'code':            ('code',            'CodeTool'),
    'edit':            ('edit',            'EditTool'),
    'file':            ('file',            'FileTool'),
    'run_nb_code':     ('run_nb_code',     'RunNbCodeTool'),
    'run_python_code': ('run_python_code', 'RunPythonCodeTool'),
    'shell':           ('shell',           'ShellTool'),
    'string':          ('string',          'StringTool'),
    'text_analysis':   ('text_analysis',   'TextAnalysisTool'),
    'workspace':       ('workspace',       'WorkspaceTool'),

    # Data Tools
    'hackernews':      ('hackernews',      'HackerNewsTool'),
    'sql':             ('sql',             'SqlTool'),
    'weather':         ('weather',         'WeatherTool'),
    'yfinance':        ('yfinance',        'YFinanceTool'),

    # Integration Tools
    'airflow':         ('airflow',         'AirflowTool'),
    'apify':           ('apify',           'ApifyTool'),
    'mcp':             ('mcp',             'MCPTool'),
    'memori':          ('memori',          'MemoriTool'),
    'skill':           ('skill',           'SkillTool'),
    'video_download':  ('video_download',  'VideoDownloadTool'),
}


def _get_tool_import_path(tool_name: str) -> str:
    """Get full import path for a tool."""
    module_name, class_name = TOOL_REGISTRY[tool_name]
    return f"agentica.tools.{module_name}_tool.{class_name}"


# Color constants
class TermColor:
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


# Rich console color scheme
COLORS = {
    "user": "bright_cyan",
    "agent": "bright_green",
    "thinking": "yellow",
    "tool": "cyan",
    "error": "red",
}


def parse_args():
    parser = argparse.ArgumentParser(description='CLI for agentica')
    
    # Check if running in ACP mode (special handling)
    if len(sys.argv) > 1 and sys.argv[1] == 'acp':
        return None  # Signal to run in ACP mode
    
    parser.add_argument('--query', type=str, help='Question to ask the LLM', default=None)
    parser.add_argument('--model_provider', type=str,
                        choices=['openai', 'azure', 'moonshot', 'zhipuai', 'deepseek', 'yi'],
                        help='LLM model provider', default='zhipuai')
    parser.add_argument('--model_name', type=str,
                        help='LLM model name to use, can be gpt-5/glm-4.7-flash/deepseek-chat/yi-lightning/...',
                        default='glm-4.7-flash')
    parser.add_argument('--api_base', type=str, help='API base URL for the LLM')
    parser.add_argument('--api_key', type=str, help='API key for the LLM')
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens for the LLM')
    parser.add_argument('--temperature', type=float, help='Temperature for the LLM')
    parser.add_argument('--verbose', type=int, help='enable verbose mode', default=0)
    parser.add_argument('--work_dir', type=str, help='Working directory for file operations', default=None)
    parser.add_argument('--enable_multi_round', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        help='Enable multi-round conversation mode (default: True)', default=None)
    parser.add_argument('--tools', nargs='*',
                        choices=list(TOOL_REGISTRY.keys()),
                        help='Additional tools to enable (on top of DeepAgent built-in tools)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Workspace directory path (default: ~/.agentica/workspace)')
    parser.add_argument('--user', type=str, default=None,
                        help='User ID for workspace memory isolation (default: "default")')
    parser.add_argument('--session', type=str, default=None,
                        help='Session ID for conversation isolation (default: auto-generated)')
    parser.add_argument('--no-workspace', action='store_true',
                        help='Disable workspace context injection')
    parser.add_argument('--no-skills', action='store_true',
                        help='Disable skills loading')
    parser.add_argument('command', nargs='?', choices=['acp'],
                        help='Run in ACP mode for IDE integration (agentica acp)')
    return parser.parse_args()


def configure_tools(tool_names: Optional[List[str]] = None) -> List:
    """Configure and instantiate tools based on their names.

    Args:
        tool_names: List of tool names to enable. Must be keys in TOOL_REGISTRY.

    Returns:
        List of instantiated tool objects.
    """
    if not tool_names:
        return []

    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            console.print(f"[yellow]Warning: Tool '{name}' not recognized. Skipping.[/yellow]")
            continue

        try:
            import_path = _get_tool_import_path(name)
            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            tools.append(tool_instance)
            console.print(f"[green]Loaded additional tool: {name}[/green]")
        except ImportError as e:
            console.print(f"[red]Error: Could not import tool '{name}'. Missing dependencies? {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize tool '{name}': {str(e)}[/red]")

    return tools


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


def _create_agent(agent_config: dict, extra_tools: Optional[List] = None,
                  workspace: Optional[Workspace] = None, skills_registry=None,
                  user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Helper to create or recreate a DeepAgent with current config.

    Args:
        agent_config: Agent configuration parameters
        extra_tools: Additional tool instances
        workspace: Workspace instance for context and memory
        skills_registry: Skills registry for skill triggers
        user_id: User ID for workspace memory isolation
        session_id: Session ID for conversation isolation
    """
    model = get_model(
        model_provider=agent_config["model_provider"],
        model_name=agent_config["model_name"],
        api_base=agent_config.get("api_base"),
        api_key=agent_config.get("api_key"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
    )

    # Build instructions with workspace context and skills
    instructions = []

    # Add workspace context if available
    if workspace and workspace.exists():
        context = workspace.get_context_prompt()
        if context:
            instructions.append(context)
        # Add recent memory
        memory = workspace.get_memory_prompt(days=2)
        if memory:
            instructions.append(f"\n# Recent Memory\n{memory}")

    # Add skills summary if available
    if skills_registry and len(skills_registry) > 0:
        skills_summary = skills_registry.get_skills_summary()
        if skills_summary:
            instructions.append(f"\n# Available Skills\n{skills_summary}")

    # Build kwargs for DeepAgent
    deep_agent_kwargs = {
        "model": model,
        "work_dir": agent_config.get("work_dir"),
        "tools": extra_tools,  # Additional tools on top of built-in tools
        "add_datetime_to_instructions": True,
        "add_history_to_messages": True,
        "debug_mode": agent_config["debug_mode"],
    }

    # Add user_id and session_id for data isolation
    if user_id:
        deep_agent_kwargs["user_id"] = user_id
    if session_id:
        deep_agent_kwargs["session_id"] = session_id

    # Add instructions if we have any
    if instructions:
        deep_agent_kwargs["instructions"] = instructions

    # Only pass enable_multi_round if explicitly set by user
    if agent_config.get("enable_multi_round") is not None:
        deep_agent_kwargs["enable_multi_round"] = agent_config["enable_multi_round"]

    new_agent = DeepAgent(**deep_agent_kwargs)
    return new_agent


def print_header(model_provider: str, model_name: str, work_dir: Optional[str] = None,
                 extra_tools: Optional[List[str]] = None):
    """Print the application header with version and model information"""
    header_color = TermColor.BRIGHT_CYAN
    accent_color = TermColor.BRIGHT_GREEN
    dim_color = TermColor.WHITE
    reset = TermColor.RESET

    box_width = 70
    border_top = f"{header_color}{'=' * box_width}{reset}"
    border_bottom = f"{header_color}{'=' * box_width}{reset}"

    print(border_top)
    print(f"{header_color}  Agentica CLI{reset} - Interactive AI Assistant with DeepAgent")
    print(f"  Model: {accent_color}{model_provider}/{model_name}{reset}")

    # Working directory
    cwd = work_dir or os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    if len(cwd) > 50:
        cwd = "..." + cwd[-47:]
    print(f"  Working Directory: {cwd}")

    # Built-in tools (always shown)
    builtin_tools = ["ls", "read_file", "write_file", "edit_file", "glob", "grep",
                     "execute", "web_search", "fetch_url", "write_todos", "read_todos", "task"]
    print(f"  Built-in Tools: {dim_color}{', '.join(builtin_tools)}{reset}")

    # Extra tools info
    if extra_tools:
        tools_str = ", ".join(extra_tools)
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        print(f"  Extra Tools: {accent_color}{tools_str}{reset}")

    print(border_bottom)
    print()
    print(f"  {accent_color}Enter{reset}       Submit your message")
    print(f"  {accent_color}Alt+Enter{reset}   Insert newline for multi-line input")
    print(f"  {accent_color}Ctrl+J{reset}      Insert newline (alternative)")
    print(f"  {accent_color}@filename{reset}   Type @ to auto-complete files and inject content")
    print(f"  {accent_color}/command{reset}    Type / to see available commands")
    print()


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    """Parse @file mentions and return text with mentioned files.
    
    Args:
        text: User input text potentially containing @file mentions
        
    Returns:
        Tuple of (processed_text, list_of_file_paths)
    """
    pattern = r"@([\w./-]+)"
    mentioned_files = []
    
    for match in re.finditer(pattern, text):
        file_path_str = match.group(1)
        file_path = Path(file_path_str).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if file_path.exists() and file_path.is_file():
            mentioned_files.append(file_path)
    
    # Remove @ mentions from text for cleaner display
    processed_text = re.sub(pattern, r'\1', text)
    return processed_text, mentioned_files


def inject_file_contents(prompt_text: str, mentioned_files: list[Path]) -> str:
    """Inject file contents into the prompt.
    
    Args:
        prompt_text: The user's prompt text
        mentioned_files: List of file paths to inject
        
    Returns:
        Final prompt with file contents injected
    """
    if not mentioned_files:
        return prompt_text
    
    context_parts = [prompt_text, "\n\n## Referenced Files\n"]
    for file_path in mentioned_files:
        try:
            content = file_path.read_text()
            # Limit file content to reasonable size
            if len(content) > 20000:
                content = content[:20000] + "\n... (file truncated)"
            context_parts.append(
                f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
            )
        except Exception as e:
            context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")
    
    return "\n".join(context_parts)


def display_user_message(text: str) -> None:
    """Display user message with file mentions colored."""
    pattern = r"(@[\w./-]+)"
    parts = re.split(pattern, text)
    rich_text = Text()
    
    for part in parts:
        if part.startswith("@"):
            rich_text.append(part, style="magenta")
        else:
            rich_text.append(part, style=COLORS["user"])
    
    console.print(rich_text)


def get_file_completions(document_text: str) -> List[str]:
    """Get file completions for @ mentions.
    
    Args:
        document_text: Current input text
        
    Returns:
        List of file path completions
    """
    # Find the @ mention being typed
    match = re.search(r"@([\w./-]*)$", document_text)
    if not match:
        return []
    
    partial = match.group(1)
    
    if partial:
        # Search for files matching the partial path
        search_pattern = f"{partial}*"
        matches = glob_module.glob(search_pattern, recursive=False)
        # Also search in subdirectories
        matches.extend(glob_module.glob(f"**/{partial}*", recursive=True))
    else:
        # Show files in current directory
        matches = glob_module.glob("*")
    
    # Filter to only files (not directories) and limit results
    completions = []
    for m in matches[:20]:
        if os.path.isfile(m):
            completions.append(m)
        elif os.path.isdir(m):
            completions.append(m + "/")
    
    return completions


def show_help():
    """Display help information."""
    help_text = """
Available Commands:
  /help           Show this help message
  /clear          Clear screen and reset conversation
  /newchat        Start a new chat session (new session_id)
  /user [id]      Show current user or switch to another user
  /users          List all registered users in workspace
  /tools          List available additional tools
  /skills         List available skills and triggers
  /memory         Show recent memory from workspace
  /save <text>    Save text to daily memory
  /workspace      Show workspace status and files
  /exit, /quit    Exit the CLI

Data Isolation:
  --user <id>     Set user ID for workspace memory isolation
  --session <id>  Set session ID for conversation history isolation
  /user <id>      Switch to another user (reloads their memories)
  /newchat        Create new session (clears conversation history)

Skill Triggers:
  /commit <msg>   Trigger commit skill (if loaded)
  /github <cmd>   Trigger github skill (if loaded)
  /<skill> <args> Trigger any loaded skill by its trigger

Input Features:
  Enter           Submit your message
  Alt+Enter       Insert newline for multi-line input (Option+Enter on Mac)
  Ctrl+J          Insert newline (alternative)
  @filename       Reference a file - content will be injected into prompt

Tips:
  - Type @ followed by a filename to reference files
  - Use Alt+Enter or Ctrl+J to write multi-line messages
  - DeepAgent has built-in tools: ls, read_file, write_file, edit_file, glob, grep,
    execute, web_search, fetch_url, write_todos, read_todos, task
  - Use --tools to add extra tools, e.g.: --tools calculator shell wikipedia
  - Use --workspace <path> to specify workspace directory
  - Use --user <id> for multi-user memory isolation
"""
    console.print(help_text, style="yellow")


def _extract_filename(file_path: str) -> str:
    """Extract filename from a file path."""
    return Path(file_path).name


def _format_line_range(offset: int, limit: int) -> str:
    """Format line range as L{start}-{end}."""
    start = offset + 1 if offset else 1
    end = start + (limit or 500) - 1
    return f"L{start}-{end}"


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool call for user-friendly display.
    
    Formats each tool type with appropriate display:
    - read_file: filename L{start}-{end}
    - write_file/edit_file: filename + content preview
    - execute: full command (up to 300 chars)
    - write_todos: list todo items
    - web_search: search queries
    - Others: key=value format
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
        
    Returns:
        Formatted string for display
    """
    # File reading tools - show filename and line range
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        filename = _extract_filename(file_path)
        offset = tool_args.get("offset", 0)
        limit = tool_args.get("limit", 500)
        line_range = _format_line_range(offset, limit)
        return f"{filename} ({line_range})"
    
    # File writing tools - show filename only
    if tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)
    
    # File editing tools - show filename only
    if tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)
    
    # Execute command - show full command (important for users)
    if tool_name == "execute":
        command = tool_args.get("command", "")
        # Show up to 300 chars of command
        if len(command) > 300:
            return command[:297] + "..."
        return command
    
    # Todo tools - list the todo items
    if tool_name == "write_todos":
        todos = tool_args.get("todos", [])
        if isinstance(todos, list) and todos:
            # Format todo items like Cursor
            todo_lines = []
            for todo in todos[:5]:  # Show max 5 items
                if isinstance(todo, dict):
                    content = todo.get("content", "")[:50]
                    status = todo.get("status", "pending")
                    status_icon = "✓" if status == "completed" else "○" if status == "pending" else "◐"
                    todo_lines.append(f"{status_icon} {content}")
                else:
                    todo_lines.append(f"○ {str(todo)[:50]}")
            if len(todos) > 5:
                todo_lines.append(f"  ... and {len(todos) - 5} more")
            return "\n    ".join(todo_lines)
        return f"{len(todos)} items"
    
    # Web search - show search queries
    if tool_name == "web_search":
        queries = tool_args.get("queries", "")
        if isinstance(queries, list):
            return ", ".join(str(q)[:40] for q in queries[:3])
        return str(queries)[:80]
    
    # Fetch URL - show the URL
    if tool_name == "fetch_url":
        url = tool_args.get("url", "")
        # Truncate long URLs but keep domain visible
        if len(url) > 60:
            return url[:57] + "..."
        return url
    
    # ls/glob/grep - show path/pattern
    if tool_name == "ls":
        directory = tool_args.get("directory", ".")
        return _extract_filename(directory) if directory != "." else "."
    
    if tool_name == "glob":
        pattern = tool_args.get("pattern", "*")
        path = tool_args.get("path", ".")
        return f"{pattern} in {_extract_filename(path) if path != '.' else '.'}"
    
    if tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", ".")
        return f"'{pattern[:40]}' in {_extract_filename(path) if path != '.' else '.'}"
    
    # Task tool - show description
    if tool_name == "task":
        description = tool_args.get("description", "")
        if len(description) > 80:
            return description[:77] + "..."
        return description
    
    # Default format for other tools
    brief_args = []
    for key, value in tool_args.items():
        if isinstance(value, str):
            if len(value) > 40:
                value = value[:37] + "..."
            brief_args.append(f"{key}={value!r}")
        elif isinstance(value, (int, float, bool)):
            brief_args.append(f"{key}={value}")
        elif isinstance(value, list):
            brief_args.append(f"{key}=[{len(value)} items]")
        elif isinstance(value, dict):
            brief_args.append(f"{key}={{...}}")
    
    args_str = ", ".join(brief_args[:3])
    if len(brief_args) > 3:
        args_str += ", ..."
    
    return args_str if args_str else ""


class StreamDisplayManager:
    """Manages CLI output display state for streaming responses.
    
    Handles visual separation between different content types:
    - Thinking (reasoning_content)
    - Tool calls
    - Final response
    """
    
    SEPARATOR = "─" * 50
    
    def __init__(self, console_instance):
        self.console = console_instance
        self.reset()
    
    def reset(self):
        """Reset state for a new response."""
        self.in_thinking = False
        self.thinking_shown = False
        self.tool_count = 0
        self.in_tool_section = False
        self.response_started = False
        self.has_content_output = False
    
    def start_thinking(self):
        """Start thinking section."""
        if not self.thinking_shown:
            self.console.print()
            self.console.print(f"[dim]{self.SEPARATOR}[/dim]")
            self.console.print("[dim italic]💭 Thinking...[/dim italic]")
            self.thinking_shown = True
            self.in_thinking = True
    
    def stream_thinking(self, content: str):
        """Stream thinking content."""
        self.console.print(content, end="", style="dim")
    
    def end_thinking(self):
        """End thinking section."""
        if self.in_thinking:
            self.console.print()  # End current line
            self.console.print(f"[dim]{self.SEPARATOR}[/dim]")
            self.console.print()
            self.in_thinking = False
    
    def start_tool_section(self):
        """Start tool section with separator."""
        if not self.in_tool_section:
            # End thinking first if active
            if self.in_thinking:
                self.end_thinking()
            # End any previous content
            if self.has_content_output and not self.response_started:
                self.console.print()
            self.console.print()
            self.console.print(f"[dim]{self.SEPARATOR}[/dim]")
            self.console.print("[bold cyan]🔧 Tool Calls[/bold cyan]")
            self.in_tool_section = True
    
    def display_tool(self, tool_name: str, tool_args: dict):
        """Display a single tool call."""
        self.start_tool_section()
        self.tool_count += 1
        
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        display_str = format_tool_display(tool_name, tool_args)
        
        # Add blank line between tools for readability
        if self.tool_count > 1:
            self.console.print()
        
        # Special handling for write_todos - multi-line display
        if tool_name == "write_todos" and "\n" in display_str:
            self.console.print(f"  {icon} ", end="")
            self.console.print(f"{tool_name}", end="", style="bold magenta")
            self.console.print(" tasks:", style="dim")
            self.console.print(f"    {display_str}", style="dim")
        else:
            self.console.print(f"  {icon} ", end="")
            self.console.print(f"{tool_name}", end="", style="bold magenta")
            if display_str:
                self.console.print(f" {display_str}", style="dim")
            else:
                self.console.print()
    
    def end_tool_section(self):
        """End tool section."""
        if self.in_tool_section:
            self.in_tool_section = False
    
    def start_response(self):
        """Start response section."""
        if not self.response_started:
            # End any open sections
            if self.in_thinking:
                self.end_thinking()
            if self.in_tool_section:
                self.end_tool_section()
            
            # Always add newline before response content
            self.console.print()
            self.console.print()  # Extra newline for separation from tool calls
            self.response_started = True
    
    def stream_response(self, content: str):
        """Stream response content."""
        self.start_response()
        self.console.print(content, end="", style=COLORS["agent"])
        self.has_content_output = True
    
    def finalize(self):
        """Finalize output, close any open sections."""
        if self.in_thinking:
            self.end_thinking()
        if self.in_tool_section:
            self.end_tool_section()
        if self.has_content_output:
            self.console.print()  # Final newline


def display_tool_call(tool_name: str, tool_args: dict) -> None:
    """Display a tool call with icon and colored tool name.
    
    Format: {icon} {tool_name} {args_display}
    Example: 📖 read_file test_guardrails.py (L1-400)
    
    Color scheme:
    - Icon: bright (no dim)
    - Tool name: magenta/pink bold
    - Args: dim
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
    """
    icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
    display_str = format_tool_display(tool_name, tool_args)
    
    # Special handling for write_todos - multi-line display
    if tool_name == "write_todos" and "\n" in display_str:
        # Print icon (bright) + colored tool name (magenta) + "tasks:"
        console.print(f"  {icon} ", end="")
        console.print(f"{tool_name}", end="", style="bold magenta")
        console.print(" tasks:", style="dim")
        console.print(f"    {display_str}", style="dim")
    else:
        # Print icon (bright) + colored tool name (magenta) + args (dim)
        console.print(f"  {icon} ", end="")
        console.print(f"{tool_name}", end="", style="bold magenta")
        if display_str:
            console.print(f" {display_str}", style="dim")
        else:
            console.print()  # Ensure newline when no args to display


def run_interactive(agent_config: dict, extra_tool_names: Optional[List[str]] = None,
                    workspace: Optional[Workspace] = None, skills_registry=None,
                    user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Run the interactive CLI with prompt_toolkit support.

    Args:
        agent_config: Agent configuration parameters
        extra_tool_names: Names of additional tools to load
        workspace: Workspace instance for context and memory
        skills_registry: Skills registry for skill triggers
        user_id: User ID for workspace memory isolation
        session_id: Session ID for conversation isolation
    """
    # Suppress logger console output in CLI mode for cleaner UI (unless verbose mode)
    if not agent_config.get("debug_mode"):
        suppress_console_logging()

    # Generate session_id if not provided
    current_session_id = session_id or str(uuid4())
    current_user_id = user_id

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.styles import Style
        use_prompt_toolkit = True
    except ImportError:
        use_prompt_toolkit = False
        console.print("[yellow]prompt_toolkit not installed. Using basic input mode.[/yellow]")
        console.print("[yellow]Install with: pip install prompt_toolkit[/yellow]")
        console.print()

    # Configure extra tools
    extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
    current_agent = _create_agent(agent_config, extra_tools, workspace, skills_registry,
                                   user_id=current_user_id, session_id=current_session_id)

    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        work_dir=agent_config.get("work_dir"),
        extra_tools=extra_tool_names
    )

    # Show workspace, user, and session info
    if workspace and workspace.exists():
        user_info = f" (user: [cyan]{current_user_id}[/cyan])" if current_user_id else ""
        console.print(f"  Workspace: [green]{workspace.path}[/green]{user_info}")
    if current_session_id:
        console.print(f"  Session: [dim]{current_session_id[:8]}...[/dim]")
    if skills_registry and len(skills_registry) > 0:
        triggers = skills_registry.list_triggers()
        if triggers:
            trigger_str = ", ".join(triggers.keys())
            console.print(f"  Skills: [cyan]{len(skills_registry)} loaded[/cyan] (triggers: {trigger_str})")
    console.print()

    if use_prompt_toolkit:
        # Custom completer for @ file mentions and / commands
        class AgenticaCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor

                # Command completion
                if text.startswith("/"):
                    commands = ["/help", "/clear", "/newchat", "/user", "/users", "/tools",
                                "/skills", "/memory", "/save", "/workspace", "/exit", "/quit"]
                    # Add skill triggers
                    if skills_registry:
                        for trigger in skills_registry.list_triggers().keys():
                            commands.append(trigger)
                    for cmd in commands:
                        if cmd.startswith(text):
                            yield Completion(cmd, start_position=-len(text))
                    return

                # File completion after @
                match = re.search(r"@([\w./-]*)$", text)
                if match:
                    partial = match.group(1)
                    completions = get_file_completions(text)
                    for comp in completions:
                        display = comp
                        # Calculate how much to replace
                        yield Completion(
                            comp,
                            start_position=-len(partial),
                            display=display
                        )

        # Key bindings for multi-line input
        bindings = KeyBindings()

        @bindings.add('escape', 'enter')
        def _(event):
            """Alt+Enter to insert newline."""
            event.current_buffer.insert_text('\n')

        @bindings.add('c-j')
        def _(event):
            """Ctrl+J to insert newline."""
            event.current_buffer.insert_text('\n')

        # Style for prompt
        style = Style.from_dict({
            'prompt': 'ansicyan bold',
        })

        # Create history file directory if needed
        history_dir = os.path.dirname(history_file)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)

        session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=AgenticaCompleter(),
            key_bindings=bindings,
            style=style,
            multiline=False,  # Enter submits, Alt+Enter for newline
        )

        def get_input():
            try:
                return session.prompt([('class:prompt', '> ')], multiline=False)
            except KeyboardInterrupt:
                return None
            except EOFError:
                return None
    else:
        # Fallback to basic input
        def get_input():
            try:
                console.print(Text("> ", style="green"), end="")
                sys.stdout.flush()
                return input()
            except KeyboardInterrupt:
                return None
            except EOFError:
                return None

    # Main interaction loop
    while True:
        try:
            user_input = get_input()
            
            if user_input is None:
                console.print("\nExiting...", style="yellow")
                break
            
            user_input = user_input.strip()
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd in ["/exit", "/quit"]:
                    break
                elif cmd == "/help":
                    show_help()
                    continue
                elif cmd == "/tools":
                    console.print("Available additional tools:", style="cyan")
                    for name in sorted(TOOL_REGISTRY.keys()):
                        marker = " [active]" if extra_tool_names and name in extra_tool_names else ""
                        console.print(f"  - {name}{marker}")
                    console.print()
                    console.print("Use --tools <name> when starting CLI to enable tools.", style="dim")
                    continue
                elif cmd == "/skills":
                    if skills_registry and len(skills_registry) > 0:
                        console.print("Available Skills:", style="cyan")
                        for skill in skills_registry.list_all():
                            trigger_info = f" (trigger: [green]{skill.trigger}[/green])" if skill.trigger else ""
                            console.print(f"  - [bold]{skill.name}[/bold]{trigger_info}")
                            console.print(f"    {skill.description[:60]}...", style="dim")
                        console.print()
                        triggers = skills_registry.list_triggers()
                        if triggers:
                            console.print("Triggers:", style="cyan")
                            for trigger, skill_name in triggers.items():
                                console.print(f"  {trigger} -> {skill_name}")
                    else:
                        console.print("No skills loaded. Use --no-skills=false to enable.", style="yellow")
                    continue
                elif cmd == "/memory":
                    if workspace and workspace.exists():
                        memory = workspace.get_memory_prompt(days=3)
                        if memory:
                            console.print("Recent Memory:", style="cyan")
                            console.print(memory[:2000] + "..." if len(memory) > 2000 else memory)
                        else:
                            console.print("No memory found.", style="yellow")
                    else:
                        console.print("Workspace not available.", style="yellow")
                    continue
                elif cmd == "/save":
                    if workspace and workspace.exists():
                        if cmd_args:
                            workspace.write_memory(cmd_args, to_daily=True)
                            console.print(f"Saved to daily memory: {cmd_args[:50]}...", style="green")
                        else:
                            console.print("Usage: /save <text to save>", style="yellow")
                    else:
                        console.print("Workspace not available.", style="yellow")
                    continue
                elif cmd == "/workspace":
                    if workspace:
                        console.print(f"Workspace: [bold]{workspace.path}[/bold]", style="cyan")
                        user_info = f"User: [cyan]{current_user_id}[/cyan]" if current_user_id else "User: [dim]default[/dim]"
                        console.print(user_info)
                        console.print(f"Session: [dim]{current_session_id[:8]}...[/dim]")
                        console.print(f"Exists: {'Yes' if workspace.exists() else 'No'}")
                        if workspace.exists():
                            files = workspace.list_files()
                            console.print("Files:", style="cyan")
                            for fname, exists in files.items():
                                status_icon = "✓" if exists else "✗"
                                console.print(f"  {status_icon} {fname}")
                    else:
                        console.print("Workspace not configured.", style="yellow")
                    continue
                elif cmd == "/newchat":
                    # Create a new session (new session_id, keep user_id)
                    current_session_id = str(uuid4())
                    current_agent = _create_agent(agent_config, extra_tools, workspace, skills_registry,
                                                   user_id=current_user_id, session_id=current_session_id)
                    console.print(f"[green]New chat session created: {current_session_id[:8]}...[/green]")
                    console.print("[dim]Conversation history cleared. Workspace memory preserved.[/dim]")
                    continue
                elif cmd == "/user":
                    if cmd_args:
                        # Switch to new user
                        new_user_id = cmd_args.strip()
                        current_user_id = new_user_id
                        # Update workspace user
                        if workspace:
                            workspace.set_user(new_user_id)
                            if not (workspace._get_user_path() / workspace.config.memory_dir).exists():
                                workspace._initialize_user_dir()
                        # Recreate agent with new user context
                        current_agent = _create_agent(agent_config, extra_tools, workspace, skills_registry,
                                                       user_id=current_user_id, session_id=current_session_id)
                        console.print(f"[green]Switched to user: {new_user_id}[/green]")
                        console.print("[dim]Loaded user's workspace memory.[/dim]")
                    else:
                        # Show current user
                        console.print(f"Current user: [cyan]{current_user_id or 'default'}[/cyan]")
                        console.print(f"Session: [dim]{current_session_id[:8]}...[/dim]")
                        if workspace and current_user_id:
                            info = workspace.get_user_info()
                            console.print(f"Memory count: {info.get('memory_count', 0)}")
                    continue
                elif cmd == "/users":
                    if workspace:
                        users = workspace.list_users()
                        if users:
                            console.print("Registered Users:", style="cyan")
                            for user in users:
                                marker = " [current]" if user == current_user_id else ""
                                info = workspace.get_user_info(user)
                                memory_count = info.get('memory_count', 0)
                                console.print(f"  - {user}{marker} ({memory_count} memories)")
                        else:
                            console.print("No users registered yet.", style="yellow")
                        console.print()
                        console.print("Use /user <id> to switch users.", style="dim")
                    else:
                        console.print("Workspace not configured.", style="yellow")
                    continue
                elif cmd == "/clear":
                    os.system('clear' if os.name != 'nt' else 'cls')
                    # Reset agent conversation but keep user
                    current_session_id = str(uuid4())
                    current_agent = _create_agent(agent_config, extra_tools, workspace, skills_registry,
                                                   user_id=current_user_id, session_id=current_session_id)
                    print_header(
                        agent_config["model_provider"],
                        agent_config["model_name"],
                        work_dir=agent_config.get("work_dir"),
                        extra_tools=extra_tool_names
                    )
                    console.print("[info]Screen cleared and conversation reset.[/info]")
                    continue
                else:
                    # Check if it's a skill trigger
                    if skills_registry:
                        matched_skill = skills_registry.match_trigger(user_input)
                        if matched_skill:
                            # Inject skill prompt and process the message
                            skill_prompt = matched_skill.get_prompt()
                            current_agent.add_instruction(f"\n# {matched_skill.name} Skill\n{skill_prompt}")
                            # Remove trigger prefix from user input
                            if matched_skill.trigger and user_input.lower().startswith(matched_skill.trigger):
                                user_input = user_input[len(matched_skill.trigger):].strip()
                            console.print(f"[dim]Skill activated: {matched_skill.name}[/dim]")
                            # Fall through to normal processing with modified input
                        else:
                            console.print(f"Unknown command: {cmd}. Type /help for available commands.", style="red")
                            continue
                    else:
                        console.print(f"Unknown command: {cmd}. Type /help for available commands.", style="red")
                        continue
            
            # Parse file mentions
            prompt_text, mentioned_files = parse_file_mentions(user_input)
            
            # Inject file contents if any
            final_input = inject_file_contents(prompt_text, mentioned_files)
            
            # Display user message
            display_user_message(user_input)
            
            # Show thinking indicator
            status = console.status(f"[bold {COLORS['thinking']}]Thinking...", spinner="dots")
            status.start()
            spinner_active = True
            
            try:
                response_stream = current_agent.run(final_input, stream=True, stream_intermediate_steps=True)
                
                # Use display manager for clean output
                display = StreamDisplayManager(console)
                shown_tool_count = 0
                
                for chunk in response_stream:
                    if chunk is None:
                        continue
                    
                    # Skip non-display events
                    if chunk.event in ("RunStarted", "RunCompleted", "UpdatingMemory", 
                                       "MultiRoundToolResult", "MultiRoundCompleted"):
                        continue
                    
                    # Handle tool call events
                    if chunk.event == "ToolCallStarted":
                        if spinner_active:
                            status.stop()
                            spinner_active = False
                        
                        # Display new tools
                        if chunk.tools and len(chunk.tools) > shown_tool_count:
                            for tool_info in chunk.tools[shown_tool_count:]:
                                tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                                tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                                if isinstance(tool_args, str):
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except:
                                        tool_args = {"args": tool_args}
                                display.display_tool(tool_name, tool_args)
                            shown_tool_count = len(chunk.tools)
                        continue
                    
                    elif chunk.event == "ToolCallCompleted":
                        continue
                    
                    # Handle multi-round tool calls
                    elif chunk.event == "MultiRoundToolCall":
                        if spinner_active:
                            status.stop()
                            spinner_active = False
                        
                        if chunk.content:
                            tool_content = str(chunk.content)
                            if "(" in tool_content:
                                tool_name = tool_content.split("(")[0]
                                args_part = tool_content[len(tool_name)+1:-1] if tool_content.endswith(")") else ""
                                try:
                                    tool_args = json.loads(args_part) if args_part.startswith("{") else {"args": args_part[:100]}
                                except:
                                    tool_args = {"args": args_part[:100] + "..." if len(args_part) > 100 else args_part}
                            else:
                                tool_name = tool_content
                                tool_args = {}
                            display.display_tool(tool_name, tool_args)
                        continue
                    
                    # Check for content
                    has_content = chunk.content and isinstance(chunk.content, str)
                    has_reasoning = hasattr(chunk, 'reasoning_content') and chunk.reasoning_content
                    
                    if not has_content and not has_reasoning:
                        continue
                    
                    # Handle thinking (reasoning_content only)
                    if has_reasoning and not has_content:
                        if spinner_active:
                            status.stop()
                            spinner_active = False
                        display.start_thinking()
                        display.stream_thinking(chunk.reasoning_content)
                        continue
                    
                    # Handle response content
                    if has_content:
                        if spinner_active:
                            status.stop()
                            spinner_active = False
                        display.stream_response(chunk.content)
                
                # Finalize display
                display.finalize()
                
                # Handle case of no output
                if not display.has_content_output and display.tool_count == 0 and not display.thinking_shown:
                    if spinner_active:
                        status.stop()
                    console.print("[info]Agent returned no content.[/info]")
                
            except Exception as e:
                if spinner_active:
                    status.stop()
                console.print(f"\n[bold red]Error during agent execution: {str(e)}[/bold red]")
            
            console.print()  # Blank line after response
            
        except KeyboardInterrupt:
            console.print("\nOperation interrupted. Type /exit to quit.", style="yellow")
            continue
        except Exception as e:
            console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
            continue

    console.print("\nThank you for using Agentica CLI. Goodbye!", style="bold green")


def main():
    args = parse_args()
    
    # Handle ACP mode for IDE integration
    if args is None or (hasattr(args, 'command') and args.command == 'acp'):
        from agentica.acp.server import ACPServer
        server = ACPServer()
        server.run()
        return

    # Store agent configuration parameters
    agent_config = {
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "debug_mode": args.verbose > 0,
        "work_dir": args.work_dir,
        "enable_multi_round": args.enable_multi_round,
    }
    extra_tool_names = list(args.tools) if args.tools else None

    # Get user_id and session_id from args
    # Default user_id to "default" for single-user CLI usage
    user_id = args.user or "default"
    session_id = args.session or str(uuid4())

    # Initialize workspace with user_id for memory isolation
    workspace = None
    if not args.no_workspace:
        workspace_path = args.workspace  # Can be None for default
        workspace = Workspace(workspace_path, user_id=user_id)
        if not workspace.exists():
            workspace.initialize()
        elif user_id:
            # Ensure user directory exists
            workspace._initialize_user_dir()

    # Load skills
    skills_registry = None
    if not args.no_skills:
        # Suppress logging during skill loading for cleaner output
        if args.verbose == 0:
            suppress_console_logging()
        load_skills()
        skills_registry = get_skill_registry()

    if args.query:
        # Non-interactive mode
        console.print(f"Running query: {args.query}", style="cyan")
        tools_info = f", Extra Tools: {', '.join(extra_tool_names)}" if extra_tool_names else ""
        user_info = f", User: {user_id}" if user_id else ""
        console.print(
            f"Model: {agent_config['model_provider']}/{agent_config['model_name']}{tools_info}{user_info}",
            style="magenta")

        extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
        agent_instance = _create_agent(agent_config, extra_tools, workspace, skills_registry,
                                        user_id=user_id, session_id=session_id)
        response = agent_instance.run(args.query, stream=True)
        for chunk in response:
            if chunk and chunk.content:
                console.print(chunk.content, end="")
        console.print()  # final newline
    else:
        # Interactive mode
        run_interactive(agent_config, extra_tool_names, workspace, skills_registry,
                        user_id=user_id, session_id=session_id)


if __name__ == "__main__":
    main()
