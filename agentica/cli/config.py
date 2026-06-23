# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI configuration - constants, tool registry, argument parsing, model/agent creation
"""

import argparse
import importlib
import os
import sys
from typing import List, Optional, Any

from rich.console import Console

from agentica import Agent, OpenAIChat, MoonshotChat, AzureOpenAIChat, YiChat, ZhipuAIChat, DeepSeekChat, ArkChat
from agentica.model.anthropic.claude import Claude
from agentica.agent.config import (
    ExperienceConfig,
    SkillUpgradeConfig,
    WorkspaceMemoryConfig,
)
from agentica.config import AGENTICA_HOME
from agentica.workspace import Workspace

# Plain Rich console — used outside TUI mode (non-interactive queries, startup).
_plain_console = Console()

# Active console — set to ChatConsole when TUI is running,
# falls back to the plain Rich console for non-TUI usage.
_active_console = None


def get_console():
    """Return the active console (ChatConsole during TUI, plain Rich otherwise)."""
    if _active_console is not None:
        return _active_console
    return _plain_console


def set_active_console(console_instance):
    """Set the active console (call with ChatConsole when entering TUI, None when leaving)."""
    global _active_console
    _active_console = console_instance


# Backward-compat alias — modules that imported `console` directly from config
# now get the plain console. All runtime output should use get_console().
console = _plain_console

history_file = os.path.join(AGENTICA_HOME, "cli_history.txt")


def _generate_session_id() -> str:
    """Generate a UUID session ID (CC convention)."""
    from uuid import uuid4

    return str(uuid4())


# Builtin tools — single source of truth for all CLI display/listing.
BUILTIN_TOOLS = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "web_search",
    "fetch_url",
    "task",
]

# Tool icons for CLI display
TOOL_ICONS = {
    "ls": "📁",
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "✂️",
    "glob": "🔍",
    "grep": "🔎",
    "execute": "⚡",
    "web_search": "🌐",
    "fetch_url": "🔗",
    "write_todos": "📋",
    "task": "🤖",
    "default": "🔧",
}

# Tool registry - maps tool names to (module_name, class_name, category, description)
# Module path: agentica.tools.{module_name}_tool.{ClassName}
TOOL_REGISTRY = {
    # AI/ML Tools
    "cogvideo": ("cogvideo", "CogVideoTool", "AI/ML", "Text-to-video generation with CogVideo"),
    "cogview": ("cogview", "CogViewTool", "AI/ML", "Text-to-image generation with CogView"),
    "dalle": ("dalle", "DalleTool", "AI/ML", "Image generation with DALL-E"),
    "image_analysis": ("image_analysis", "ImageAnalysisTool", "AI/ML", "Image analysis and description"),
    "ocr": ("ocr", "OcrTool", "AI/ML", "Optical character recognition"),
    "video_analysis": ("video_analysis", "VideoAnalysisTool", "AI/ML", "Video content analysis"),
    "volc_tts": ("volc_tts", "VolcTtsTool", "AI/ML", "Text-to-speech with Volcengine"),
    # Search Tools
    "arxiv": ("arxiv", "ArxivTool", "Search", "Search academic papers on arXiv"),
    "baidu_search": ("baidu_search", "BaiduSearchTool", "Search", "Web search via Baidu"),
    "dblp": ("dblp", "DblpTool", "Search", "Search computer science papers on DBLP"),
    "duckduckgo": ("duckduckgo", "DuckDuckGoTool", "Search", "Web search via DuckDuckGo"),
    "search_bocha": ("search_bocha", "SearchBochaTool", "Search", "Web search via Bocha"),
    "search_exa": ("search_exa", "SearchExaTool", "Search", "Web search via Exa"),
    "search_serper": ("search_serper", "SearchSerperTool", "Search", "Web search via Serper (Google)"),
    "web_search_pro": ("web_search_pro", "WebSearchProTool", "Search", "Advanced web search with ZhipuAI"),
    "wikipedia": ("wikipedia", "WikipediaTool", "Search", "Search and read Wikipedia articles"),
    # Web/Network Tools
    "browser": ("browser", "BrowserTool", "Web", "Headless browser for web automation"),
    "jina": ("jina", "JinaTool", "Web", "Web content extraction via Jina Reader"),
    "newspaper": ("newspaper", "NewspaperTool", "Web", "Article extraction from news URLs"),
    "url_crawler": ("url_crawler", "UrlCrawlerTool", "Web", "Recursive URL crawling"),
    # File/Code Tools
    "calculator": ("calculator", "CalculatorTool", "Code & Files", "Mathematical expression evaluation"),
    "code": ("code", "CodeTool", "Code & Files", "Code generation and execution"),
    "edit": ("edit", "EditTool", "Code & Files", "File editing with diff patches"),
    "file": ("file", "FileTool", "Code & Files", "File system operations"),
    "run_nb_code": ("run_nb_code", "RunNbCodeTool", "Code & Files", "Execute Jupyter notebook code"),
    "run_python_code": ("run_python_code", "RunPythonCodeTool", "Code & Files", "Execute Python code snippets"),
    "shell": ("shell", "ShellTool", "Code & Files", "Shell command execution"),
    "string": ("string", "StringTool", "Code & Files", "String manipulation utilities"),
    "text_analysis": ("text_analysis", "TextAnalysisTool", "Code & Files", "Text analysis and NLP"),
    "workspace": ("workspace", "WorkspaceTool", "Code & Files", "Workspace file management"),
    # Data Tools
    "hackernews": ("hackernews", "HackerNewsTool", "Data", "Fetch Hacker News stories"),
    "sql": ("sql", "SqlTool", "Data", "SQL database queries"),
    "weather": ("weather", "WeatherTool", "Data", "Weather information"),
    "yfinance": ("yfinance", "YFinanceTool", "Data", "Financial data from Yahoo Finance"),
    # Integration Tools
    "airflow": ("airflow", "AirflowTool", "Integration", "Apache Airflow DAG management"),
    "apify": ("apify", "ApifyTool", "Integration", "Web scraping via Apify"),
    "mcp": ("mcp", "MCPTool", "Integration", "Model Context Protocol integration"),
    "memori": ("memori", "MemoriTool", "Integration", "Long-term memory management"),
    "skill": ("skill", "SkillTool", "Integration", "Skill document management"),
    "video_download": ("video_download", "VideoDownloadTool", "Integration", "Video download from URLs"),
}

# Model provider registry - maps provider name to model class
MODEL_REGISTRY = {
    "openai": OpenAIChat,
    "azure": AzureOpenAIChat,
    "moonshot": MoonshotChat,
    "zhipuai": ZhipuAIChat,
    "deepseek": DeepSeekChat,
    "yi": YiChat,
    "ark": ArkChat,
    "anthropic": Claude,
}

# Example models for each provider (for /model command display)
EXAMPLE_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5.2", "o3-mini"],
    "azure": ["gpt-4o", "gpt-4o-mini"],
    "moonshot": ["kimi-k2.5", "moonshot-v1-128k"],
    "zhipuai": ["glm-5", "glm-4-flash", "glm-4.7-flash"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner", "deepseek-chat"],
    "yi": ["yi-lightning", "yi-large"],
    "ark": ["doubao-1.5-pro-32k", "doubao-1.5-lite-32k", "doubao-1.5-vision-pro-32k"],
    "anthropic": ["claude-opus-4.8", "claude-sonnet-4.5", "claude-3-5-sonnet-20241022"],
}


def _get_tool_import_path(tool_name: str) -> str:
    """Get full import path for a tool."""
    module_name, class_name, _cat, _desc = TOOL_REGISTRY[tool_name]
    return f"agentica.tools.{module_name}_tool.{class_name}"


def parse_args():
    # Check if running in ACP mode (special handling)
    if len(sys.argv) > 1 and sys.argv[1] == "acp":
        return None  # Signal to run in ACP mode

    # `agentica setup` — re-run the model provider onboarding wizard.
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return argparse.Namespace(command="setup")

    # `agentica doctor` — run the environment health check and exit.
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        doctor_parser = argparse.ArgumentParser(description="Run Agentica environment diagnostics")
        doctor_parser.add_argument(
            "--enable-diagnostics", action="store_true", help="Report diagnostics as enabled for this invocation"
        )
        doctor_parser.add_argument(
            "--diagnostics-server",
            action="append",
            dest="diagnostics_servers",
            default=None,
            help="LSP server to check (repeatable, default: pyright)",
        )
        doctor_parser.add_argument(
            "--work_dir", type=str, default=None, help="Workspace directory to inspect for git/LSP suitability"
        )
        args = doctor_parser.parse_args(sys.argv[2:])
        args.command = "doctor"
        return args

    if len(sys.argv) > 1 and sys.argv[1] in ("skills", "extensions"):
        parser = argparse.ArgumentParser(description="Manage Agentica skills")
        subparsers = parser.add_subparsers(dest="skills_command", required=True)

        install_parser = subparsers.add_parser(
            "install",
            help="Install skills from a git repository URL or local directory",
        )
        install_parser.add_argument("source", help="Git repository URL or local path")
        install_parser.add_argument(
            "--target-dir",
            default=None,
            help="Install target directory (default: ~/.agentica/skills)",
        )
        install_parser.add_argument(
            "--force",
            action="store_true",
            help="Replace already installed skills with the same name",
        )

        list_parser = subparsers.add_parser(
            "list",
            help="List installed skills from the target directory",
        )
        list_parser.add_argument(
            "--target-dir",
            default=None,
            help="Skill directory to inspect (default: ~/.agentica/skills)",
        )

        remove_parser = subparsers.add_parser(
            "remove",
            help="Remove an installed skill by name",
        )
        remove_parser.add_argument("skill_name", help="Installed skill directory name")
        remove_parser.add_argument(
            "--target-dir",
            default=None,
            help="Skill directory to modify (default: ~/.agentica/skills)",
        )

        reload_parser = subparsers.add_parser(
            "reload",
            help="Reload skills from disk and print the current registry count",
        )
        reload_parser.add_argument(
            "--target-dir",
            default=None,
            help="Skill directory to inspect (default: ~/.agentica/skills)",
        )

        args = parser.parse_args(sys.argv[2:])
        args.command = "skills"
        return args

    parser = argparse.ArgumentParser(description="CLI for agentica")

    parser.add_argument("--query", type=str, help="Question to ask the LLM", default=None)
    # Default is None so saved CLI config (from the first-run wizard) can take
    # effect; main.py resolves args > saved config > hardcoded default.
    parser.add_argument(
        "--model_provider", type=str, choices=list(MODEL_REGISTRY.keys()), help="LLM model provider", default=None
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="LLM model name to use, can be deepseek-v4-flash/deepseek-v4-pro/gpt-5/glm-4.7-flash/...",
        default=None,
    )
    parser.add_argument("--base_url", type=str, help="API base URL for the LLM")
    parser.add_argument("--api_key", type=str, help="API key for the LLM")
    parser.add_argument("--max_tokens", type=int, help="Max output tokens (output limit) for the LLM")
    parser.add_argument("--temperature", type=float, help="Temperature for the LLM")
    parser.add_argument("--top_p", type=float, help="Top-p (nucleus sampling) for the LLM")
    parser.add_argument(
        "--context_window",
        type=int,
        help="Context window size (context limit) in tokens; overrides the value auto-detected "
        "from the model catalog. Used for context-budget display and compression "
        "(not sent to the API)",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high", "max"],
        help="Reasoning/thinking depth for thinking models; DeepSeek CLI defaults to max",
    )

    # Auxiliary model (compression / memory extraction / experience lifecycle).
    # Omit to reuse the main model (single API key). Any field can differ — a
    # different provider, a different API key, a different base_url, etc.
    parser.add_argument(
        "--aux_model_provider",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Provider for DeepAgent auxiliary_model (defaults to --model_provider)",
    )
    parser.add_argument(
        "--aux_model_name", type=str, help="Model id for auxiliary_model (required to enable a different aux)"
    )
    parser.add_argument("--aux_base_url", type=str, help="Base URL for auxiliary_model")
    parser.add_argument("--aux_api_key", type=str, help="API key for auxiliary_model")

    # Task model (used by the `task` subagent tool). Same rules as aux.
    parser.add_argument(
        "--task_model_provider",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Provider for the task-subagent model (defaults to --model_provider)",
    )
    parser.add_argument("--task_model_name", type=str, help="Model id for the task-subagent model")
    parser.add_argument("--task_base_url", type=str, help="Base URL for the task-subagent model")
    parser.add_argument("--task_api_key", type=str, help="API key for the task-subagent model")

    parser.add_argument("--debug", type=int, help="enable verbose mode", default=0)
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Show only inter-agent CHAT messages (suppress INFO/DEBUG/WARNING)",
    )
    parser.add_argument("--work_dir", type=str, help="Working directory for file operations", default=None)
    parser.add_argument(
        "--tools",
        nargs="*",
        choices=list(TOOL_REGISTRY.keys()),
        help="Additional tools to enable (on top of built-in tools)",
    )
    parser.add_argument(
        "--sync-memories-to-global-agent-md",
        action="store_true",
        help="Sync durable memories into ~/.agentica/AGENTS.md",
    )
    parser.add_argument(
        "--no-experience", action="store_true", help="Disable DeepAgent experience capture and self-evolution hooks"
    )
    parser.add_argument(
        "--sync-experience-to-global-agent-md",
        action="store_true",
        help="Sync confirmed experiences into ~/.agentica/AGENTS.md",
    )
    parser.add_argument(
        "--enable-skill-upgrade", action="store_true", help="Enable automatic experience-to-skill upgrade"
    )
    parser.add_argument(
        "--skill-upgrade-mode",
        type=str,
        default="shadow",
        choices=["shadow", "draft"],
        help="Skill upgrade mode when --enable-skill-upgrade is set",
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="Workspace directory path (default: ~/.agentica/workspace)"
    )
    parser.add_argument("--no-workspace", action="store_true", help="Disable workspace context injection")
    parser.add_argument(
        "--enable-diagnostics", action="store_true", help="Enable edit-time LSP diagnostics for built-in file tools"
    )
    parser.add_argument(
        "--diagnostics-server",
        action="append",
        dest="diagnostics_servers",
        default=None,
        help="LSP server to use for diagnostics (repeatable, default: pyright)",
    )
    parser.add_argument("--enable-skills", action="store_true", help="Enable skills loading (disabled by default)")
    parser.add_argument("--allow-all", action="store_true", help="Auto-approve all tool executions without prompting")
    parser.add_argument(
        "--permissions",
        type=str,
        default="auto",
        choices=["allow-all", "auto", "strict"],
        help="Permission mode: allow-all (no prompts), auto (prompt for writes), strict (prompt for all)",
    )
    parser.add_argument(
        "command", nargs="?", choices=["acp"], help="Run in ACP mode for IDE integration (agentica acp)"
    )
    return parser.parse_args()


def configure_tools(tool_names: Optional[List[str]] = None) -> List[Any]:
    """Configure and instantiate tools based on their names."""
    if not tool_names:
        return []

    con = get_console()
    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            con.print(f"[yellow]Warning: Tool '{name}' not recognized. Skipping.[/yellow]")
            continue

        try:
            import_path = _get_tool_import_path(name)
            module_path, class_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            tools.append(tool_instance)
            con.print(f"[green]Loaded additional tool: {name}[/green]")
        except ImportError as e:
            con.print(f"[red]Error: Could not import tool '{name}'. Missing dependencies? {str(e)}[/red]")
        except Exception as e:
            con.print(f"[red]Error: Failed to initialize tool '{name}': {str(e)}[/red]")

    return tools


def get_model(
    model_provider,
    model_name,
    base_url=None,
    api_key=None,
    max_tokens=None,
    temperature=None,
    reasoning_effort=None,
    top_p=None,
    context_window=None,
):
    """Create a model instance based on the provider name.

    Uses MODEL_REGISTRY for provider lookup instead of if/elif chains.
    """
    params = {"id": model_name}
    if api_key is not None:
        params["api_key"] = api_key
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    # context_window is a capability field (not sent to the API) used for
    # context-budget display and compression. A user-set value overrides the
    # value auto-filled from the model catalog. Every Model subclass has it.
    if context_window is not None:
        params["context_window"] = context_window
    # Anthropic's Claude has no base_url / reasoning_effort fields; the SDK
    # reads its endpoint from client defaults, so skip those params for it.
    if model_provider != "anthropic":
        if base_url is not None:
            params["base_url"] = base_url
        if model_provider == "deepseek":
            params["reasoning_effort"] = reasoning_effort or "max"
        elif reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

    model_class = MODEL_REGISTRY.get(model_provider)
    if model_class is None:
        raise ValueError(f"Unsupported model provider: {model_provider}. Supported: {', '.join(MODEL_REGISTRY.keys())}")
    return model_class(**params)


def _build_sibling_model(agent_config: dict, prefix: str):
    """Build an auxiliary/task sibling model from CLI args.

    Returns None when no `{prefix}_model_name` was provided — in that case
    the caller should either not pass the arg to DeepAgent (so it reuses
    the main model) or pass the main model explicitly.

    Fields fall through to main-model values when a sibling field is None,
    so the user can override just the pieces that differ (e.g. only the
    model name, or only the base_url+api_key).
    """
    sibling_name = agent_config.get(f"{prefix}_model_name")
    if not sibling_name:
        return None
    return get_model(
        model_provider=agent_config.get(f"{prefix}_model_provider") or agent_config["model_provider"],
        model_name=sibling_name,
        base_url=agent_config.get(f"{prefix}_base_url") or agent_config.get("base_url"),
        api_key=agent_config.get(f"{prefix}_api_key") or agent_config.get("api_key"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
        reasoning_effort=agent_config.get("reasoning_effort"),
        top_p=agent_config.get("top_p"),
        context_window=agent_config.get("context_window"),
    )


def _build_cli_experience_config(agent_config: dict) -> ExperienceConfig:
    """Build the CLI's opinionated DeepAgent experience configuration."""
    skill_upgrade = None
    if agent_config.get("enable_skill_upgrade"):
        skill_upgrade = SkillUpgradeConfig(
            mode=agent_config.get("skill_upgrade_mode") or "shadow",
        )
    return ExperienceConfig(
        capture_tool_errors=True,
        capture_user_corrections=True,
        # Pure success sequences (e.g. "read_file x76") teach the model nothing
        # actionable; only failures and corrections carry real lessons.
        capture_success_patterns=False,
        # Batch the LLM judge: 1 call per 10 turns instead of per turn.
        judge_every_n_turns=10,
        judge_min_seconds_between=60,
        sync_to_global_agent_md=bool(agent_config.get("sync_experience_to_global_agent_md")),
        skill_upgrade=skill_upgrade,
    )


def _build_cli_memory_config(agent_config: dict) -> WorkspaceMemoryConfig:
    """Build the CLI's opinionated DeepAgent memory configuration."""
    return WorkspaceMemoryConfig(
        auto_archive=True,
        auto_extract_memory=True,
        # Batch the LLM extraction: 1 call per 10 turns instead of per turn.
        extract_every_n_turns=10,
        extract_min_seconds_between=60,
        load_workspace_context=True,
        load_workspace_memory=True,
        max_memory_entries=10,
        sync_memories_to_global_agent_md=bool(agent_config.get("sync_memories_to_global_agent_md")),
    )


def create_agent(
    agent_config: dict, extra_tools: Optional[List] = None, workspace: Optional[Workspace] = None, skills_registry=None
):
    """Helper to create or recreate an Agent with built-in tools and current config."""
    model = get_model(
        model_provider=agent_config["model_provider"],
        model_name=agent_config["model_name"],
        base_url=agent_config.get("base_url"),
        api_key=agent_config.get("api_key"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
        reasoning_effort=agent_config.get("reasoning_effort"),
        top_p=agent_config.get("top_p"),
        context_window=agent_config.get("context_window"),
    )

    # Optional sibling models. When the user doesn't pass --aux_model_name /
    # --task_model_name, these stay None and DeepAgent falls back to the
    # main model — same API key drives the whole stack.
    auxiliary_model = _build_sibling_model(agent_config, "aux")
    task_model = _build_sibling_model(agent_config, "task")
    experience_config = _build_cli_experience_config(agent_config)
    long_term_memory_config = _build_cli_memory_config(agent_config)

    # Build extra tools list
    work_dir = agent_config.get("work_dir")

    # Use DeepAgent for full-featured CLI experience.
    from agentica.agent.deep import DeepAgent
    from agentica.tools.skill_tool import SkillTool

    new_agent = DeepAgent(
        model=model,
        auxiliary_model=auxiliary_model,
        task_model=task_model,
        tools=extra_tools or [],  # user-specified extra tools
        work_dir=work_dir,
        workspace=workspace,
        session_id=agent_config.get("session_id") or _generate_session_id(),
        debug=agent_config["debug"],
        enable_experience_capture=agent_config.get("enable_experience_capture", True),
        experience_config=experience_config,
        long_term_memory_config=long_term_memory_config,
        include_user_input=True,  # CLI is interactive, always enable human-in-the-loop
        enable_diagnostics=bool(agent_config.get("enable_diagnostics")),
        diagnostics_servers=agent_config.get("diagnostics_servers"),
    )

    if skills_registry and len(skills_registry) > 0:
        has_skill_tool = any(isinstance(tool, SkillTool) for tool in (new_agent.tools or []))
        if not has_skill_tool:
            skills_summary = skills_registry.get_skills_summary()
            if skills_summary:
                new_agent.add_session_guidance(skills_summary)
    return new_agent
