# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI configuration - constants, tool registry, argument parsing, model/agent creation
"""

import argparse
import importlib
import inspect
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
from agentica.config import AGENTICA_CACHE_DIR
from agentica.tools.base import Tool
from agentica.version import __version__
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

# Re-exported so agentica.cli.interactive can share one cache-root constant.
CACHE_DIR = AGENTICA_CACHE_DIR

history_file = os.path.join(CACHE_DIR, "cli_history.txt")


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
            "--enable-diagnostics",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Report diagnostics as enabled (default: on; use --no-enable-diagnostics to disable)",
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

    # `agentica cron daemon` — run the standalone cron scheduler (no interactive CLI).
    if len(sys.argv) > 1 and sys.argv[1] == "cron":
        cron_parser = argparse.ArgumentParser(description="Agentica cron scheduler")
        cron_sub = cron_parser.add_subparsers(dest="cron_command", required=True)
        daemon_parser = cron_sub.add_parser("daemon", help="Run the cron scheduler in the foreground (Ctrl-C to stop)")
        daemon_parser.add_argument(
            "--interval", type=int, default=60, help="Seconds between schedule checks (default: 60)"
        )
        daemon_parser.add_argument("--verbose", action="store_true", help="Verbose tick logging")
        args = cron_parser.parse_args(sys.argv[2:])
        args.command = "cron"
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
    parser.add_argument(
        "-V", "-v", "--version", action="version", version=f"agentica {__version__}"
    )

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

    # Auxiliary model: the cheap/fast model for all non-user-facing LLM work — memory
    # extraction, context compression, user-correction classification, goal
    # judging, skill upgrade, AND the `task` subagent tool. Omit to reuse the
    # main model. Any field can differ (provider / api_key / base_url).
    parser.add_argument(
        "--auxiliary_model_provider",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Provider for the auxiliary model (defaults to --model_provider)",
    )
    parser.add_argument(
        "--auxiliary_model_name",
        type=str,
        help="Model id for the auxiliary model (background tasks + `task` subagent; required to enable a separate auxiliary)",
    )
    parser.add_argument("--auxiliary_base_url", type=str, help="Base URL for the auxiliary model")
    parser.add_argument("--auxiliary_api_key", type=str, help="API key for the auxiliary model")

    # Prompt caching for OpenAI-compatible proxies that front Anthropic Claude
    # (e.g. Venus). Default None = use the active profile's value (or off if the
    # profile doesn't set it); --enable_cache_control / --no-enable_cache_control
    # force on/off for this run.
    parser.add_argument(
        "--enable_cache_control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Anthropic-style cache_control blocks (for Venus-style proxies).",
    )
    parser.add_argument(
        "--cache_control_messages",
        type=int,
        default=None,
        help="Max cache breakpoints on trailing messages (Anthropic caps total at 4).",
    )
    parser.add_argument(
        "--cache_control_session_header",
        type=str,
        default=None,
        help="Sticky-routing header name for cache hits (e.g. Venus-Session-Id).",
    )

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
        "--enable-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable edit-time LSP diagnostics for built-in file tools (default: on; use --no-enable-diagnostics to disable)",
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
        default="allow-all",
        choices=["ask", "auto", "allow-all"],
        help="Permission mode: ask (read-only tools only), auto (writes restricted to work_dir), "
        "allow-all (no restriction; default — the CLI is a single-user tool)",
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
    enable_cache_control=None,
    cache_control_messages=None,
    cache_control_session_header=None,
    extra_body=None,
    extra_headers=None,
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
    # base_url applies to every provider, including anthropic: a corporate
    # proxy that forwards the native /v1/messages endpoint (e.g. Venus
    # http://.../llmproxy/anthropic) can be targeted, and the Claude client
    # seeds the bearer header for such proxies.
    if base_url is not None:
        params["base_url"] = base_url
    # OpenAI-only tuning: reasoning_effort + raw passthrough dicts. Anthropic
    # takes reasoning_effort too (mapped to adaptive thinking inside the Claude
    # model class), but NOT the OpenAI extra_body/extra_headers passthrough.
    if model_provider == "anthropic":
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
    else:
        if model_provider == "deepseek":
            params["reasoning_effort"] = reasoning_effort or "max"
        elif reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        # Raw passthrough dicts for endpoints whose tuning knobs don't map to
        # a standard OpenAI param (e.g. Hunyuan's taiji gateway wants
        # reasoning_effort inside extra_body.chat_template_kwargs).
        if extra_body is not None:
            params["extra_body"] = extra_body
        if extra_headers is not None:
            params["extra_headers"] = extra_headers

    model_class = MODEL_REGISTRY.get(model_provider)
    if model_class is None:
        raise ValueError(f"Unsupported model provider: {model_provider}. Supported: {', '.join(MODEL_REGISTRY.keys())}")
    # Prompt caching. ``enable_cache_control`` applies to any model class that
    # declares it (OpenAIChat for OpenAI-compatible proxies fronting Claude,
    # Claude itself for native Anthropic caching) — filling it in CLI/config
    # takes effect everywhere. The OpenAIChat-only message/header knobs are not
    # passed to Claude, which manages its own message caching natively.
    if inspect.isclass(model_class):
        if enable_cache_control is not None and hasattr(model_class, "enable_cache_control"):
            params["enable_cache_control"] = enable_cache_control
        if issubclass(model_class, OpenAIChat):
            if cache_control_messages is not None:
                params["cache_control_messages"] = cache_control_messages
            if cache_control_session_header is not None:
                params["cache_control_session_header"] = cache_control_session_header
    return model_class(**params)


def _build_sibling_model(agent_config: dict, prefix: str):
    """Build an auxiliary/task sibling model from CLI args.

    Returns None when no `{prefix}_model_name` was provided — in that case
    the caller should either not pass the arg to DeepAgent (so it reuses
    the main model) or pass the main model explicitly.

    Fields fall through to main-model values when a sibling field is None AND
    the sibling shares the main model's provider, so the user can override just
    the pieces that differ (e.g. only the model name, or only the
    base_url+api_key). When the sibling uses a *different* provider, base_url
    and api_key are NOT inherited from the main model — a different provider's
    endpoint/key never works for the main provider, so falling back would
    silently produce a broken client. In that case a None base_url/api_key is
    passed to the model factory, which uses the provider preset / env var.
    """
    sibling_name = agent_config.get(f"{prefix}_model_name")
    if not sibling_name:
        return None
    main_provider = agent_config["model_provider"]
    sibling_provider = agent_config.get(f"{prefix}_model_provider") or main_provider
    same_provider = sibling_provider == main_provider
    return get_model(
        model_provider=sibling_provider,
        model_name=sibling_name,
        base_url=agent_config.get(f"{prefix}_base_url") or (agent_config.get("base_url") if same_provider else None),
        api_key=agent_config.get(f"{prefix}_api_key") or (agent_config.get("api_key") if same_provider else None),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
        reasoning_effort=agent_config.get("reasoning_effort"),
        top_p=agent_config.get("top_p"),
        context_window=agent_config.get("context_window"),
        enable_cache_control=agent_config.get("enable_cache_control"),
        cache_control_messages=agent_config.get("cache_control_messages"),
        cache_control_session_header=agent_config.get("cache_control_session_header"),
        # Auxiliary passthrough dicts are their own field (auxiliary_extra_*),
        # never inherited from the main model even when same provider — a
        # different deployment/endpoint may not want the same raw params.
        extra_body=agent_config.get(f"{prefix}_extra_body"),
        extra_headers=agent_config.get(f"{prefix}_extra_headers"),
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


def _get_active_skill_names(agent) -> List[str]:
    """Best-effort enabled skill names from the agent's SkillTool.

    External boundary: SkillTool registry init / disk-backed usage loading can
    fail in odd environments, and environment_context construction must never
    break agent creation, so any failure omits the line instead of raising.
    Returns [] when the agent has no SkillTool.
    """
    from agentica.tools.skill_tool import SkillTool

    for tool in agent.tools or []:
        if isinstance(tool, SkillTool):
            try:
                skills = tool._get_enabled_skills()
            except Exception:
                return []
            return [s.name for s in skills]
    return []


def _build_environment_context(agent: Any, agent_config: dict) -> Optional[str]:
    """Build a stable self-description block for the agent's system prompt.

    Only includes information that rarely changes during a session so the
    prompt prefix stays cache-friendly: framework, model endpoint, auxiliary model,
    active tools/skills, builtin subagent types, slash commands, and extension
    hints. Intentionally excludes work_dir (already injected by prompts.py) and
    cost/context usage (owned by /status). Reused by _apply_profile to refresh
    the block after a model/profile switch — call with the live agent and the
    post-switch agent_config.
    """
    provider = agent_config.get("model_provider")
    model_name = agent_config.get("model_name")
    base_url = agent_config.get("base_url")

    lines: List[str] = ["You are an Agentica agent running in CLI mode."]
    lines.append("- Framework: Agentica")
    if provider and model_name:
        endpoint = f"  (endpoint: {base_url})" if base_url else ""
        lines.append(f"- Model: {provider}/{model_name}{endpoint}")

    auxiliary_provider = agent_config.get("auxiliary_model_provider")
    auxiliary_model_name = agent_config.get("auxiliary_model_name")
    if auxiliary_provider and auxiliary_model_name:
        lines.append(
            f"- Auxiliary model: {auxiliary_provider}/{auxiliary_model_name}  (background calls + task subagent)"
        )

    tool_names = sorted(
        name for t in (agent.tools or []) if isinstance(t, Tool) and t.functions for name in t.functions.keys()
    )
    lines.append(f"- Active tools: {', '.join(tool_names) if tool_names else 'none'}")

    skill_names = _get_active_skill_names(agent)
    if skill_names:
        lines.append(f"- Active skills: {', '.join(sorted(skill_names))}")

    subagent_types = ["explore", "research", "code"]
    try:
        from agentica.subagent import get_custom_subagent_configs

        subagent_types.extend(sorted(get_custom_subagent_configs().keys()))
    except Exception:
        pass
    lines.append(f"- Subagent types: {', '.join(subagent_types)}")
    lines.append("- Slash commands: /rename /resume /status /model /tools /skills /agents /config /cost /permissions /help /exit")
    lines.append("- To extend: /skills install <name>, /agents create <name>")

    return "\n".join(lines)


def create_agent(
    agent_config: dict,
    extra_tools: Optional[List] = None,
    workspace: Optional[Workspace] = None,
    skills_registry=None,
    ask_user_question_callback=None,
    enable_cron_immediate_run: bool = True,
    permission_mode: Optional[str] = None,
):
    """Helper to create or recreate an Agent with built-in tools and current config.

    ask_user_question_callback: optional ``(prompt, options) -> str`` used by the
        ask_user_question/confirm tools. The interactive CLI passes a prompt_toolkit-aware
        callback so the tool reads via the TUI input box instead of a bare
        ``input()`` (which deadlocks against prompt_toolkit's stdin ownership in
        the background agent thread).
    enable_cron_immediate_run: when True (interactive CLI) the ``cronjob`` tool's
        ``action='run'`` executes a job once immediately and returns its output.
        Set False for cron-spawned agents so a scheduled job cannot recursively
        trigger further immediate runs.
    permission_mode: unified 3-tier tool permission ("ask"/"auto"/"allow-all",
        see agentica.agent.permissions). Falls back to ``agent_config["permissions"]``,
        then "allow-all" (the CLI's actual --permissions default; see parse_args()).
    """
    if permission_mode is None:
        permission_mode = agent_config.get("permissions", "allow-all")
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
        enable_cache_control=agent_config.get("enable_cache_control"),
        cache_control_messages=agent_config.get("cache_control_messages"),
        cache_control_session_header=agent_config.get("cache_control_session_header"),
        extra_body=agent_config.get("extra_body"),
        extra_headers=agent_config.get("extra_headers"),
    )

    # Auxiliary model: the cheap/fast model for all background LLM work (memory
    # extraction, compression, classification, goal judging, skill upgrade) AND
    # the `task` subagent tool. When --auxiliary_model_name is unset this stays None
    # and DeepAgent falls back to the main model for auxiliary work.
    auxiliary_model = _build_sibling_model(agent_config, "auxiliary")
    # The task subagent tool shares the auxiliary model (one cheap model for all
    # non-user-facing LLM work), so the CLI exposes only main + auxiliary.
    task_model = auxiliary_model
    experience_config = _build_cli_experience_config(agent_config)
    long_term_memory_config = _build_cli_memory_config(agent_config)

    # Build extra tools list
    work_dir = agent_config.get("work_dir")

    # Resolve an explicit user_id for the agent so session storage is keyed
    # deterministically. The CLI has no first-class user concept, so it falls
    # back to the workspace's user_id (if any) or "default". Passing this
    # explicitly (instead of leaving user_id=None) makes the CLI's session
    # directory match the Web gateway's (which uses settings.default_user_id,
    # "default") without relying on downstream None->"default" normalization.
    from agentica.workspace import Workspace as _Workspace

    cli_user_id = agent_config.get("user_id")
    if cli_user_id is None and workspace is not None:
        cli_user_id = workspace.user_id
    if cli_user_id is None:
        cli_user_id = _Workspace.DEFAULT_USER_ID

    # Use DeepAgent for full-featured CLI experience.
    from agentica.agent.deep import DeepAgent
    from agentica.tools.skill_tool import SkillTool
    from agentica.tools.self_manage_tool import SelfManageTool, CLI_RESTART_HINT
    from agentica.tools.cron_tool import CronTool, CLI_DAEMON_HINT

    # Immediate-run executor for the cronjob tool: builds a fresh CLI agent per
    # run (mirrors the `/cron run` command) so `action='run'` is a real trial run
    # returning output, not just a "mark due" that silently needs the daemon.
    cron_job_runner = None
    if enable_cron_immediate_run:

        def cron_job_runner(job):
            import asyncio
            from agentica.cron.scheduler import _execute_job
            from agentica.cron.cli_runner import CliAgentRunner, build_cli_agent_factory

            factory = build_cli_agent_factory(agent_config, extra_tools, workspace, skills_registry)
            runner = CliAgentRunner(factory)
            return asyncio.run(_execute_job(job, agent_runner=runner, verbose=False))

    # Always give the CLI agent the self-management + cron tools so it can
    # inspect/optimize its own config (config.yaml / .env), self-upgrade, and
    # schedule/manage its own recurring tasks by natural language. Prepended so a
    # user-supplied extra tool with the same name could still override.
    cli_tools = [
        SelfManageTool(restart_hint=CLI_RESTART_HINT),
        CronTool(job_runner=cron_job_runner, daemon_hint=CLI_DAEMON_HINT),
    ] + list(extra_tools or [])

    new_agent = DeepAgent(
        model=model,
        auxiliary_model=auxiliary_model,
        task_model=task_model,
        description=(
            "You are DeepAgent, an interactive CLI coding agent running in the "
            "user's terminal. You help with software engineering tasks: reading "
            "and editing files, running commands, and iterating until the task "
            "is done."
        ),
        tools=cli_tools,  # self-management + user-specified extra tools
        work_dir=work_dir,
        workspace=workspace,
        user_id=cli_user_id,
        session_id=agent_config.get("session_id") or _generate_session_id(),
        debug=agent_config["debug"],
        enable_experience_capture=agent_config.get("enable_experience_capture", True),
        experience_config=experience_config,
        long_term_memory_config=long_term_memory_config,
        include_ask_user_question=True,  # CLI is interactive, always enable human-in-the-loop
        ask_user_question_callback=ask_user_question_callback,
        enable_diagnostics=bool(agent_config.get("enable_diagnostics")),
        diagnostics_servers=agent_config.get("diagnostics_servers"),
        permission_mode=permission_mode,
    )

    if skills_registry and len(skills_registry) > 0:
        has_skill_tool = any(isinstance(tool, SkillTool) for tool in (new_agent.tools or []))
        if not has_skill_tool:
            skills_summary = skills_registry.get_skills_summary()
            if skills_summary:
                new_agent.add_session_guidance(skills_summary)

    # Inject a stable self-description (framework / model / tools / skills) so
    # the agent can answer "what model am I / what tools do I have". Built from
    # the live agent + agent_config so _apply_profile can refresh it after a
    # model/profile switch by calling _build_environment_context again.
    new_agent.environment_context = _build_environment_context(new_agent, agent_config)
    return new_agent
