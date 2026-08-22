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
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console

from agentica.agent.config import (
    ExperienceConfig,
    SkillUpgradeConfig,
    ToolConfig,
    WorkspaceMemoryConfig,
)
from agentica.config import AGENTICA_CACHE_DIR
from agentica.global_config import get_setting
from agentica.compression.manager import parse_compact_token_limit
from agentica.skills import load_system_skills
from agentica.tools.base import Tool
from agentica.utils.log import logger
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
    "read_file",
    "write_file",
    "apply_patch",
    "glob",
    "grep",
    "execute",
    "web_search",
    "fetch_url",
    "task",
]

# Tool icons for CLI display
TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "apply_patch": "✎",
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
    "wikipedia": ("wikipedia", "WikipediaTool", "Search", "Search and read Wikipedia articles"),
    "zhipu_web_search": ("zhipu_web_search", "ZhipuWebSearchTool", "Search", "Web search via ZhipuAI"),
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

# Private model import registry. CLI startup reads only these strings; the
# public MODEL_REGISTRY remains a mapping of provider names to callables and is
# resolved only when a caller explicitly imports it.
_MODEL_IMPORTS = {
    "openai": ("agentica.model.openai.chat", "OpenAIChat"),
    "azure": ("agentica.model.azure.openai_chat", "AzureOpenAIChat"),
    "moonshot": ("agentica", "MoonshotChat"),
    "zhipuai": ("agentica", "ZhipuAIChat"),
    "deepseek": ("agentica", "DeepSeekChat"),
    "yi": ("agentica", "YiChat"),
    "ark": ("agentica", "ArkChat"),
    "anthropic": ("agentica.model.anthropic.claude", "Claude"),
}

_MODEL_REGISTRY_CACHE: Optional[Dict[str, Callable]] = None

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


def _load_symbol(module_path: str, attr_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _load_model_factory(model_provider: str):
    target = _MODEL_IMPORTS.get(model_provider)
    if target is None:
        return None
    module_path, attr_name = target
    return _load_symbol(module_path, attr_name)


def _get_public_model_registry() -> Dict[str, Callable]:
    """Resolve the backward-compatible public provider-to-callable mapping."""
    global _MODEL_REGISTRY_CACHE
    if _MODEL_REGISTRY_CACHE is None:
        _MODEL_REGISTRY_CACHE = {
            provider: _load_symbol(module_path, attr_name)
            for provider, (module_path, attr_name) in _MODEL_IMPORTS.items()
        }
    return _MODEL_REGISTRY_CACHE


def __getattr__(name: str):
    if name == "MODEL_REGISTRY":
        return _get_public_model_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"MODEL_REGISTRY"})


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
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="With --query: write only the agent's final answer to stdout, no banner "
        "or styling. For scripts and for one agentica session delegating to another.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Run on a saved config.yaml profile instead of the active one, for this "
        "session only (nothing is written). The way to start a session on a "
        "different provider; --model_name alone cannot leave the current endpoint.",
    )
    # Default is None so saved CLI config (from the first-run wizard) can take
    # effect; main.py resolves args > saved config > hardcoded default.
    parser.add_argument(
        "--model_provider", type=str, choices=list(_MODEL_IMPORTS), help="LLM model provider", default=None
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
        "--compact-token-limit",
        type=int,
        dest="compact_token_limit",
        help="Working token cap for auto-compression. Layer 2 fires at "
        "min(this, 95%% of context_window). Does not change the model's "
        "context_window. Profile compact_token_limit / settings.compact_token_limit "
        "also apply.",
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
        choices=list(_MODEL_IMPORTS),
        help="Provider for the auxiliary model (defaults to --model_provider)",
    )
    parser.add_argument(
        "--auxiliary_model_name",
        type=str,
        help="Model id for the auxiliary model (background tasks + `task` subagent; required to enable a separate auxiliary)",
    )
    parser.add_argument("--auxiliary_base_url", type=str, help="Base URL for the auxiliary model")
    parser.add_argument("--auxiliary_api_key", type=str, help="API key for the auxiliary model")

    # Prompt caching for OpenAI-compatible proxies that front Anthropic Claude.
    # Default None = use the active profile's value (or off if the
    # profile doesn't set it); --enable_cache_control / --no-enable_cache_control
    # force on/off for this run.
    parser.add_argument(
        "--enable_cache_control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Anthropic-style cache_control blocks (for OpenAI-compatible proxies).",
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
        help="Sticky-routing header name for cache hits (e.g. X-Session-Id).",
    )

    parser.add_argument("--debug", type=int, help="enable verbose mode", default=0)
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Show only inter-agent CHAT messages (suppress INFO/DEBUG/WARNING)",
    )
    parser.add_argument("--work_dir", type=str, help="Working directory for file operations", default=None)
    parser.add_argument(
        "--worktree",
        type=str,
        default=None,
        help=(
            "Work in a per-task git worktree of this repository "
            "(.agentica/worktrees/<name>, branch wt/<name>). Created on first "
            "use; merge lands on local main and removes it."
        ),
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        choices=list(TOOL_REGISTRY.keys()),
        help="Additional tools to enable (on top of built-in tools)",
    )
    parser.add_argument(
        "--no-experience", action="store_true", help="Disable DeepAgent experience capture and self-evolution hooks"
    )
    parser.add_argument(
        "--evict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Layer 1: evict old tool results under context pressure "
        "(default: on; --no-evict disables). config.yaml settings.enable_evict also applies.",
    )
    parser.add_argument(
        "--auto-compact",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="auto_compact",
        help="Layer 2: auto-summarise when the context window fills "
        "(default: on; --no-auto-compact disables). /compact still works. "
        "config.yaml settings.enable_auto_compact also applies.",
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
        help="Enable edit-time LSP diagnostics for built-in file tools (default: on; use --no-enable-diagnostics to disable). Needs a working language server, e.g. pip install 'pyright[nodejs]'; startup degrades if initialize fails.",
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
        "command",
        nargs="?",
        choices=["acp", "resume"],
        help="Run in ACP mode or resume a prior CLI session",
    )
    parser.add_argument(
        "resume_session_id",
        nargs="?",
        help="Session id for `agentica resume <session-id>`",
    )
    parser.add_argument(
        "--resume-at",
        dest="resume_at_uuid",
        default=None,
        help="Resume the session at a specific message UUID",
    )
    args = parser.parse_args()
    if args.command == "resume" and not args.resume_session_id:
        parser.error("agentica resume requires a session id")
    if args.command != "resume" and args.resume_session_id is not None:
        parser.error("a session id is only valid with `agentica resume`")
    return args


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
    wire_api=None,
    max_tokens=None,
    temperature=None,
    reasoning_effort=None,
    reasoning=None,
    top_p=None,
    context_window=None,
    enable_cache_control=None,
    cache_control_messages=None,
    cache_control_session_header=None,
    cache_keepalive=None,
    extra_body=None,
    extra_headers=None,
    default_headers=None,
):
    """Create a model instance based on the provider name.

    Uses the private lazy import registry instead of if/elif chains. Provider
    SDK modules are imported only when a model instance is actually built.
    """
    effective_wire_api = wire_api or "chat_completions"
    if effective_wire_api not in ("chat_completions", "responses"):
        raise ValueError("wire_api must be either 'chat_completions' or 'responses'.")
    if wire_api is not None and model_provider != "openai":
        raise ValueError("The 'wire_api' config field requires model_provider: openai.")
    if reasoning is not None and effective_wire_api != "responses":
        raise ValueError("The 'reasoning' config field requires wire_api: responses.")
    if effective_wire_api == "responses" and reasoning_effort is not None:
        raise ValueError(
            "Responses API uses 'reasoning', not Chat Completions' 'reasoning_effort'."
        )
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
    # proxy that forwards the native /v1/messages endpoint can be targeted,
    # and the Claude client seeds the bearer header for such proxies.
    if base_url is not None:
        params["base_url"] = base_url
    # OpenAI-only tuning: reasoning_effort + raw passthrough dicts. Anthropic
    # takes reasoning_effort too (mapped to adaptive thinking inside the Claude
    # model class), but NOT the OpenAI extra_body/extra_headers passthrough.
    # default_headers goes to the Anthropic client's static headers — the only
    # way to pin sticky routing (e.g. X-Sticky-Routing) on the native
    # /v1/messages path, which has no per-request extra_headers mechanism.
    if model_provider == "anthropic":
        is_claude_opus_5 = _load_symbol("agentica.model.anthropic.claude", "is_claude_opus_5")
        effective_effort = reasoning_effort
        if is_claude_opus_5(model_name):
            if effective_effort is None:
                effective_effort = "high"
            elif effective_effort == "off":
                effective_effort = None
        if effective_effort is not None:
            params["reasoning_effort"] = effective_effort
        if default_headers is not None:
            params["default_headers"] = default_headers
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

    if model_provider == "openai" and effective_wire_api == "responses":
        model_class = _load_symbol("agentica.model.openai.responses", "OpenAIResponses")
    else:
        model_class = _load_model_factory(model_provider)
    if model_class is None:
        raise ValueError(f"Unsupported model provider: {model_provider}. Supported: {', '.join(_MODEL_IMPORTS)}")
    if reasoning is not None:
        params["reasoning"] = reasoning
    # Prompt caching. ``enable_cache_control`` applies to any model class that
    # declares it (OpenAIChat for OpenAI-compatible proxies fronting Claude,
    # Claude itself for native Anthropic caching) — filling it in CLI/config
    # takes effect everywhere. The OpenAIChat-only message/header knobs are not
    # passed to Claude, which manages its own message caching natively.
    if inspect.isclass(model_class):
        model_fields = model_class.__dataclass_fields__
        if enable_cache_control is not None and "enable_cache_control" in model_fields:
            params["enable_cache_control"] = enable_cache_control
        if effective_wire_api == "chat_completions" and "cache_control_messages" in model_fields:
            if cache_control_messages is not None:
                params["cache_control_messages"] = cache_control_messages
        if effective_wire_api == "chat_completions" and "cache_control_session_header" in model_fields:
            if cache_control_session_header is not None:
                params["cache_control_session_header"] = cache_control_session_header
        if effective_wire_api == "chat_completions" and "cache_keepalive" in model_fields:
            if cache_keepalive is not None:
                params["cache_keepalive"] = cache_keepalive
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
        wire_api=agent_config.get(f"{prefix}_wire_api"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
        reasoning_effort=agent_config.get(f"{prefix}_reasoning_effort"),
        reasoning=agent_config.get(f"{prefix}_reasoning"),
        top_p=agent_config.get("top_p"),
        context_window=agent_config.get("context_window"),
        enable_cache_control=agent_config.get("enable_cache_control"),
        cache_control_messages=agent_config.get("cache_control_messages"),
        cache_control_session_header=agent_config.get("cache_control_session_header"),
        cache_keepalive=agent_config.get("cache_keepalive"),
        # Auxiliary passthrough dicts are their own field (auxiliary_extra_*),
        # never inherited from the main model even when same provider — a
        # different deployment/endpoint may not want the same raw params.
        extra_body=agent_config.get(f"{prefix}_extra_body"),
        extra_headers=agent_config.get(f"{prefix}_extra_headers"),
    )


def _build_fallback_models(agent_config: dict) -> List[Any]:
    """Build cross-provider fallback model instances from resolved flat dicts.

    ``agent_config["fallback_models"]`` is a list of flat dicts already resolved
    by ``cli/setup.py`` — each carries ``model_provider``/``model_name``/
    ``base_url``/``api_key`` plus optional tuning. Returns [] when no fallback
    is configured. Entries missing a model_name are skipped (defensive; the
    setup validation already rejects them, but a hand-built agent_config may
    slip through).
    """
    models: List[Any] = []
    for fb in agent_config.get("fallback_models") or []:
        if not isinstance(fb, dict) or not fb.get("model_name"):
            continue
        models.append(
            get_model(
                model_provider=fb["model_provider"],
                model_name=fb["model_name"],
                base_url=fb.get("base_url"),
                api_key=fb.get("api_key"),
                wire_api=fb.get("wire_api"),
                reasoning=fb.get("reasoning"),
                reasoning_effort=fb.get("reasoning_effort"),
                extra_body=fb.get("extra_body"),
                extra_headers=fb.get("extra_headers"),
            )
        )
    return models


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

    fallback_models = agent_config.get("fallback_models") or []
    if fallback_models:
        fb_ids = ", ".join(
            f"{fb.get('model_provider', '?')}/{fb.get('model_name', '?')}"
            for fb in fallback_models if isinstance(fb, dict)
        )
        if fb_ids:
            lines.append(f"- Fallback models: {fb_ids}")

    tool_names = sorted(
        name for t in (agent.tools or []) if isinstance(t, Tool) and t.functions for name in t.functions.keys()
    )
    lines.append(f"- Active tools: {', '.join(tool_names) if tool_names else 'none'}")

    skill_names = _get_active_skill_names(agent)
    if skill_names:
        lines.append(f"- Active skills: {', '.join(sorted(skill_names))}")

    from agentica.subagent import get_subagent_configs

    subagent_types = sorted(get_subagent_configs())
    lines.append(f"- Subagent types: {', '.join(subagent_types)}")
    lines.append("- Slash commands: /rename /resume /status /model /tools /skills /agents /config /usage /permissions /help /exit")
    lines.append("- To extend: /skills install <name>, /agents create <name>")

    return "\n".join(lines)


def _resolve_compression_flags(agent_config: dict) -> tuple[bool, bool]:
    """CLI flag (via agent_config) wins; else config.yaml settings; else on."""
    evict = agent_config.get("enable_evict")
    if evict is None:
        evict = get_setting("enable_evict", True)
    auto = agent_config.get("enable_auto_compact")
    if auto is None:
        auto = get_setting("enable_auto_compact", True)
    return bool(evict), bool(auto)


def _resolve_compact_token_limit(agent_config: dict) -> Optional[int]:
    """CLI flag / profile wins; else config.yaml settings; else unset."""
    cap = parse_compact_token_limit(agent_config.get("compact_token_limit"))
    if cap is not None:
        return cap
    return parse_compact_token_limit(get_setting("compact_token_limit", None))


def create_agent(
    agent_config: dict,
    extra_tools: Optional[List] = None,
    workspace: Optional[Workspace] = None,
    skills_registry=None,
    ask_user_question_callback=None,
    background_process_registry=None,
    enable_cron_immediate_run: bool = True,
    permission_mode: Optional[str] = None,
    peer_session=None,
    worktree_binder=None,
):
    """Helper to create or recreate an Agent with built-in tools and current config.

    ask_user_question_callback: optional ``(prompt, options) -> str`` used by the
        ask_user_question/confirm tools. The interactive CLI passes a prompt_toolkit-aware
        callback so the tool reads via the TUI input box instead of a bare
        ``input()`` (which deadlocks against prompt_toolkit's stdin ownership in
        the background agent thread).
    background_process_registry: optional shared registry used by
        execute(background=True), /ps, /stop, and the status bar.
    enable_cron_immediate_run: when True (interactive CLI) the ``cronjob`` tool's
        ``action='run'`` executes a job once immediately and returns its output.
        Set False for cron-spawned agents so a scheduled job cannot recursively
        trigger further immediate runs.
    permission_mode: unified 3-tier tool permission ("ask"/"auto"/"allow-all",
        see agentica.agent.permissions). Falls back to ``agent_config["permissions"]``,
        then "allow-all" (the CLI's actual --permissions default; see parse_args()).
    peer_session: optional ``agentica.peers.PeerSession`` for cross-session
        messaging. When given, the agent gets the ``list_agents`` /
        ``send_message`` tools and its inbox is drained between tool batches.
    worktree_binder: optional ``agentica.cli.worktree_binding.WorktreeBinder``.
        When given, the agent gets the ``worktree`` tool and can move this
        session into a per-task checkout on its own — including when the
        instruction arrived from another session as a peer message.
    """
    if permission_mode is None:
        configured_permission_mode = agent_config.get("permissions")
        permission_mode = configured_permission_mode if isinstance(configured_permission_mode, str) else "allow-all"
    enable_evict, enable_auto_compact = _resolve_compression_flags(agent_config)
    compact_token_limit = _resolve_compact_token_limit(agent_config)
    load_system_skills()
    model = get_model(
        model_provider=agent_config["model_provider"],
        model_name=agent_config["model_name"],
        base_url=agent_config.get("base_url"),
        api_key=agent_config.get("api_key"),
        wire_api=agent_config.get("wire_api"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
        reasoning_effort=agent_config.get("reasoning_effort"),
        reasoning=agent_config.get("reasoning"),
        top_p=agent_config.get("top_p"),
        context_window=agent_config.get("context_window"),
        enable_cache_control=agent_config.get("enable_cache_control"),
        cache_control_messages=agent_config.get("cache_control_messages"),
        cache_control_session_header=agent_config.get("cache_control_session_header"),
        cache_keepalive=agent_config.get("cache_keepalive"),
        extra_body=agent_config.get("extra_body"),
        extra_headers=agent_config.get("extra_headers"),
        default_headers=agent_config.get("default_headers"),
    )

    # Auxiliary model: the cheap/fast model for all background LLM work (memory
    # extraction, compression, classification, goal judging, skill upgrade) AND
    # the `task` subagent tool. When --auxiliary_model_name is unset this stays None
    # and DeepAgent falls back to the main model for auxiliary work.
    auxiliary_model = _build_sibling_model(agent_config, "auxiliary")
    # The task subagent tool shares the auxiliary model (one cheap model for all
    # non-user-facing LLM work), so the CLI exposes only main + auxiliary.
    task_model = auxiliary_model
    # Cross-provider fallback chain (resilience): built from the resolved flat
    # dicts in agent_config["fallback_models"]. [] when none configured.
    fallback_models = _build_fallback_models(agent_config)
    # Runner-level API attempts per model. CLI default is 2 (main model retries
    # once before the fallback chain takes over); the SDK default is 1.
    max_api_retry = agent_config.get("max_api_retry")
    if max_api_retry is None:
        max_api_retry = 2
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
    if background_process_registry is not None:
        background_process_registry.set_user_id(cli_user_id)

    # Branching must not append the new line of work to the transcript it came
    # from, or the two branches share one file and neither can be resumed on its
    # own. Both entry points land here: `/fork` (whole log) and `/fork <n>` /
    # `resume <id> at <uuid>` (truncated). Popped rather than read so rebuilding
    # the agent later (a `/model` switch) does not fork again from the same point.
    session_id = agent_config.get("session_id") or _generate_session_id()
    # Set when resuming a session that lives in another project directory: the
    # transcript keeps growing where it was written, wherever the agent works.
    session_base_dir = agent_config.get("session_base_dir")
    fork_at_uuid = agent_config.pop("_resume_at_uuid", None)
    fork_whole_log = agent_config.pop("_fork_session", False)
    if fork_at_uuid or fork_whole_log:
        from agentica.memory.session_log import SessionLog

        source = SessionLog(
            session_id, base_dir=session_base_dir, work_dir=work_dir, user_id=cli_user_id
        )
        if source.exists():
            forked = source.fork(_generate_session_id(), at_uuid=fork_at_uuid)
            logger.info(
                f"Forked session {session_id} -> {forked.session_id}"
                f"{f' at {fork_at_uuid}' if fork_at_uuid else ' (whole log)'}"
            )
            session_id = forked.session_id
            agent_config["session_id"] = session_id

    # Use DeepAgent for full-featured CLI experience.
    from agentica.agent.deep import DeepAgent
    from agentica.tools.skill_tool import SkillTool
    from agentica.tools.self_manage_tool import SelfManageTool, CLI_RESTART_HINT
    from agentica.tools.cron_tool import CronTool, CLI_DAEMON_HINT

    # Immediate-run executor for the cronjob tool: builds a fresh CLI agent per
    # run (mirrors the `/cron run` command) so `action='run'` is a real trial run
    # returning output, not just a "mark due" that silently needs the daemon.
    cron_job_runner: Optional[Callable[[Any], Dict[str, Any]]] = None
    if enable_cron_immediate_run:

        def _cron_job_runner(job):
            import asyncio
            from agentica.cron.scheduler import _execute_job
            from agentica.cron.cli_runner import CliAgentRunner, build_cli_agent_factory

            factory = build_cli_agent_factory(agent_config, extra_tools, workspace, skills_registry)
            runner = CliAgentRunner(factory)
            return asyncio.run(_execute_job(job, agent_runner=runner, verbose=False))

        cron_job_runner = _cron_job_runner

    # Always give the CLI agent the self-management + cron tools so it can
    # inspect/optimize its own config (config.yaml / .env), self-upgrade, and
    # schedule/manage its own recurring tasks by natural language. Prepended so a
    # user-supplied extra tool with the same name could still override.
    cli_tools = [
        SelfManageTool(restart_hint=CLI_RESTART_HINT),
        CronTool(job_runner=cron_job_runner, daemon_hint=CLI_DAEMON_HINT),
    ] + list(extra_tools or [])

    if peer_session is not None:
        from agentica.tools.peer_tool import PeerMessagingTool

        cli_tools.insert(0, PeerMessagingTool(peer_session))

    # Worktrees are the other half of peer messaging: list_agents shows that
    # another session is dirty in the same directory, and this is what the agent
    # does about it without asking a human to restart it somewhere else.
    if worktree_binder is not None:
        from agentica.tools.worktree_tool import WorktreeTool

        cli_tools.insert(0, WorktreeTool(worktree_binder))

    from agentica.peer_conflicts import build_checker as build_peer_conflict_checker

    # Delegating needs the session's process registry (that is how the worker is
    # tracked, waited on and reported), so a one-shot `--query` run and a
    # cron-spawned agent — neither of which has one — simply do not get the tool.
    # Nor does a worker that was itself delegated: MAX_DEPTH stops a tree of
    # agents spawning agents with nobody watching the bill.
    # The tool itself is registered inside DeepAgent (the worker needs the model
    # object's credentials, and config.yaml may not have any); the removal below
    # runs after the agent is built.

    new_agent = DeepAgent(
        model=model,
        auxiliary_model=auxiliary_model,
        task_model=task_model,
        fallback_models=fallback_models,
        max_api_retry=max_api_retry,
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
        session_id=session_id,
        session_base_dir=session_base_dir,
        debug=agent_config["debug"],
        enable_experience_capture=agent_config.get("enable_experience_capture", True),
        experience_config=experience_config,
        long_term_memory_config=long_term_memory_config,
        include_ask_user_question=True,  # CLI is interactive, always enable human-in-the-loop
        ask_user_question_callback=ask_user_question_callback,
        background_process_registry=background_process_registry,
        enable_diagnostics=bool(agent_config.get("enable_diagnostics")),
        diagnostics_servers=agent_config.get("diagnostics_servers"),
        permission_mode=permission_mode,
        tool_config=ToolConfig(
            auto_load_mcp=True,
            permission_mode=permission_mode,
            enable_evict=enable_evict,
            enable_auto_compact=enable_auto_compact,
            compact_token_limit=compact_token_limit,
        ),
        # Tell the model when another live session already has the file it just
        # wrote uncommitted, instead of letting both find out at merge time.
        peer_conflict_checker=build_peer_conflict_checker(peer_session),
    )

    # A one-shot `--query` run and a cron-spawned agent have no process
    # registry, and a worker they started could never be waited on or reported
    # back; MAX_DEPTH likewise stops a worker from delegating further. The tool
    # is registered inside DeepAgent, so here it is only ever removed.
    from agentica.tools.builtin.delegate_tool import (
        BuiltinDelegateTool,
        MAX_DEPTH,
        delegation_depth,
    )

    if background_process_registry is None or delegation_depth() >= MAX_DEPTH:
        new_agent.tools = [
            tool for tool in (new_agent.tools or []) if not isinstance(tool, BuiltinDelegateTool)
        ]

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
    if (
        agent_config.get("profile_name")
        and not agent_config.get("_skip_session_profile_persist")
        and new_agent._session_log is not None
    ):
        new_agent._session_log.set_profile(
            agent_config["profile_name"], agent_config.get("profile_source") or ""
        )
    new_agent.peer_session = peer_session
    return new_agent
