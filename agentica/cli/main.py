# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI main entry point
"""

from agentica.cli.config import get_console, parse_args, configure_tools, create_agent
from agentica.cli.interactive import run_interactive
from agentica.cli.setup import resolve_model_config, run_onboarding
from agentica.utils.log import suppress_console_logging
from agentica.workspace import Workspace
from agentica.skills import load_skills, get_skill_registry


def main():
    args = parse_args()

    if getattr(args, "chat_only", False):
        from agentica.utils.log import logger, CHAT_LEVEL

        logger.setLevel(CHAT_LEVEL)
        for h in logger.handlers:
            h.setLevel(CHAT_LEVEL)

    # Handle ACP mode for IDE integration
    if args is None or (hasattr(args, "command") and args.command == "acp"):
        from agentica.acp.server import ACPServer

        server = ACPServer()
        server.run()
        return

    if hasattr(args, "command") and args.command in ("skills", "extensions"):
        from agentica.cli.extensions import run_extensions_command

        run_extensions_command(args)
        return

    # `agentica setup` — re-run the onboarding wizard and exit.
    if hasattr(args, "command") and args.command == "setup":
        run_onboarding(get_console())
        return

    # `agentica doctor` — run the environment health check and exit.
    if hasattr(args, "command") and args.command == "doctor":
        from agentica.cli.doctor_display import show_doctor

        show_doctor(
            get_console(),
            enable_diagnostics=args.enable_diagnostics,
            diagnostics_servers=args.diagnostics_servers,
            work_dir=args.work_dir,
        )
        return

    # `agentica cron daemon` — run the standalone cron scheduler in foreground.
    if hasattr(args, "command") and args.command == "cron":
        from agentica.cli.cron_daemon import run_cron_daemon

        run_cron_daemon(args, get_console())
        return

    # Resolve provider/model/base_url: CLI args > saved config > defaults.
    # Triggers the first-run wizard when no key/config is present on a TTY.
    resolved = resolve_model_config(args, console=get_console())

    # Store agent configuration parameters
    agent_config = {
        "model_provider": resolved["model_provider"],
        "model_name": resolved["model_name"],
        "base_url": resolved["base_url"],
        # CLI flag wins; otherwise use the key stored in a config.yaml profile
        # for the resolved provider/base_url. If both are None the model factory
        # falls back to the provider's env var (backwards-compat).
        "api_key": args.api_key or resolved.get("api_key"),
        # Model tuning params: CLI flag wins, else the active profile's value
        # (resolved from config.yaml), else None (model/factory default).
        "max_tokens": args.max_tokens if args.max_tokens is not None else resolved.get("max_tokens"),
        "temperature": args.temperature if args.temperature is not None else resolved.get("temperature"),
        "reasoning_effort": args.reasoning_effort or resolved.get("reasoning_effort"),
        "top_p": args.top_p if args.top_p is not None else resolved.get("top_p"),
        "context_window": args.context_window if args.context_window is not None else resolved.get("context_window"),
        # Prompt caching (OpenAI-compatible proxies fronting Claude, e.g. Venus).
        # CLI flag wins; otherwise the active profile's value; else None (auto:
        # on for known proxy base_urls, off otherwise).
        "enable_cache_control": args.enable_cache_control if args.enable_cache_control is not None else resolved.get("enable_cache_control"),
        "cache_control_messages": args.cache_control_messages if args.cache_control_messages is not None else resolved.get("cache_control_messages"),
        "cache_control_session_header": args.cache_control_session_header or resolved.get("cache_control_session_header"),
        # Aux model (None means reuse main model). CLI flags win (applied
        # inside resolve_model_config); otherwise the active profile's optional
        # ``aux_model`` block is used; else None. The aux model drives all
        # background LLM work AND the `task` subagent tool.
        "aux_model_provider": resolved.get("aux_model_provider"),
        "aux_model_name": resolved.get("aux_model_name"),
        "aux_base_url": resolved.get("aux_base_url"),
        "aux_api_key": resolved.get("aux_api_key"),
        "debug": args.debug > 0,
        "work_dir": args.work_dir,
        "enable_experience_capture": not args.no_experience,
        "sync_memories_to_global_agent_md": args.sync_memories_to_global_agent_md,
        "sync_experience_to_global_agent_md": args.sync_experience_to_global_agent_md,
        "enable_skill_upgrade": args.enable_skill_upgrade,
        "skill_upgrade_mode": args.skill_upgrade_mode,
        "permissions": "allow-all" if args.allow_all else args.permissions,
        "enable_diagnostics": args.enable_diagnostics,
        "diagnostics_servers": args.diagnostics_servers,
    }
    extra_tool_names = list(args.tools) if args.tools else None

    # Initialize workspace with default user
    workspace = None
    if not args.no_workspace:
        workspace_path = args.workspace  # Can be None for default
        workspace = Workspace(workspace_path, user_id="default")
        if not workspace.exists():
            workspace.initialize()
        else:
            # Ensure user directory exists
            workspace._initialize_user_dir()

    # Load skills only if explicitly enabled
    skills_registry = None
    if args.enable_skills:
        # Suppress logging during skill loading for cleaner output
        if args.debug == 0:
            suppress_console_logging()
        load_skills()
        skills_registry = get_skill_registry()

    if args.query:
        # Non-interactive mode
        con = get_console()
        con.print(f"Running query: {args.query}", style="cyan")
        tools_info = f", Extra Tools: {', '.join(extra_tool_names)}" if extra_tool_names else ""
        con.print(f"Model: {agent_config['model_provider']}/{agent_config['model_name']}{tools_info}", style="magenta")

        extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
        agent_instance = create_agent(agent_config, extra_tools, workspace, skills_registry)
        try:
            response = agent_instance.run_stream_sync(args.query)
            for chunk in response:
                if chunk and chunk.content:
                    con.print(chunk.content, end="")
            con.print()  # final newline
        except KeyboardInterrupt:
            con.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            con.print(f"\n[bold red]Error: {str(e)}[/bold red]")
    else:
        # Interactive mode
        run_interactive(agent_config, extra_tool_names, workspace, skills_registry)


if __name__ == "__main__":
    main()
