# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI main entry point
"""

import sys
import time

from agentica.cli.runtime import get_console, parse_args, configure_tools, create_agent
from agentica.cli.display import display_agent_execution_error, format_session_summary, resumable_session_id
from agentica.cli.setup import resolve_model_config, run_onboarding
from agentica.cost_tracker import refresh_model_catalog_in_background
from agentica.run_response import AgentCancelledError
from agentica.utils.log import suppress_console_logging, enable_process_file_logging
from agentica.workspace import Workspace
from agentica.skills import load_system_skills, get_skill_registry


def _enable_cli_file_logging() -> str:
    """Attach a file handler for this CLI session. See ``enable_process_file_logging``."""
    return enable_process_file_logging()


def main():
    args = parse_args()

    # Enable a default file log for the CLI (SDK stays silent by default).
    _enable_cli_file_logging()

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

    # Keep local catalog reads on the startup path and refresh stale pricing /
    # capability metadata in parallel. The daemon has a hard network timeout
    # and never delays the first prompt.
    refresh_model_catalog_in_background()

    # Resolve provider/model/base_url: CLI args > saved config > defaults.
    # Triggers the first-run wizard when no key/config is present on a TTY.
    resolved = resolve_model_config(args, console=get_console())

    # Store agent configuration parameters
    agent_config = {
        # Which saved profile this session is on (empty when a flag replaced
        # the model, so no profile describes it). Read via setup.session_profile.
        "profile_name": resolved.get("profile_name"),
        "profile_source": resolved.get("profile_source"),
        "model_provider": resolved["model_provider"],
        "model_name": resolved["model_name"],
        "base_url": resolved["base_url"],
        "wire_api": resolved.get("wire_api"),
        # CLI flag wins; otherwise use the key stored in a config.yaml profile
        # for the resolved provider/base_url. If both are None the model factory
        # falls back to the provider's env var (backwards-compat).
        "api_key": args.api_key or resolved.get("api_key"),
        # Model tuning params: CLI flag wins, else the active profile's value
        # (resolved from config.yaml), else None (model/factory default).
        "max_tokens": args.max_tokens if args.max_tokens is not None else resolved.get("max_tokens"),
        "temperature": args.temperature if args.temperature is not None else resolved.get("temperature"),
        "reasoning_effort": args.reasoning_effort or resolved.get("reasoning_effort"),
        # Responses reasoning maps to reasoning: {effort: ...}. ``wire_api``
        # selects the protocol independently, so Responses also works without
        # a reasoning override.
        "reasoning": resolved.get("reasoning"),
        "top_p": args.top_p if args.top_p is not None else resolved.get("top_p"),
        "context_window": args.context_window if args.context_window is not None else resolved.get("context_window"),
        "compact_token_limit": args.compact_token_limit if args.compact_token_limit is not None else resolved.get("compact_token_limit"),
        # Raw passthrough dicts (profile-only, no CLI flag — see cli/setup.py's
        # _RAW_PASSTHROUGH_KEYS) for endpoints whose tuning knobs don't map to
        # a standard OpenAI param (e.g. Hunyuan's taiji gateway).
        "extra_body": resolved.get("extra_body"),
        "extra_headers": resolved.get("extra_headers"),
        # Prompt caching (OpenAI-compatible proxies fronting Claude).
        # CLI flag wins; otherwise the active profile's value; else None (auto:
        # on for known proxy base_urls, off otherwise).
        "enable_cache_control": args.enable_cache_control if args.enable_cache_control is not None else resolved.get("enable_cache_control"),
        "cache_control_messages": args.cache_control_messages if args.cache_control_messages is not None else resolved.get("cache_control_messages"),
        "cache_control_session_header": args.cache_control_session_header or resolved.get("cache_control_session_header"),
        # Auxiliary model (None means reuse main model). CLI flags win (applied
        # inside resolve_model_config); otherwise the active profile's optional
        # ``auxiliary_model`` block is used; else None. The auxiliary model drives all
        # background LLM work AND the `task` subagent tool.
        "auxiliary_model_provider": resolved.get("auxiliary_model_provider"),
        "auxiliary_model_name": resolved.get("auxiliary_model_name"),
        "auxiliary_base_url": resolved.get("auxiliary_base_url"),
        "auxiliary_api_key": resolved.get("auxiliary_api_key"),
        "auxiliary_wire_api": resolved.get("auxiliary_wire_api"),
        "auxiliary_extra_body": resolved.get("auxiliary_extra_body"),
        "auxiliary_extra_headers": resolved.get("auxiliary_extra_headers"),
        "auxiliary_reasoning": resolved.get("auxiliary_reasoning"),
        "auxiliary_reasoning_effort": resolved.get("auxiliary_reasoning_effort"),
        # Cross-provider fallback chain (resilience) + per-model API attempts.
        # Both are hand-edited in config.yaml (no CLI flags / wizard prompt).
        # Default max_api_retry=2 so the main model retries once before the
        # fallback chain takes over (SDK default is 1 = no same-model retry).
        "fallback_models": resolved.get("fallback_models") or [],
        "max_api_retry": (
            resolved.get("max_api_retry")
            if resolved.get("max_api_retry") is not None
            else 2
        ),
        "debug": args.debug > 0,
        "work_dir": args.work_dir,
        # Resolved by the interactive app after `resume` has settled the
        # directory, because both compete to decide where this session works.
        "worktree": args.worktree,
        "enable_experience_capture": not args.no_experience,
        "enable_evict": args.evict,
        "enable_auto_compact": args.auto_compact,
        "enable_skill_upgrade": args.enable_skill_upgrade,
        "skill_upgrade_mode": args.skill_upgrade_mode,
        "permissions": "allow-all" if args.allow_all else args.permissions,
        "enable_diagnostics": args.enable_diagnostics,
        "diagnostics_servers": args.diagnostics_servers,
        "_model_config_explicit": any(
            value is not None
            for value in (
                args.profile,
                args.model_provider,
                args.model_name,
                args.base_url,
                args.api_key,
                args.max_tokens,
                args.temperature,
                args.reasoning_effort,
                args.top_p,
                args.context_window,
                args.compact_token_limit,
                args.enable_cache_control,
                args.cache_control_messages,
                args.cache_control_session_header,
                args.auxiliary_model_provider,
                args.auxiliary_model_name,
                args.auxiliary_base_url,
                args.auxiliary_api_key,
            )
        ),
    }
    if getattr(args, "command", None) == "resume":
        agent_config["session_id"] = args.resume_session_id
        agent_config["_resume_at_uuid"] = args.resume_at_uuid
        agent_config["_resume_requested"] = True
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
        load_system_skills()
        skills_registry = get_skill_registry()

    if args.query:
        # Non-interactive mode
        con = get_console()
        # --print means the caller wants the answer and nothing else: another
        # agentica session delegating work, or a shell pipeline. Anything on
        # stdout that is not the answer is noise there, including log lines.
        if args.print_only:
            suppress_console_logging()
        else:
            con.print(f"Running query: {args.query}", style="cyan")
            tools_info = f", Extra Tools: {', '.join(extra_tool_names)}" if extra_tool_names else ""
            con.print(
                f"Model: {agent_config['model_provider']}/{agent_config['model_name']}{tools_info}",
                style="magenta",
            )

        extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
        agent_instance = create_agent(agent_config, extra_tools, workspace, skills_registry)
        started_at = time.monotonic()
        try:
            response = agent_instance.run_stream_sync(args.query)
            for chunk in response:
                if chunk and chunk.content:
                    if args.print_only:
                        # Raw write: the answer may contain [brackets] that
                        # rich would read as markup and swallow.
                        sys.stdout.write(chunk.content)
                    else:
                        con.print(chunk.content, end="")
            if args.print_only:
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                con.print()  # final newline
        except (KeyboardInterrupt, AgentCancelledError):
            con.print("\n[yellow]Interrupted.[/yellow]")
            assert agent_instance.model is not None
            con.print(
                format_session_summary(
                    elapsed_seconds=time.monotonic() - started_at,
                    usage=agent_instance.model.usage,
                    session_id=resumable_session_id(agent_instance),
                )
            )
            sys.exit(130)
        except Exception as e:
            display_agent_execution_error(con, e)
            # A one-shot run that failed must say so in its exit status: the
            # caller is a script or a delegating session, and both decide what
            # to do next from the return code.
            sys.exit(1)
        finally:
            # Same Langfuse atexit block as the interactive exit: bound it so
            # a `agentica "query"` subprocess handed to a script/parent agent
            # doesn't dangle for seconds after its last token.
            try:
                from agentica.utils.langfuse_integration import shutdown_langfuse_bounded
                shutdown_langfuse_bounded(timeout=0.8)
            except Exception:
                pass
    else:
        # Interactive mode
        from agentica.cli.interactive import run_interactive

        run_interactive(agent_config, extra_tool_names, workspace, skills_registry)


if __name__ == "__main__":
    main()
