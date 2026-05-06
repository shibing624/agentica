# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI main entry point
"""
from agentica.cli.config import get_console, parse_args, configure_tools, create_agent
from agentica.cli.interactive import run_interactive
from agentica.utils.log import suppress_console_logging
from agentica.workspace import Workspace
from agentica.skills import load_skills, get_skill_registry


def main():
    args = parse_args()

    if getattr(args, 'chat_only', False):
        from agentica.utils.log import logger, CHAT_LEVEL
        logger.setLevel(CHAT_LEVEL)
        for h in logger.handlers:
            h.setLevel(CHAT_LEVEL)

    # Handle ACP mode for IDE integration
    if args is None or (hasattr(args, 'command') and args.command == 'acp'):
        from agentica.acp.server import ACPServer
        server = ACPServer()
        server.run()
        return

    if hasattr(args, "command") and args.command in ("skills", "extensions"):
        from agentica.cli.extensions import run_extensions_command
        run_extensions_command(args)
        return

    # Store agent configuration parameters
    agent_config = {
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        # Auxiliary model (None means reuse main model).
        "aux_model_provider": args.aux_model_provider,
        "aux_model_name": args.aux_model_name,
        "aux_base_url": args.aux_base_url,
        "aux_api_key": args.aux_api_key,
        # Task-subagent model (None means reuse main model).
        "task_model_provider": args.task_model_provider,
        "task_model_name": args.task_model_name,
        "task_base_url": args.task_base_url,
        "task_api_key": args.task_api_key,
        "debug": args.debug > 0,
        "work_dir": args.work_dir,
        "enable_experience_capture": not args.no_experience,
        "sync_memories_to_global_agent_md": args.sync_memories_to_global_agent_md,
        "sync_experience_to_global_agent_md": args.sync_experience_to_global_agent_md,
        "enable_skill_upgrade": args.enable_skill_upgrade,
        "skill_upgrade_mode": args.skill_upgrade_mode,
        "permissions": "allow-all" if args.allow_all else args.permissions,
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
        con.print(
            f"Model: {agent_config['model_provider']}/{agent_config['model_name']}{tools_info}",
            style="magenta")

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
