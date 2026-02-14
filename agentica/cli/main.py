# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI main entry point
"""
from agentica.cli.config import console, parse_args, configure_tools, create_agent
from agentica.cli.interactive import run_interactive
from agentica.utils.log import suppress_console_logging
from agentica.workspace import Workspace
from agentica.skills import load_skills, get_skill_registry


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
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "debug": args.verbose > 0,
        "work_dir": args.work_dir,
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
        console.print(
            f"Model: {agent_config['model_provider']}/{agent_config['model_name']}{tools_info}",
            style="magenta")

        extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
        agent_instance = create_agent(agent_config, extra_tools, workspace, skills_registry)
        try:
            response = agent_instance.run_stream_sync(args.query)
            for chunk in response:
                if chunk and chunk.content:
                    console.print(chunk.content, end="")
            console.print()  # final newline
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
    else:
        # Interactive mode
        run_interactive(agent_config, extra_tool_names, workspace, skills_registry)


if __name__ == "__main__":
    main()
