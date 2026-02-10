# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI interactive mode - main interaction loop
"""
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

from rich.text import Text

from agentica.cli.config import (
    console,
    history_file,
    TOOL_REGISTRY,
    MODEL_REGISTRY,
    EXAMPLE_MODELS,
    configure_tools,
    create_agent,
)
from agentica.cli.display import (
    COLORS,
    StreamDisplayManager,
    print_header,
    show_help,
    parse_file_mentions,
    inject_file_contents,
    display_user_message,
    get_file_completions,
)
from agentica.run_response import AgentCancelledError
from agentica.utils.log import suppress_console_logging
from agentica.workspace import Workspace


# ==================== Command Handlers ====================

def _cmd_help(**kwargs):
    show_help()


def _cmd_exit(**kwargs):
    return "EXIT"


def _cmd_tools(extra_tool_names=None, **kwargs):
    console.print("Available additional tools:", style="cyan")
    for name in sorted(TOOL_REGISTRY.keys()):
        marker = " [active]" if extra_tool_names and name in extra_tool_names else ""
        console.print(f"  - {name}{marker}")
    console.print()
    console.print("Use --tools <name> when starting CLI to enable tools.", style="dim")


def _cmd_skills(skills_registry=None, **kwargs):
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


def _cmd_memory(current_agent=None, **kwargs):
    if hasattr(current_agent, 'memory') and current_agent.memory:
        messages = current_agent.memory.messages if hasattr(current_agent.memory, 'messages') else []
        if messages:
            console.print(f"[bold cyan]Conversation History ({len(messages)} messages)[/bold cyan]")
            console.print()
            for i, msg in enumerate(messages[-20:], 1):
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
                
                role_colors = {'user': '[cyan]User[/cyan]', 'assistant': '[green]Assistant[/green]',
                               'system': '[yellow]System[/yellow]'}
                role_display = role_colors.get(role, f"[dim]{role}[/dim]")
                
                if isinstance(content, str):
                    preview = content[:300] + "..." if len(content) > 300 else content
                elif isinstance(content, list):
                    preview = str(content)[:300] + "..."
                else:
                    preview = str(content)[:300]
                
                console.print(f"  {role_display}: {preview}")
                console.print()
        else:
            console.print("[yellow]No conversation history yet.[/yellow]")
    else:
        console.print("[yellow]No memory available.[/yellow]")


def _cmd_workspace(workspace=None, **kwargs):
    if workspace:
        console.print(f"Workspace: [bold]{workspace.path}[/bold]", style="cyan")
        console.print(f"User: [cyan]{workspace.user_id or 'default'}[/cyan]")
        console.print(f"Exists: {'Yes' if workspace.exists() else 'No'}")
        if workspace.exists():
            files = workspace.list_files()
            console.print("Config Files:", style="cyan")
            for fname, exists in files.items():
                status_icon = "✓" if exists else "✗"
                console.print(f"  {status_icon} {fname}")
            
            memory_files = workspace.get_all_memory_files()
            console.print(f"User Memory:", style="cyan")
            if memory_files:
                for mf in memory_files[:5]:
                    if mf.name == "MEMORY.md":
                        console.print(f"  ✓ {mf.name} [dim](long-term)[/dim]")
                    else:
                        console.print(f"  ✓ {mf.name} [dim](daily)[/dim]")
                if len(memory_files) > 5:
                    console.print(f"  ... and {len(memory_files) - 5} more")
            else:
                console.print("  [dim]No memory files yet[/dim]")
            console.print()
            console.print("[dim]Tip: Long-term memory is saved via Agent's save_memory tool[/dim]")
    else:
        console.print("Workspace not configured.", style="yellow")


def _cmd_newchat(agent_config=None, extra_tools=None, workspace=None, skills_registry=None, **kwargs):
    current_agent = create_agent(agent_config, extra_tools, workspace, skills_registry)
    console.print("[green]New chat session created.[/green]")
    console.print("[dim]Conversation history cleared.[/dim]")
    return {"current_agent": current_agent}


def _cmd_clear(agent_config=None, extra_tools=None, extra_tool_names=None,
               workspace=None, skills_registry=None, shell_mode=False, **kwargs):
    os.system('clear' if os.name != 'nt' else 'cls')
    current_agent = create_agent(agent_config, extra_tools, workspace, skills_registry)
    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        work_dir=agent_config.get("work_dir"),
        extra_tools=extra_tool_names,
        shell_mode=shell_mode
    )
    console.print("[info]Screen cleared and conversation reset.[/info]")
    return {"current_agent": current_agent}


def _cmd_model(cmd_args="", agent_config=None, extra_tools=None,
               workspace=None, skills_registry=None, **kwargs):
    supported_providers = set(MODEL_REGISTRY.keys())
    if cmd_args:
        if "/" in cmd_args:
            new_provider, new_model = cmd_args.split("/", 1)
            new_provider = new_provider.strip().lower()
            new_model = new_model.strip()
        else:
            new_model = cmd_args.strip()
            new_provider = agent_config["model_provider"]
        
        if new_provider not in supported_providers:
            console.print(f"[red]Unknown provider: {new_provider}[/red]")
            console.print(f"Supported: {', '.join(sorted(supported_providers))}", style="dim")
            return
        
        agent_config["model_provider"] = new_provider
        agent_config["model_name"] = new_model
        current_agent = create_agent(agent_config, extra_tools, workspace, skills_registry)
        console.print(f"[green]Switched to: {new_provider}/{new_model}[/green]")
        return {"current_agent": current_agent}
    else:
        console.print(f"Current model: [bold cyan]{agent_config['model_provider']}/{agent_config['model_name']}[/bold cyan]")
        console.print()
        console.print("Supported providers and example models:", style="cyan")
        for provider in sorted(EXAMPLE_MODELS.keys()):
            marker = " [current]" if provider == agent_config["model_provider"] else ""
            models_str = ", ".join(EXAMPLE_MODELS[provider][:3]) + ", ..."
            console.print(f"  {provider}{marker}: [dim]{models_str}[/dim]")
        console.print()
        console.print("Usage: /model <provider>/<model>  (any model name accepted)", style="dim")
        console.print("Examples: /model openai/gpt-5, /model deepseek/deepseek-chat", style="dim")


def _cmd_compact(current_agent=None, **kwargs):
    if hasattr(current_agent, 'memory') and current_agent.memory:
        messages = current_agent.memory.messages if hasattr(current_agent.memory, 'messages') else []
        msg_count = len(messages)
        
        if msg_count > 0:
            console.print(f"[dim]Compacting {msg_count} messages...[/dim]")
            
            summary_parts = []
            for msg in messages[-10:]:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
                if isinstance(content, str) and content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    summary_parts.append(f"- {role}: {preview}")
                elif isinstance(content, list) and content:
                    preview = str(content)[:200] + "..."
                    summary_parts.append(f"- {role}: {preview}")
            
            if summary_parts:
                summary = "Previous conversation summary:\n" + "\n".join(summary_parts)
                
                from agentica.model.message import Message
                current_agent.memory.messages = []
                summary_msg = Message(
                    role="system",
                    content=f"[Context Summary]\n{summary}\n\n[Note: Previous detailed messages were compacted to save context space.]"
                )
                current_agent.memory.messages.append(summary_msg)
                console.print(f"[green]Context compacted: {msg_count} messages → 1 summary.[/green]")
            else:
                current_agent.memory.messages = []
                console.print(f"[green]Context cleared ({msg_count} messages).[/green]")
        else:
            console.print("[yellow]No messages to compact.[/yellow]")
        console.print("[dim]Workspace memory preserved.[/dim]")
    else:
        console.print("[yellow]No conversation history to compact.[/yellow]")


def _cmd_debug(agent_config=None, current_agent=None, shell_mode=False,
               workspace=None, skills_registry=None, extra_tool_names=None, **kwargs):
    console.print("[bold cyan]Debug Info[/bold cyan]")
    console.print(f"  Model: {agent_config['model_provider']}/{agent_config['model_name']}")
    console.print(f"  Shell Mode: {'[green]ON[/green]' if shell_mode else '[dim]OFF[/dim]'}")
    console.print(f"  Work Dir: {agent_config.get('work_dir') or os.getcwd()}")
    
    if hasattr(current_agent, 'memory') and current_agent.memory:
        msg_count = len(current_agent.memory.messages) if hasattr(current_agent.memory, 'messages') else 0
        console.print(f"  History Messages: {msg_count}")
    
    if hasattr(current_agent, 'tools') and current_agent.tools:
        tool_names = [t.name if hasattr(t, 'name') else str(t) for t in current_agent.tools]
        console.print(f"  Extra Tools: {len(tool_names)}")
    
    if workspace:
        console.print(f"  Workspace: {workspace.path}")
        console.print(f"  Workspace Exists: {workspace.exists()}")
    
    if skills_registry:
        console.print(f"  Skills Loaded: {len(skills_registry)}")


# Command dispatch table
COMMAND_HANDLERS = {
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
    "/help": _cmd_help,
    "/tools": _cmd_tools,
    "/skills": _cmd_skills,
    "/memory": _cmd_memory,
    "/workspace": _cmd_workspace,
    "/newchat": _cmd_newchat,
    "/clear": _cmd_clear,
    "/reset": _cmd_clear,
    "/model": _cmd_model,
    "/compact": _cmd_compact,
    "/debug": _cmd_debug,
}


def _handle_shell_command(user_input: str, work_dir: Optional[str] = None) -> None:
    """Execute a shell command directly."""
    console.print(f"[dim]$ {user_input}[/dim]")
    try:
        result = subprocess.run(
            user_input,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir or os.getcwd()
        )
        if result.stdout:
            console.print(result.stdout, end="")
        if result.stderr:
            console.print(result.stderr, style="red", end="")
        if result.returncode != 0:
            console.print(f"[dim]Exit code: {result.returncode}[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    console.print()


def _process_stream_response(current_agent, final_input: str) -> None:
    """Process the agent's streaming response and display it."""
    status = console.status(f"[bold {COLORS['thinking']}]Thinking...", spinner="dots")
    status.start()
    spinner_active = True
    
    try:
        response_stream = current_agent.run(final_input, stream=True, stream_intermediate_steps=True)
        
        display = StreamDisplayManager(console)
        shown_tool_count = 0
        interrupted = False
        
        for chunk in response_stream:
            if interrupted:
                break
            
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
                
                if chunk.tools and len(chunk.tools) > shown_tool_count:
                    for tool_info in chunk.tools[shown_tool_count:]:
                        tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                        tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except Exception:
                                tool_args = {"args": tool_args}
                        display.display_tool(tool_name, tool_args)
                    shown_tool_count = len(chunk.tools)
                continue
            
            elif chunk.event == "ToolCallCompleted":
                if chunk.tools:
                    for tool_info in reversed(chunk.tools):
                        if "content" in tool_info:
                            tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                            result_content = tool_info.get("content", "")
                            is_error = tool_info.get("tool_call_error", False)
                            elapsed = (tool_info.get("metrics") or {}).get("time")
                            display.display_tool_result(
                                tool_name, str(result_content) if result_content else "",
                                is_error=is_error, elapsed=elapsed,
                            )
                            break
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
                        except Exception:
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
    
    except KeyboardInterrupt:
        current_agent.cancel()
        if spinner_active:
            status.stop()
        console.print("\n[yellow]⚠ Agent cancelled.[/yellow]")
    except AgentCancelledError:
        if spinner_active:
            status.stop()
        console.print("\n[yellow]⚠ Agent cancelled.[/yellow]")
    except Exception as e:
        if spinner_active:
            status.stop()
        console.print(f"\n[bold red]Error during agent execution: {str(e)}[/bold red]")


def _setup_prompt_toolkit(shell_mode_ref: list, skills_registry):
    """Set up prompt_toolkit session and input function.
    
    Args:
        shell_mode_ref: Single-element list holding shell_mode bool (mutable reference)
        skills_registry: Skills registry for command completion
        
    Returns:
        Tuple of (get_input function, use_prompt_toolkit bool)
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.styles import Style
    except ImportError:
        console.print("[yellow]prompt_toolkit not installed. Using basic input mode.[/yellow]")
        console.print("[yellow]Install with: pip install prompt_toolkit[/yellow]")
        console.print()

        def get_input():
            try:
                prompt_char = "$ " if shell_mode_ref[0] else "> "
                console.print(Text(prompt_char, style="green" if shell_mode_ref[0] else "cyan"), end="")
                sys.stdout.flush()
                return input()
            except KeyboardInterrupt:
                return "__CTRL_C__"
            except EOFError:
                return None
        return get_input, False

    # Custom completer for @ file mentions and / commands
    class AgenticaCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

            if text.startswith("/"):
                commands = list(COMMAND_HANDLERS.keys())
                if skills_registry:
                    for trigger in skills_registry.list_triggers().keys():
                        commands.append(trigger)
                for cmd in commands:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text))
                return

            match = re.search(r"@([\w./-]*)$", text)
            if match:
                partial = match.group(1)
                completions = get_file_completions(text)
                for comp in completions:
                    yield Completion(comp, start_position=-len(partial), display=comp)

    # Key bindings
    bindings = KeyBindings()

    @bindings.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text('\n')

    @bindings.add('c-j')
    def _(event):
        event.current_buffer.insert_text('\n')

    @bindings.add('c-d')
    def _(event):
        event.app.exit(result=None)

    @bindings.add('c-x')
    def _(event):
        event.current_buffer.text = "__TOGGLE_SHELL_MODE__"
        event.current_buffer.validate_and_handle()

    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'shell_prompt': 'ansigreen bold',
    })

    history_dir = os.path.dirname(history_file)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=AgenticaCompleter(),
        key_bindings=bindings,
        style=style,
        multiline=False,
    )

    def get_input():
        try:
            if shell_mode_ref[0]:
                return session.prompt([('class:shell_prompt', '$ ')], multiline=False)
            else:
                return session.prompt([('class:prompt', '> ')], multiline=False)
        except KeyboardInterrupt:
            return "__CTRL_C__"
        except EOFError:
            return None

    return get_input, True


def run_interactive(agent_config: dict, extra_tool_names: Optional[List[str]] = None,
                    workspace: Optional[Workspace] = None, skills_registry=None):
    """Run the interactive CLI with prompt_toolkit support."""
    if not agent_config.get("debug_mode"):
        suppress_console_logging()
    
    # Shell mode: use list as mutable reference for closures
    shell_mode_ref = [False]

    # Configure extra tools
    extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
    current_agent = create_agent(agent_config, extra_tools, workspace, skills_registry)

    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        work_dir=agent_config.get("work_dir"),
        extra_tools=extra_tool_names,
        shell_mode=shell_mode_ref[0]
    )

    # Show workspace info
    if workspace and workspace.exists():
        console.print(f"  Workspace: [green]{workspace.path}[/green]")
    if skills_registry and len(skills_registry) > 0:
        triggers = skills_registry.list_triggers()
        if triggers:
            trigger_str = ", ".join(triggers.keys())
            console.print(f"  Skills: [cyan]{len(skills_registry)} loaded[/cyan] (triggers: {trigger_str})")
    console.print()

    get_input, use_prompt_toolkit = _setup_prompt_toolkit(shell_mode_ref, skills_registry)

    # Track consecutive Ctrl+C presses for double-press exit
    ctrl_c_count = 0
    
    # Main interaction loop
    while True:
        try:
            user_input = get_input()
            
            # Handle Ctrl+D (EOF) - exit immediately
            if user_input is None:
                console.print("\nExiting...", style="yellow")
                break
            
            # Handle Ctrl+C
            if user_input == "__CTRL_C__":
                ctrl_c_count += 1
                if ctrl_c_count >= 2:
                    console.print("\nExiting...", style="yellow")
                    break
                console.print("\n[dim]Press Ctrl+C again to exit, or Ctrl+D to quit immediately.[/dim]")
                continue
            
            # Reset Ctrl+C counter on normal input
            ctrl_c_count = 0
            
            user_input = user_input.strip()
            if not user_input:
                continue
            
            # Handle Ctrl-X toggle shell mode
            if user_input == "__TOGGLE_SHELL_MODE__":
                shell_mode_ref[0] = not shell_mode_ref[0]
                mode_str = "[green]Shell Mode ON[/green] - Commands execute directly" if shell_mode_ref[0] else "[cyan]Agent Mode ON[/cyan] - AI processes your input"
                console.print(f"\n{mode_str}")
                continue
            
            # Shell mode: execute commands directly
            if shell_mode_ref[0]:
                # Allow /commands even in shell mode
                if user_input.startswith("/") and user_input.split()[0].lower() in {"/exit", "/quit", "/help", "/model", "/debug", "/clear", "/reset"}:
                    pass  # Fall through to command handling
                else:
                    _handle_shell_command(user_input, agent_config.get("work_dir"))
                    continue
            
            # Handle commands via dispatch table
            first_word = user_input.split()[0].lower() if user_input else ""
            is_command = first_word in COMMAND_HANDLERS or (
                skills_registry and first_word.startswith("/") and 
                skills_registry.match_trigger(user_input) is not None
            )
            if is_command:
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                handler = COMMAND_HANDLERS.get(cmd)
                if handler:
                    result = handler(
                        cmd_args=cmd_args,
                        agent_config=agent_config,
                        current_agent=current_agent,
                        extra_tools=extra_tools,
                        extra_tool_names=extra_tool_names,
                        workspace=workspace,
                        skills_registry=skills_registry,
                        shell_mode=shell_mode_ref[0],
                    )
                    if result == "EXIT":
                        break
                    if isinstance(result, dict) and "current_agent" in result:
                        current_agent = result["current_agent"]
                    continue
                else:
                    # Handle skill triggers
                    if skills_registry:
                        matched_skill = skills_registry.match_trigger(user_input)
                        if matched_skill:
                            skill_prompt = matched_skill.get_prompt()
                            current_agent.add_instruction(f"\n# {matched_skill.name} Skill\n{skill_prompt}")
                            if matched_skill.trigger and user_input.lower().startswith(matched_skill.trigger):
                                user_input = user_input[len(matched_skill.trigger):].strip()
                            console.print(f"[dim]Skill activated: {matched_skill.name}[/dim]")
                            # Fall through to normal processing with modified input
            
            # Parse file mentions
            prompt_text, mentioned_files = parse_file_mentions(user_input)
            
            # Inject file contents if any
            final_input = inject_file_contents(prompt_text, mentioned_files)
            
            # Display user message
            display_user_message(user_input)
            
            # Process agent response
            _process_stream_response(current_agent, final_input)
            
            console.print()  # Blank line after response
            
        except KeyboardInterrupt:
            console.print("\n[dim]Input cancelled. (Type /exit to quit)[/dim]")
            continue
        except Exception as e:
            console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
            continue

    console.print("\nThank you for using Agentica CLI. Goodbye!", style="bold green")
