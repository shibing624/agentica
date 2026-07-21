# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI handlers for skill management (external subcommand).

Handles both `agentica skills ...` and legacy `agentica extensions ...`.
"""
from agentica.cli.runtime import get_console
from agentica.config import AGENTICA_SKILL_DIR
from agentica.skills import get_skill_registry, load_skills, reset_skill_registry
from agentica.skills.installer import install_skills, list_installed_skills, remove_skill


def run_extensions_command(args) -> None:
    """Execute `agentica skills ...` (or legacy `agentica extensions ...`) subcommands."""
    # Support both new 'skills_command' and legacy 'extensions_command' attr
    subcmd = getattr(args, "skills_command", None) or getattr(args, "extensions_command", None)
    target_dir = getattr(args, "target_dir", None) or AGENTICA_SKILL_DIR

    if subcmd == "install":
        replaced_symlinked_skills: list[str] = []
        installed = install_skills(
            args.source,
            destination_dir=target_dir,
            force=args.force,
            replaced_symlinked_skills=replaced_symlinked_skills,
        )
        get_console().print(
            f"[green]Installed {len(installed)} skill(s) into {target_dir}[/green]"
        )
        for skill in installed:
            get_console().print(f"  - [bold]{skill.name}[/bold]: {skill.description}")
        for skill_name in replaced_symlinked_skills:
            get_console().print(f"[green]replaced existing symlinked skill: {skill_name}[/green]")
        if getattr(args, "target_dir", None):
            get_console().print(
                "[yellow]Note: custom --target-dir is only auto-discovered when it is "
                "a standard skills path or included in AGENTICA_EXTRA_SKILL_PATH.[/yellow]"
            )
        return

    if subcmd == "list":
        skills = list_installed_skills(destination_dir=target_dir)
        if not skills:
            get_console().print(f"[yellow]No skills installed in {target_dir}[/yellow]")
            return
        get_console().print(f"[green]Installed skills in {target_dir}[/green]")
        for skill in skills:
            get_console().print(f"  - [bold]{skill.name}[/bold]: {skill.description}")
        return

    if subcmd == "remove":
        removed_path = remove_skill(args.skill_name, destination_dir=target_dir)
        get_console().print(
            f"[green]Removed skill {args.skill_name} from {removed_path}[/green]"
        )
        return

    if subcmd == "reload":
        reset_skill_registry()
        load_skills()
        registry = get_skill_registry()
        get_console().print(
            f"[green]Reloaded {len(registry)} skill(s) from standard search paths[/green]"
        )
        return

    raise ValueError(f"Unsupported skills command: {subcmd}")
