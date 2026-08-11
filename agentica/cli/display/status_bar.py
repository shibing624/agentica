# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Status-bar fragments and context occupancy helpers
"""

import os
from pathlib import Path
from typing import Optional

def _format_tokens_short(n: int) -> str:
    """Format token count with K/M suffix for compact display."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def context_pct_style(pct: float) -> str:
    """Return Rich style name based on context usage percentage."""
    if pct >= 95:
        return "bold red"
    if pct >= 80:
        return "red"
    if pct >= 50:
        return "yellow"
    return "green"


def build_context_bar(pct: float, width: int = 10) -> str:
    """Build a visual context usage bar like [████░░░░░░]."""
    safe = max(0.0, min(100.0, pct))
    filled = round((safe / 100) * width)
    return f"[{'█' * filled}{'░' * max(0, width - filled)}]"


def display_token_stats(
    console_instance,
    cost_tracker,
    *,
    context_window: int = 128000,
    context_tokens: int = 0,
    tool_use_count: int = 0,
    elapsed_seconds: float = 0.0,
) -> None:
    """Display compact per-response stats footer with color-graded context.

    Format example::

        ctx 50.0% (64K / 128K) [████░░░░░░] · 2 tools · 5.32s · $0.0034
    """
    if cost_tracker is None:
        return

    used_pct = (
        context_tokens / context_window * 100 if context_window > 0 else 0.0
    )
    pct_style = context_pct_style(used_pct)
    bar = build_context_bar(used_pct)

    parts = [
        f"[{pct_style}]ctx {used_pct:.1f}%[/{pct_style}] "
        f"({_format_tokens_short(context_tokens)} / "
        f"{_format_tokens_short(context_window)}) "
        f"[{pct_style}]{bar}[/{pct_style}]"
    ]

    if tool_use_count > 0:
        label = "tool" if tool_use_count == 1 else "tools"
        parts.append(f"[dim]{tool_use_count} {label}[/dim]")

    if elapsed_seconds > 0:
        parts.append(f"[dim]{elapsed_seconds:.2f}s[/dim]")

    # Prompt-cache hits / writes (Anthropic-style, e.g. Venus proxying Claude).
    cache_read = cost_tracker.total_cache_read_tokens
    cache_write = cost_tracker.total_cache_write_tokens
    if cache_read or cache_write:
        seg = []
        if cache_read:
            seg.append(f"{_format_tokens_short(cache_read)} cache_read")
        if cache_write:
            seg.append(f"{_format_tokens_short(cache_write)} cache_write")
        parts.append(f"[dim]{' · '.join(seg)}[/dim]")

    cost = cost_tracker.total_cost_usd
    cost_str = f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"
    parts.append(f"[dim]{cost_str}[/dim]")

    console_instance.print(f"{'  ·  '.join(parts)}")


# ---------------------------------------------------------------------------
# Persistent TUI status bar (prompt_toolkit fragments)
# ---------------------------------------------------------------------------

def _ctx_fg_style(pct: float) -> str:
    """Return a prompt_toolkit style class for context usage percentage."""
    if pct >= 95:
        return "class:sb-critical"
    if pct >= 80:
        return "class:sb-bad"
    if pct >= 50:
        return "class:sb-warn"
    return "class:sb-good"


def format_duration_compact(seconds: float) -> str:
    """Format seconds into compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _format_status_work_dir(work_dir: str) -> str:
    """Return a home-relative absolute path for the persistent status bar."""
    if not work_dir:
        return ""
    path = os.path.abspath(os.path.expanduser(work_dir))
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _compact_status_work_dir(work_dir: str) -> str:
    """Keep the project name visible when the full path does not fit."""
    formatted = _format_status_work_dir(work_dir)
    if not formatted or formatted == "~":
        return formatted
    return Path(formatted).name


def build_status_bar_fragments(
    *,
    model_name: str = "",
    model_provider: str = "",
    profile_name: str = "",
    thinking_mode: str = "",
    work_dir: str = "",
    git_branch: str = "",
    peer_name: str = "",
    context_tokens: int = 0,
    context_window: int = 0,
    cost_usd: float = 0.0,
    active_seconds: float = 0.0,
    last_turn_seconds: float = 0.0,
    spinner_text: str = "",
    terminal_width: int = 80,
    agent_running: bool = False,
    background_terminal_count: int = 0,
    goal_tokens_used: Optional[int] = None,
    goal_token_budget: Optional[int] = None,
):
    """Build prompt_toolkit formatted-text fragments for the persistent status bar.

    Time display uses *agent active time* (sum of all LLM + tool
    execution durations) rather than session wall-clock, plus the
    most recent turn's latency.

    Adapts to terminal width by trying progressively smaller layouts. A wide
    terminal shows model/effort, project path, Git branch/profile, context,
    cost, and timing. The narrowest layout retains model/effort and turn time.

    The model label is rendered as ``provider/model`` when a provider is
    supplied (e.g. ``openai/gpt-4o``). The active Agentica profile name is
    shown first; it is independent from the Git branch.

    When a standing ``/goal`` is active and ``goal_token_budget`` is set, a
    compact ``goal used/budget`` segment is shown so users can watch token
    spend during long goal runs.

    When ``agent_running`` is ``True``:
      - ``spinner_text`` (typically a single spinner glyph like ``⠋``) is
        prepended as the leftmost fragment, giving users a heartbeat
        signal that the agent is working.
      - Every ``class:sb*`` class name is swapped for its ``-active``
        variant, which the CLI style sheet paints with a slightly darker
        ``bg:#0f0f1a`` background. This visual downshift makes it clear
        the bar is in "working" state without hiding any of the (still
        updating) numeric fields — users often want to watch tokens and
        cost tick during long turns.
    """
    base = model_name.split("/")[-1] if "/" in model_name else model_name
    if model_provider:
        label = f"{model_provider}/{base}"
    else:
        label = base
    if len(label) > 26:
        label = label[:23] + "..."
    pct = (context_tokens / context_window * 100) if context_window > 0 else 0.0
    pct_label = f"{pct:.0f}%"
    fg = _ctx_fg_style(pct)
    cost_str = f"${cost_usd:.4f}" if cost_usd < 0.01 else f"${cost_usd:.2f}"

    turn_str = f"⏱ {last_turn_seconds:.1f}s" if last_turn_seconds > 0 else ""
    total_str = f"Σ {format_duration_compact(active_seconds)}" if active_seconds > 0 else ""
    bg_full = ""
    bg_short = ""
    if background_terminal_count > 0:
        noun = "terminal" if background_terminal_count == 1 else "terminals"
        bg_full = (
            f"{background_terminal_count} background {noun} running"
            " · /ps to view · /stop <id> to close"
        )
        bg_short = f"{background_terminal_count} bg · /ps"
    goal_text = ""
    if goal_token_budget is not None:
        used_s = _format_tokens_short(int(goal_tokens_used or 0))
        budget_s = _format_tokens_short(int(goal_token_budget))
        goal_text = f"goal {used_s}/{budget_s}"

    full_work_dir = _format_status_work_dir(work_dir)
    compact_work_dir = _compact_status_work_dir(work_dir)
    ctx_used = _format_tokens_short(context_tokens) if context_tokens else "0"
    ctx_total = _format_tokens_short(context_window) if context_window else "?"

    def compose(
        *,
        project: str = "",
        branch: str = "",
        profile: str = "",
        context_detail: bool = True,
        show_context: bool = True,
        show_cost: bool = True,
        background_detail: bool = True,
        show_goal: bool = True,
        show_peer: bool = True,
    ):
        frags = [("class:sb", " ▸ ")]
        if profile:
            frags.append(("class:sb-dim", f"{profile} "))
        frags.append(("class:sb-strong", label))
        if thinking_mode:
            frags.append(("class:sb-dim", f" {thinking_mode}"))
        if project:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", project),
            ])
        if branch:
            separator = " · " if project else " │ "
            frags.extend([
                ("class:sb-dim", separator),
                ("class:sb", branch),
            ])
        if show_peer and peer_name:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", peer_name),
            ])
        if show_context:
            frags.append(("class:sb-dim", " │ "))
            if context_detail:
                frags.append(("class:sb", f"{ctx_used}/{ctx_total} "))
            frags.append((fg, pct_label))
        if show_goal and goal_text:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", goal_text),
            ])
        if show_cost:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", cost_str),
            ])
        if turn_str:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", turn_str),
            ])
        if total_str:
            frags.append(("class:sb-dim", "  "))
            frags.append(("class:sb-dim", total_str))
        bg_text = bg_full if background_detail else bg_short
        if bg_text:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", bg_text),
            ])
        frags.append(("class:sb", " "))
        return frags

    candidates = [
        compose(project=full_work_dir, branch=git_branch, profile=profile_name),
        compose(
            project=full_work_dir, branch=git_branch, profile=profile_name,
            show_cost=False,
        ),
        compose(
            project=compact_work_dir, branch=git_branch, profile=profile_name,
            show_cost=False,
        ),
        compose(
            project=compact_work_dir, branch=git_branch, profile=profile_name,
            context_detail=False, show_cost=False, background_detail=False,
        ),
        compose(
            profile=profile_name, context_detail=False, show_cost=False,
            background_detail=False, show_peer=False,
        ),
        compose(
            profile=profile_name, show_context=False, show_cost=False,
            background_detail=False, show_peer=False,
        ),
        compose(
            show_context=False, show_cost=False, background_detail=False,
            show_goal=False, show_peer=False,
        ),
    ]
    if terminal_width < 52:
        candidates.insert(
            0,
            compose(show_context=False, show_cost=False, background_detail=False, show_peer=False),
        )
    spinner_width = len(spinner_text) + 2 if agent_running and spinner_text else 0
    available_width = max(1, terminal_width - spinner_width)
    frags = next(
        (candidate for candidate in candidates if sum(len(text) for _, text in candidate) <= available_width),
        candidates[-1],
    )

    # ── Agent-running visual downshift ─────────────────────────────────
    # Two things happen when the agent is actively producing output:
    #   1. Prepend spinner_text as the leftmost fragment (heartbeat).
    #   2. Rewrite every ``class:sb*`` fragment to ``class:sb*-active`` so
    #      the CLI style sheet paints them on ``bg:#0f0f1a`` (one shade
    #      darker than the idle ``#1a1a2e``). This is intentionally subtle
    #      — the bar stays legible and the numeric fields keep updating.
    if agent_running:
        if spinner_text:
            # Use the base class name; the rewrite pass below tacks on ``-active``.
            frags.insert(0, ("class:sb-spin", f" {spinner_text} "))
        frags = [
            (
                cls + "-active"
                if cls.startswith("class:sb") and not cls.endswith("-active")
                else cls,
                text,
            )
            for (cls, text) in frags
        ]

    return frags
