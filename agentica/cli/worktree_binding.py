"""Backward-compatible import for the shared worktree binder.

The implementation lives in :mod:`agentica.git.binding` because gateway,
Web/Desktop, IM, headless and CLI all construct the same binder.
"""
from agentica.git.binding import WorktreeBinder

__all__ = ["WorktreeBinder"]
