# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI slash-command package entry — context types and command registry
"""

from agentica.cli.commands.context import (
    CONCURRENT_CMDS,
    IMAGE_EXTENSIONS,
    CommandContext,
    PendingQueue,
)
from agentica.cli.commands.registry import (
    COMMAND_HANDLERS,
    COMMAND_REGISTRY,
    echo_command_invocation,
)

__all__ = [
    "CONCURRENT_CMDS",
    "IMAGE_EXTENSIONS",
    "CommandContext",
    "PendingQueue",
    "COMMAND_HANDLERS",
    "COMMAND_REGISTRY",
    "echo_command_invocation",
]
