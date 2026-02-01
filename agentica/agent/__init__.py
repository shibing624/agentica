# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent module - modular agent implementation

This module provides a modular implementation of the Agent class.
The functionality is split across multiple files for better maintainability:

- base.py: Agent class definition, fields, and initialization
- prompts.py: System and user prompt building
- runner.py: Run execution logic (single-round and multi-round)
- session.py: Session and storage management  
- team.py: Team collaboration and task transfer
- tools.py: Default tools (knowledge search, memory, etc.)
- printer.py: Response printing utilities
"""

from agentica.agent.base import Agent

__all__ = ["Agent"]
