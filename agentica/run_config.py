# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: RunConfig - Per-run configuration that overrides Agent defaults.

Separates "each run may differ" parameters from Agent construction.
Agent fields are defaults; RunConfig overrides them for a specific run.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    Union,
)

from agentica.hooks import RunHooks


@dataclass
class RunConfig:
    """Per-run configuration. Overrides Agent defaults when provided.

    Packs run-level parameters that would otherwise be repeated across
    run()/run_stream()/run_sync()/run_stream_sync() signatures.

    Example:
        >>> response = await agent.run("Analyze data", config=RunConfig(
        ...     run_timeout=30,
        ...     response_model=AnalysisReport,
        ... ))
    """
    response_model: Optional[Type[Any]] = None
    structured_outputs: Optional[bool] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_timeout: Optional[float] = None
    first_token_timeout: Optional[float] = None
    save_response_to_file: Optional[str] = None
    stream_intermediate_steps: bool = False
    hooks: Optional[RunHooks] = None
