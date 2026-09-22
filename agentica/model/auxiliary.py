# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Route side-task LLM calls to the right (usually cheaper) model.

Auxiliary tasks — context compression, goal judging, memory extraction, title
generation — should not always run on the expensive main model. This router
selects a model per task by precedence (NOT a runtime retry/fallback — it picks
one model; it does not re-route if the chosen model errors):

    task-specific model  ->  auxiliary_model  ->  main model

    router = AuxiliaryModelRouter(
        main_model=gpt5,
        auxiliary_model=glm_flash,                  # default for all side tasks
        task_models={"goal_judge": claude_haiku},   # override one task
    )
    router.resolve("compression")   # -> glm_flash
    router.resolve("goal_judge")    # -> claude_haiku
    router.resolve("unknown")       # -> glm_flash (then main if no auxiliary)

Task names are plain strings; common ones are exposed as constants.
"""
from typing import Dict, Optional

# Well-known auxiliary task names (callers may use any string).
COMPRESSION = "compression"
GOAL_JUDGE = "goal_judge"
MEMORY_EXTRACT = "memory_extract"
TITLE = "title"
DEFAULT = "default"


class AuxiliaryModelRouter:
    """Resolve which model to use for a given auxiliary task."""

    def __init__(self, main_model, auxiliary_model=None, task_models: Optional[Dict] = None):
        """
        Args:
            main_model: The agent's primary model (final fallback). Required.
            auxiliary_model: Default model for all side tasks when set.
            task_models: Per-task overrides, e.g. {"compression": cheap_model}.
        """
        self.main_model = main_model
        self.auxiliary_model = auxiliary_model
        self.task_models: Dict = dict(task_models or {})

    def register(self, task: str, model) -> None:
        """Set/replace the model for a specific task."""
        self.task_models[task] = model

    def resolve(self, task: str = DEFAULT):
        """Return the model for ``task`` following task -> auxiliary -> main."""
        return self.task_models.get(task) or self.auxiliary_model or self.main_model
