# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Swarm - Multi-agent parallel autonomous collaboration.

Unlike Workflow (deterministic, hand-wired pipeline) and the lighter
``Agent.as_tool()`` composition primitive, Swarm enables peer-to-peer
autonomous collaboration where:
- A coordinator analyzes tasks and assigns subtasks to workers
- Workers execute in parallel with shared workspace
- Coordinator synthesizes all worker results into final output

Modes:
- parallel: All agents run the same task concurrently, results are merged
- autonomous: Coordinator decomposes task, assigns to workers, synthesizes results
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.handoff import default_handoff_mapper
from agentica.utils.json_parse import extract_json_array
from agentica.utils.log import logger
from agentica.run_config import RunConfig

COORDINATOR_SYSTEM_PROMPT = """\
You are a task coordinator. Your job is to:
1. Analyze the user's task
2. Decompose it into subtasks that can be handled by your team
3. Return a JSON array of subtask assignments

Available team members:
{team_description}

Return ONLY a JSON array, where each element is:
{{"agent_name": "<name>", "subtask": "<description of what this agent should do>"}}

Example:
[
  {{"agent_name": "researcher", "subtask": "Search for recent papers on transformer architectures"}},
  {{"agent_name": "coder", "subtask": "Implement a simple transformer encoder in PyTorch"}}
]

Rules:
- Each subtask should be self-contained
- Assign to the most appropriate team member based on their description
- You may assign multiple subtasks to the same agent
- Return ONLY valid JSON, no explanations
"""

SYNTHESIZER_PROMPT = """\
You are synthesizing results from multiple agents who worked on subtasks of the user's request.

Original task: {original_task}

Agent results:
{agent_results}

Synthesize these results into a single coherent response that addresses the original task.
Be concise and well-structured. Do not mention the individual agents or subtask assignments.
"""


@dataclass
class SwarmResult:
    """Result from a swarm execution."""
    content: str
    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "parallel"
    total_time: float = 0.0


def _clone_agent_for_task(source: Any) -> Any:
    """Create an isolated Agent clone for a single Swarm subtask.

    The clone shares all *configuration* (model id, api_key, instructions,
    tools, knowledge, workspace, sandbox_config, prompt_config, tool_config)
    with the source agent but gets its own isolated *runtime state*:
      - new agent_id (so _running flags don't collide)
      - fresh WorkingMemory (no shared conversation history)
      - fresh Runner instance (independent run_id / run_response)
      - independent Model copy (own tool registry, metrics, HTTP client)

    This allows Swarm autonomous mode to dispatch multiple subtasks to
    logically the same agent without triggering the concurrent-reuse warning
    and without shared-state corruption.

    The clone is ephemeral — created per-task, discarded after the task finishes.
    """
    from agentica.agent import Agent
    from agentica.memory import WorkingMemory
    from agentica.subagent import SubagentRegistry

    # Reuse the canonical model-cloning logic from SubagentRegistry to keep a
    # single source of truth: shallow-copy + reset tools/functions/tool_choice/
    # metrics + fresh ``Usage`` (critical — concurrent swarm tasks would
    # otherwise pollute each other's token counters via the shared Usage
    # instance from ``copy.copy``) + force fresh HTTP client.
    src_model = source.model
    cloned_model = (
        SubagentRegistry._clone_parent_model(src_model) if src_model is not None else None
    )

    clone = Agent(
        model=cloned_model,
        name=source.name,                          # keep same name for logging
        description=source.description,
        instructions=source.instructions,
        tools=source.tools,                        # cloned by Agent._post_init() (Tool.clone())
        knowledge=source.knowledge,
        workspace=source.workspace,
        work_dir=source.work_dir,
        response_model=source.response_model,
        add_history_to_context=source.add_history_to_context,
        num_history_turns=source.num_history_turns,
        use_structured_outputs=source.use_structured_outputs,
        debug=source.debug,
        enable_tracing=source.enable_tracing,
        hooks=source.hooks,
        prompt_config=source.prompt_config,
        tool_config=source.tool_config,
        long_term_memory_config=source.long_term_memory_config,
        sandbox_config=source.sandbox_config,
        # Safety: every guardrail family that was configured on the source
        # MUST also apply to the clone, otherwise an agent that is protected
        # in normal execution silently loses those protections in autonomous
        # swarm execution. New list copies so a clone-side mutation does not
        # leak back into the source.
        input_guardrails=list(source.input_guardrails) if source.input_guardrails else None,
        output_guardrails=list(source.output_guardrails) if source.output_guardrails else None,
        tool_input_guardrails=list(source.tool_input_guardrails) if source.tool_input_guardrails else None,
        tool_output_guardrails=list(source.tool_output_guardrails) if source.tool_output_guardrails else None,
        working_memory=WorkingMemory(),            # fresh, isolated
        # Context can be a dict, a string, a callable, or any resolved object
        # (see Agent._resolve_context). Only deep-copy dicts so per-task
        # mutations don't leak back; pass anything else through by reference.
        context=dict(source.context) if isinstance(source.context, dict) else source.context,
    )
    return clone


class Swarm:
    """Multi-agent parallel autonomous collaboration system.

    Example - Parallel mode (all agents run same task):
        >>> swarm = Swarm(agents=[agent1, agent2, agent3], mode="parallel")
        >>> result = await swarm.run("Analyze this codebase")

    Example - Autonomous mode (coordinator decomposes and assigns):
        >>> swarm = Swarm(
        ...     agents=[researcher, coder, reviewer],
        ...     mode="autonomous",
        ...     coordinator=coordinator_agent,
        ... )
        >>> result = await swarm.run("Build a REST API with tests")
    """

    def __init__(
        self,
        agents: List[Any],
        mode: str = "parallel",
        coordinator: Optional[Any] = None,
        synthesizer: Optional[Any] = None,
        max_concurrent: int = 10,
    ):
        """Initialize Swarm.

        Args:
            agents: List of Agent instances to form the swarm
            mode: "parallel" (all run same task) or "autonomous" (coordinator assigns)
            coordinator: Agent used to decompose tasks in autonomous mode.
                If None in autonomous mode, the first agent is used as coordinator.
            synthesizer: Agent used to synthesize results. If None, coordinator is used.
            max_concurrent: Maximum number of concurrent agent executions

        Raises:
            ValueError: If agents list is empty or contains duplicate names
        """
        if not agents:
            raise ValueError("Swarm requires at least one agent")

        self.agents = agents
        self.mode = mode
        self.coordinator = coordinator
        self.synthesizer = synthesizer
        self.max_concurrent = max_concurrent
        self._agent_map = {(a.name or f"agent_{i}"): a for i, a in enumerate(agents)}

        # Validate no duplicate keys in agent_map
        if len(self._agent_map) != len(agents):
            names = [(a.name or f"agent_{i}") for i, a in enumerate(agents)]
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Swarm agents must have unique names. Duplicates found: {set(duplicates)}. "
                f"Set explicit agent.name on each agent."
            )

    async def run(
        self,
        task: str,
        config: Optional[RunConfig] = None,
    ) -> SwarmResult:
        """Execute the swarm on a task.

        Args:
            task: The task description
            config: Optional RunConfig for each agent run

        Returns:
            SwarmResult with synthesized content and individual agent results
        """
        import time
        start = time.time()

        if self.mode == "parallel":
            result = await self._run_parallel(task, config)
        elif self.mode == "autonomous":
            result = await self._run_autonomous(task, config)
        else:
            raise ValueError(f"Unknown swarm mode: {self.mode}. Use 'parallel' or 'autonomous'.")

        result.total_time = round(time.time() - start, 3)
        result.mode = self.mode
        return result

    async def _run_parallel(self, task: str, config: Optional[RunConfig] = None) -> SwarmResult:
        """Run all agents on the same task in parallel, merge results."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        def _extract_token_count(metrics: Dict[str, Any], key: str) -> int:
            """Extract token count, handling both scalar and list values."""
            val = metrics.get(key, 0)
            if isinstance(val, list):
                return sum(val)
            return val if isinstance(val, int) else 0

        async def _run_agent(agent):
            async with semaphore:
                agent_name = agent.name or "unnamed"
                logger.chat(f"[swarm] {agent_name} <- task: {task[:120]}")
                try:
                    response = await agent.run(task, config=config)
                    content = response.content if response and response.content else ""
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    # Collect token metrics from response
                    metrics: Dict[str, Any] = response.metrics or {} if response else {}
                    logger.chat(
                        f"[swarm] {agent_name} -> result: "
                        f"{(content or '')[:120]}{'...' if len(content or '') > 120 else ''}"
                    )
                    return {
                        "agent": agent_name,
                        "content": content,
                        "success": True,
                        "input_tokens": _extract_token_count(metrics, "input_tokens"),
                        "output_tokens": _extract_token_count(metrics, "output_tokens"),
                        "total_tokens": _extract_token_count(metrics, "total_tokens"),
                    }
                except Exception as e:
                    logger.error(f"Swarm agent '{agent_name}' failed: {e}")
                    return {"agent": agent_name, "content": str(e), "success": False,
                            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        results = await asyncio.gather(*[_run_agent(a) for a in self.agents])
        agent_results = list(results)

        # Synthesize results
        synthesized = await self._synthesize(task, agent_results, config)

        # Aggregate metrics across all sub-agents
        total_input = sum(r.get("input_tokens", 0) for r in agent_results)
        total_output = sum(r.get("output_tokens", 0) for r in agent_results)
        total_tokens = sum(r.get("total_tokens", 0) for r in agent_results)
        logger.debug(
            f"Swarm parallel completed: {len(agent_results)} agents, "
            f"total_tokens={total_tokens} (in={total_input}, out={total_output})"
        )
        return SwarmResult(content=synthesized, agent_results=agent_results)

    async def _run_autonomous(self, task: str, config: Optional[RunConfig] = None) -> SwarmResult:
        """Coordinator decomposes task, assigns to agents, synthesizes results."""
        # Step 1: Coordinator decomposes task
        coordinator = self.coordinator or self.agents[0]
        assignments = await self._decompose_task(coordinator, task, config)

        if not assignments:
            logger.warning("Coordinator returned no assignments, falling back to parallel mode")
            return await self._run_parallel(task, config)

        # Step 2: Execute assignments in parallel
        # Each assignment gets a cloned Agent with isolated runtime state so that
        # the same logical agent (e.g. "coder") can handle multiple subtasks
        # concurrently without sharing mutable state (run_id, run_response, model.functions).
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_assignment(assignment: Dict):
            async with semaphore:
                agent_name = assignment.get("agent_name", "")
                subtask = assignment.get("subtask", "")
                logger.chat(f"[swarm] {agent_name} <- subtask: {subtask[:120]}")
                source_agent = self._agent_map.get(agent_name)
                if source_agent is None:
                    return {
                        "agent": agent_name,
                        "subtask": subtask,
                        "content": f"Agent '{agent_name}' not found in swarm",
                        "success": False,
                    }
                # Clone the agent for this task to isolate runtime state.
                # Each clone shares configuration (model id, instructions, tools)
                # but gets its own agent_id, working_memory, and Runner instance.
                agent = _clone_agent_for_task(source_agent)
                # Prepend a structured handoff block so the child sees the
                # coordinator's identity, condensed instructions, and the
                # task it was assigned. Replaces the prior bare-string
                # subtask which dropped all parent context.
                handoff_ctx = default_handoff_mapper(
                    parent_agent=coordinator,
                    task=subtask,
                )
                full_subtask = handoff_ctx.render() + "\n\n" + subtask
                try:
                    response = await agent.run(full_subtask, config=config)
                    content = response.content if response and response.content else ""
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    logger.chat(
                        f"[swarm] {agent_name} -> result: "
                        f"{(content or '')[:120]}{'...' if len(content or '') > 120 else ''}"
                    )
                    return {
                        "agent": agent_name,
                        "subtask": subtask,
                        "content": content,
                        "success": True,
                    }
                except Exception as e:
                    logger.error(f"Swarm agent '{agent_name}' failed on subtask: {e}")
                    return {
                        "agent": agent_name,
                        "subtask": subtask,
                        "content": str(e),
                        "success": False,
                    }

        results = await asyncio.gather(*[_run_assignment(a) for a in assignments])
        agent_results = list(results)

        # Step 3: Synthesize results
        synthesized = await self._synthesize(task, agent_results, config)
        return SwarmResult(content=synthesized, agent_results=agent_results)

    async def _decompose_task(
        self, coordinator: Any, task: str, config: Optional[RunConfig] = None
    ) -> List[Dict]:
        """Use coordinator to decompose task into subtask assignments.

        Passes the coordinator prompt via add_messages to avoid mutating
        the coordinator's instructions (which would be unsafe under concurrency).
        """
        team_desc = "\n".join(
            f"- {name}: {a.description or a.name or name}"
            for name, a in self._agent_map.items()
        )
        prompt = COORDINATOR_SYSTEM_PROMPT.format(team_description=team_desc)

        try:
            # Pass coordinator prompt as a system message via add_messages
            # instead of mutating coordinator.instructions (concurrency-safe)
            response = await coordinator.run(
                task,
                add_messages=[{"role": "system", "content": prompt}],
                config=config,
            )
            content = response.content if response and response.content else ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            assignments = extract_json_array(content)
            if assignments is None:
                logger.warning(f"Coordinator returned unparseable response: {content[:200]}")
                return []

            logger.info(f"Coordinator decomposed task into {len(assignments)} subtasks")
            return assignments
        except Exception as e:
            logger.warning(f"Failed to parse coordinator response: {e}")
            return []

    async def _synthesize(
        self,
        original_task: str,
        agent_results: List[Dict],
        config: Optional[RunConfig] = None,
    ) -> str:
        """Synthesize results from multiple agents into a single response.

        Failed agent results are clearly marked so the synthesizer can distinguish
        between valid outputs and errors.
        """
        if len(agent_results) == 1:
            return agent_results[0].get("content", "")

        # Build results summary, marking failures explicitly
        results_text = ""
        for r in agent_results:
            name = r.get("agent", "unknown")
            subtask = r.get("subtask", "")
            content = r.get("content", "")
            success = r.get("success", True)
            status = "[FAILED] " if not success else ""
            if subtask:
                results_text += f"\n### {status}{name} (subtask: {subtask})\n{content}\n"
            else:
                results_text += f"\n### {status}{name}\n{content}\n"

        synthesizer = self.synthesizer or self.coordinator or self.agents[0]
        prompt = SYNTHESIZER_PROMPT.format(
            original_task=original_task,
            agent_results=results_text,
        )

        # Use add_messages instead of mutating synthesizer.instructions (concurrency-safe)
        try:
            response = await synthesizer.run(
                prompt,
                add_messages=[{
                    "role": "system",
                    "content": "You synthesize results from multiple agents into a coherent response.",
                }],
                config=config,
            )
            content = response.content if response and response.content else ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            return content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return results_text
