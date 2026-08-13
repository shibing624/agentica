# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Stable `use_capability` proxy (Reasonix TOOL_CONTRACT.zh-CN.md
use_capability 移植).

Optional tools — MCP servers, dynamic skills, anything whose provider-visible
schema is supplied by an external source at runtime — are registered in the host
registry with ``deferred=True`` so they stay executable but do NOT expand into
the top-level provider tool list. The model discovers and invokes them through
this single stable proxy whose name/description/schema never change, so MCP
inventory churn no longer cold-starts the prompt cache from the tools array.

Actions (closed enum, cannot drift):
- ``list``    -> sorted names + descriptions of deferred capabilities
- ``inspect`` -> canonical schema of one deferred capability
- ``call``    -> dispatch a deferred capability by name with its arguments
- ``decline`` -> explicitly refuse a capability (acknowledged, no dispatch)
"""
import json
from typing import Literal

from agentica.tools.base import FunctionCall, Tool


class UseCapabilityTool(Tool):
    """Expose a fixed-schema proxy for discovering/calling deferred tools."""

    def __init__(self):
        super().__init__(name="use_capability", description="Discover and call optional capabilities")
        self.register(self.use_capability)

    async def use_capability(
        self,
        agent,
        action: Literal["list", "inspect", "call", "decline"],
        name: str = "",
        arguments: dict | None = None,
    ) -> str:
        """Discover and invoke deferred (optional) capabilities without changing the provider tool schema.

        Args:
            agent: The owning agent (injected by the runtime; not part of the schema).
            action: One of list / inspect / call / decline.
            name: Capability (tool) name for inspect / call / decline.
            arguments: JSON object of arguments passed to the capability on call.
        """
        functions = getattr(getattr(agent, "model", None), "functions", None) or {}
        deferred = {n: f for n, f in functions.items() if getattr(f, "deferred", False)}

        if action == "list":
            if not deferred:
                return "No optional capabilities are currently available."
            lines = []
            for n in sorted(deferred):
                desc = (deferred[n].description or "").strip().splitlines()[0] if deferred[n].description else ""
                lines.append(f"- {n}: {desc}")
            return "\n".join(lines)

        if action == "inspect":
            fn = deferred.get(name)
            if fn is None:
                return f"Unknown capability {name!r}. Use action='list' to see available capabilities."
            return json.dumps(fn.to_dict(), ensure_ascii=False)

        if action == "call":
            fn = deferred.get(name)
            if fn is None:
                return f"Unknown capability {name!r}. Use action='list' to see available capabilities."
            fc = FunctionCall(function=fn, arguments=arguments or {})
            ok = await fc.execute()
            if ok:
                return str(fc.result)
            return f"Capability {name!r} failed: {fc.error}"

        if action == "decline":
            return f"Declined capability {name!r}." if name else "Declined."

        return f"Unknown action {action!r}. Must be one of list / inspect / call / decline."
