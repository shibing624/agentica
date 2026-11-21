# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Strong-typed browser tools backed by a replaceable CLI driver.

The tool surface mirrors a managed in-app browser: compact accessibility-tree
observation with short-lived refs, composite actions in one call, and explicit
session cleanup. The default driver talks to the `agent-browser` CLI over a
warm daemon; swapping in another backend only requires a BrowserDriver.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import List, Optional, Protocol

from agentica.tools.base import Tool
from agentica.tools.helpers import ToolDisplayOutput
from agentica.utils.log import logger

_REF_RE = re.compile(r"ref=(e\d+)")
_ACTIONS = ("click", "fill", "type", "hover", "check", "select")
_VALUE_ACTIONS = ("fill", "type", "select")


class BrowserDriver(Protocol):
    """Backend contract: one warm browser session driven by argv commands."""

    async def run(self, *args: str) -> str:
        ...

    async def close(self) -> None:
        ...


class AgentBrowserDriver:
    """Drive `agent-browser` commands against one named daemon session."""

    def __init__(
            self,
            executable: str = "agent-browser",
            session: str = "agentica",
            profile: str = "",
            restore: bool = True,
            timeout: float = 60.0,
            isolated: bool = False,
    ) -> None:
        self.executable = executable
        self.session = session
        self.profile = profile
        self.restore = restore
        self.timeout = timeout
        self.isolated = isolated

    def _argv(self, *command: str) -> List[str]:
        flags = [self.executable, "--session", self.session]
        if self.isolated:
            flags += ["--config", str(Path(__file__).with_name("browser_gateway.json"))]
        if self.profile:
            flags += ["--profile", self.profile]
        if self.restore:
            flags.append("--restore")
        return flags + list(command)

    async def run(self, *args: str) -> str:
        """Run one CLI command; argv only, never a shell string."""
        argv = self._argv(*args)
        env = (
            {key: value for key, value in os.environ.items() if not key.startswith("AGENT_BROWSER_")}
            if self.isolated else None
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"browser command timed out after {self.timeout}s: {' '.join(argv)}") from None
        finally:
            # A cancelled or failed command must not leave the daemon's CLI
            # process alive; `wait_for` cancels `communicate`, not the child.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"browser command failed ({proc.returncode}): {err or out}")
        # Chrome version advisories go to stderr; they are not tool results.
        if err:
            logger.debug("browser CLI stderr: %s", err)
        return out

    async def close(self) -> None:
        """End the daemon. `close` succeeds even when the session is gone."""
        await self.run("close")


class BrowserTool(Tool):
    """A small, strong-typed browser tool surface for an Agent.

    Usage::

        agent = Agent(tools=[BrowserTool()])
        ...
        await browser.close()  # in a finally block at the application boundary

    Not part of the default tool list: browser schemas only cost tokens for
    agents that actually need a browser.
    """

    def __init__(
            self,
            driver: Optional[BrowserDriver] = None,
            *,
            session: str = "agentica",
            profile: str = "",
            restore: bool = True,
            max_elements: int = 240,
            timeout: float = 60.0,
            executable: str = "agent-browser",
    ) -> None:
        super().__init__(name="browser_tool")
        self.driver: BrowserDriver = driver or AgentBrowserDriver(
            executable=executable,
            session=session,
            profile=profile,
            restore=restore,
            timeout=timeout,
        )
        self.max_elements = max_elements
        self._generation = 0
        self._refs: set[str] = set()

        self.register(self.browser_observe, is_read_only=True)
        self.register(self.browser_act, is_destructive=True)
        self.register(self.browser_press, is_destructive=True)
        self.register(self.browser_upload, is_destructive=True)
        self.register(self.browser_close, is_destructive=True)

        # Pin the action enum in the LLM-facing schema, then freeze the schema
        # so model.add_tool() does not regenerate it from type hints.
        act_fn = self.functions["browser_act"]
        act_fn.process_entrypoint()
        act_fn.parameters["properties"]["action"]["enum"] = list(_ACTIONS)
        act_fn.skip_entrypoint_processing = True

    def get_system_prompt(self) -> Optional[str]:
        """Tool-usage policy, injected automatically by Agent."""
        return (
            "## Managed browser\n"
            "- Call browser_observe first. It returns a compact accessibility tree with short "
            "refs and a generation number.\n"
            "- Refs belong to the generation that produced them. Navigation, any action, or a "
            "re-observation invalidates them: call browser_observe again before the next action.\n"
            "- Prefer browser_act for click + wait in ONE call (pass wait_for); do not split "
            "an action and its verification into separate turns.\n"
            "- Use browser_press(text=...) to insert a whole text block at once; never send "
            "one character per call.\n"
            f"- browser_observe returns at most {self.max_elements} interactive elements by "
            "default; raise max_elements only when the target is missing.\n"
            "- Call browser_close when the browser task is finished."
        )

    def _invalidate_refs(self) -> None:
        self._refs.clear()

    def _selector(self, ref: str) -> str:
        key = ref.lstrip("@")
        if not key.startswith("e") or not key[1:].isdigit() or key not in self._refs:
            raise ValueError(
                f"stale or unknown ref {ref!r}; call browser_observe before acting again"
            )
        return f"@{key}"

    async def browser_observe(self, url: str = "", max_elements: int = 0) -> str:
        """Navigate (optionally) and read the page as a compact AX tree with refs.

        Args:
            url: Optional http(s) URL to open before observing. Empty keeps the current page.
            max_elements: Max interactive elements to return; 0 uses the tool default.

        Returns:
            A generation header plus AX-tree lines carrying [ref=eN] markers.
        """
        if url:
            if not url.startswith(("http://", "https://")):
                return "Error: only http(s) URLs are allowed"
            await self.driver.run("open", url)
            self._invalidate_refs()

        cap = max_elements or self.max_elements
        snapshot = await self.driver.run("snapshot", "-i")
        kept: List[str] = []
        refs: set[str] = set()
        for line in snapshot.splitlines():
            # Cap the whole tree, not just ref lines: nested detail rows have no
            # ref and would otherwise grow the result without any bound.
            if len(kept) >= cap:
                break
            match = _REF_RE.search(line)
            if match:
                refs.add(match.group(1))
                kept.append(line)
            elif kept and line.startswith((" ", "\t")):
                # Nested detail (e.g. an option under a combobox) without a ref.
                kept.append(line)

        self._generation += 1
        self._refs = refs
        current_url = await self.driver.run("get", "url")
        header = (
            f"generation={self._generation} elements={len(refs)} cap={cap} "
            f"url={current_url}\n"
            "Refs are valid only until the next action or navigation."
        )
        return ToolDisplayOutput(header + "\n" + "\n".join(kept), {"browser_url": current_url})

    async def browser_act(
            self,
            ref: str,
            action: str = "click",
            text: str = "",
            wait_for: str = "",
    ) -> str:
        """Act on an observed ref and optionally wait, in a single call.

        Args:
            ref: Ref from the latest browser_observe, e.g. "e1" or "@e1".
            action: One of click, fill, type, hover, check, select.
            text: Value for fill/type/select actions.
            wait_for: Optional selector or milliseconds to wait for after the action.

        Returns:
            Command output, the resulting URL, and a ref-invalidation notice.
        """
        if action not in _ACTIONS:
            return f"Error: action must be one of {', '.join(_ACTIONS)}"
        if action in _VALUE_ACTIONS and not text:
            return f"Error: action '{action}' requires text"
        try:
            selector = self._selector(ref)
        except ValueError as exc:
            return f"Error: {exc}"

        args = [action, selector]
        if action in _VALUE_ACTIONS:
            args.append(text)
        out = await self.driver.run(*args)
        if wait_for:
            out += "\nwait: " + await self.driver.run("wait", wait_for)
        current_url = await self.driver.run("get", "url")
        out += "\nurl: " + current_url
        self._invalidate_refs()
        return ToolDisplayOutput(
            out + "\nRefs invalidated; call browser_observe before the next action.",
            {"browser_url": current_url},
        )

    async def browser_press(self, key: str = "", text: str = "") -> str:
        """Press a key or insert a whole text block in one call.

        Args:
            key: Key name such as Enter, Tab, or Control+a.
            text: Whole text to insert at once; never one character per call.

        Returns:
            Command output.
        """
        if text:
            out = await self.driver.run("keyboard", "inserttext", text)
        elif key:
            out = await self.driver.run("press", key)
        else:
            return "Error: provide either key or text"
        self._invalidate_refs()
        return out

    async def browser_upload(self, ref: str, files: List[str]) -> str:
        """Attach local files to an observed file input.

        Args:
            ref: Ref from the latest browser_observe, e.g. "e1".
            files: Local file paths to attach.

        Returns:
            Command output.
        """
        if not files:
            return "Error: files must not be empty"
        try:
            selector = self._selector(ref)
        except ValueError as exc:
            return f"Error: {exc}"
        out = await self.driver.run("upload", selector, *files)
        self._invalidate_refs()
        return out

    async def browser_close(self) -> str:
        """Close the browser daemon and release the session.

        Returns:
            Confirmation text.
        """
        await self.close()
        return "Browser session closed."

    async def close(self) -> None:
        """Application-boundary cleanup; safe to call from a finally block."""
        await self.driver.close()
        self._invalidate_refs()
