# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Ask User Question Tool - Human-in-the-loop tool for agent interactions

The agent asks the user a question and waits for the reply. The reply is text:
free-form prose, a picked option, a yes/no — the user's wording is never
enumerable, so the tool resolves it against any offered options with the
auxiliary LLM and hands the model a plain answer. No mode parameter, no rule
matching.

Example:
    ```python
    from agentica import Agent
    from agentica.tools.ask_user_question_tool import AskUserQuestionTool

    agent = Agent(
        tools=[AskUserQuestionTool()],
        instructions="When uncertain, ask the user for confirmation.",
    )
    ```
"""
import asyncio
import json
from typing import Optional, List, Callable

from agentica.tools.base import Tool, StopAgentRun
from agentica.model.message import Message
from agentica.utils.log import logger


# Module-level default callback registry. The CLI's TUI registers its
# ask_user_question callback here at startup (see cli/interactive.py). Any
# AskUserQuestionTool instance created WITHOUT an explicit input_callback — a
# subagent spawned mid-turn, a cron job runner, a regression, or an older
# install where the wiring was missing — will then route through the TUI
# callback instead of falling back to bare ``input()``. Bare ``input()`` inside
# a running prompt_toolkit app deadlocks: pt owns stdin in raw mode, so the
# user's keystrokes go to the TextArea and never reach ``input()``, which blocks
# forever (the "CLI froze at ask_user_question, Ctrl+C/Ctrl+D do nothing" bug).
# A registered default breaks that deadlock for every instance, not just the
# one the CLI wired explicitly.
_default_callback_holder: List[Optional[Callable]] = [None]


def set_default_ask_user_question_callback(callback: Optional[Callable]) -> None:
    """Register/clear the process-wide default ask_user_question callback.

    Called by the CLI TUI on startup and teardown. SDK/library callers that
    never start a TUI leave this None, preserving the legacy bare-``input()``
    behavior for non-interactive scripts.
    """
    _default_callback_holder[0] = callback


class AskUserQuestionTool(Tool):
    """
    Human-in-the-loop tool that lets the agent pause and ask the user a
    question mid-run.

    The tool uses a callback mechanism to get user input. If no callback is
    provided, it defaults to console input (useful for CLI applications).

    Attributes:
        input_callback: Custom callback for getting user input.
            Signature: (prompt: str, options: Optional[List[str]]) -> str
        timeout: Timeout in seconds for waiting for user input (default: 300)
        default_on_timeout: Default value to return if timeout occurs

    Example:
        ```python
        # Basic usage with console input
        tool = AskUserQuestionTool()

        # With custom callback (e.g., for web applications)
        def web_input_callback(prompt: str, options: Optional[List[str]] = None) -> str:
            return frontend_api.get_user_input(prompt, options)

        tool = AskUserQuestionTool(input_callback=web_input_callback)
        ```
    """

    ASK_USER_QUESTION_SYSTEM_PROMPT = """## `ask_user_question` (Human-in-the-loop)

You have access to the `ask_user_question` tool to request input or confirmation from the user during execution.

### When to use:
1. **Critical Operations**: Before performing irreversible actions (delete files, send emails, make purchases)
2. **Ambiguous Requests**: When the user's intent is unclear and you need clarification
3. **Sensitive Information**: When you need passwords, API keys, or personal information
4. **Decision Points**: When multiple valid approaches exist and user preference matters
5. **Offer Choices**: Pass `options` to let the user pick one; omit it for free-form answers

### How it behaves:
Ask a plain question in `prompt`. Optionally pass `options` to present numbered
choices. The user replies in their own words — a number, a paraphrase, a yes/no,
a rationale, or none of the above — and the tool resolves it against the options
by meaning and returns the answer as text. You always get back the question, the
resolved answer, and the user's raw reply.

### Who this reaches:
This renders in YOUR terminal. If the work you are doing was handed to you by
another agent session, the person who asked for it is sitting at that session,
not this one — this prompt never reaches them, it only blocks until it times
out. Send the question back to that session with `send_message` and end your
turn instead. Keep using this tool for the user who is actually here.

### Best Practices:
- If you recommend a specific option, make that the FIRST option in the list
  and add "(Recommended)" at the end of the label
- Provide clear, concise prompts that explain what you need and why
- For confirmations, clearly state what action will be taken if confirmed
- Don't overuse — only ask when truly necessary to avoid breaking flow
- Group related questions when possible to minimize interruptions"""

    def __init__(
        self,
        input_callback: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        timeout: int = 300,
        default_on_timeout: Optional[str] = None,
    ):
        """
        Initialize AskUserQuestionTool.

        Args:
            input_callback: Custom callback function for getting user input.
                           If None, uses console input.
            timeout: Timeout in seconds for waiting for user input.
            default_on_timeout: Default value to return if timeout occurs.
        """
        super().__init__(name="ask_user_question_tool")
        self.input_callback = input_callback
        self.timeout = timeout
        self.default_on_timeout = default_on_timeout
        # Bound to the owning agent at wire time so ``ask_user_question`` can
        # resolve an auxiliary model for parsing free-form replies. None for
        # SDK callers that never bind the tool to an agent.
        self._parent_agent = None

        self.register(self.ask_user_question)
        # Human-in-the-loop: wait indefinitely for the user (like CC/Cursor),
        # don't let the outer ~120s tool-executor timeout auto-pass the prompt
        # and silently continue without an answer.
        self.functions["ask_user_question"].manages_own_timeout = True

    def set_parent_agent(self, agent) -> None:
        """Bind to the owning agent so the tool can resolve an auxiliary model
        for LLM-based reply parsing."""
        self._parent_agent = agent

    def clone(self) -> "AskUserQuestionTool":
        """Fresh instance so each agent owns its ``_parent_agent`` slot."""
        new = AskUserQuestionTool(
            input_callback=self.input_callback,
            timeout=self.timeout,
            default_on_timeout=self.default_on_timeout,
        )
        from collections import OrderedDict
        if set(new.functions) != set(self.functions):
            new.functions = OrderedDict(
                (name, new.functions[name])
                for name in self.functions
                if name in new.functions
            )
        return new

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for user input tool usage guidance."""
        return self.ASK_USER_QUESTION_SYSTEM_PROMPT

    def _get_input(self, prompt: str, options: Optional[List[str]] = None) -> str:
        """Get user input via the callback or, as a last resort, bare input()."""
        callback = (
            self.input_callback
            if self.input_callback is not None
            else _default_callback_holder[0]
        )
        if callback is not None:
            try:
                return callback(prompt, options)
            except Exception as e:
                logger.error(f"Error in input callback: {e}")
                if self.default_on_timeout:
                    return self.default_on_timeout
                raise

        print("\n" + "=" * 60)
        print("🤖 Agent is requesting your input:")
        print("-" * 60)
        print(prompt)
        if options:
            print("\nAvailable options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            print(f"\nEnter option number (1-{len(options)}) or type your answer:")
        print("-" * 60)
        try:
            user_input = input("Your response: ").strip()
            print("=" * 60 + "\n")
            return user_input
        except EOFError:
            logger.warning("Non-interactive environment detected, using default")
            if self.default_on_timeout:
                return self.default_on_timeout
            return ""

    def _resolve_parse_model(self):
        """Get a cheap model for parsing free-form replies, or None.

        Uses the parent agent's auxiliary model (the same cheap tier that
        memory extraction / compression run on). Returns None when the tool
        was never bound to an agent (SDK callers, cron jobs) — callers then
        fall back to returning the raw reply rather than guessing.
        """
        agent = self._parent_agent
        if agent is None:
            return None
        try:
            return agent.resolve_auxiliary_model("ask_user_question")
        except Exception:
            return None

    async def _parse_reply(
        self, prompt: str, raw_input: str, options: Optional[List[str]]
    ) -> Optional[str]:
        """Resolve a free-form reply into a plain answer via the auxiliary LLM.

        The user's wording is never enumerable — "3, because workers=10 is ok",
        "第二个吧", "嗯行吧可以" — so a hard rule (isdigit / regex / matching) is
        exactly what fabricated a choice the user never made. The LLM reads the
        reply in context and returns the answer: the chosen option's text, a
        yes/no, or a concise restatement. Returns None when the parse can't run
        (no model / call failed), so the caller falls back to the raw reply.
        """
        model = self._resolve_parse_model()
        if model is None:
            return None

        lines = [
            "The user was asked a question and replied in their own words.",
            f"Question: {prompt}",
        ]
        if options:
            lines.append("Options offered:")
            for i, opt in enumerate(options, 1):
                lines.append(f"{i}. {opt}")
        lines.append(f"User's reply: {raw_input}")
        lines.append("")
        lines.append(
            "State the answer to the question as the user intended it, by "
            "meaning rather than wording. If options were offered and the reply "
            "clearly picks one, reply with that option's text verbatim. If it is "
            "a yes/no question, reply 'yes' or 'no'. Otherwise reply with the "
            "user's answer, concisely. Output only the answer, nothing else."
        )
        try:
            resp = await asyncio.wait_for(
                model.invoke([Message(role="user", content="\n".join(lines))]),
                timeout=30.0,
            )
        except Exception as e:
            logger.warning(f"ask_user_question LLM parse failed: {e}")
            return None
        text = None
        if hasattr(resp, "choices") and resp.choices:
            try:
                text = resp.choices[0].message.content
            except (AttributeError, IndexError):
                pass
        if text is None and hasattr(resp, "content") and isinstance(resp.content, str):
            text = resp.content
        if text is None and isinstance(resp, str):
            text = resp
        if text is None:
            return None
        return text.strip() or None

    async def ask_user_question(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Request input from the user during agent execution.

        This pauses the agent and waits for the user's reply. Use it when you
        need clarification, confirmation, or additional information to proceed.

        Args:
            prompt: Clear description of what input is needed and why. For a
                confirmation, describe the action that will be taken.
            options: Optional list of choices to present. When given, the user
                picks one (in any wording); when omitted, the reply is free-form.

        Returns:
            str: JSON with ``prompt``, the resolved ``response``, the offered
            ``options`` when any were given, and the user's ``raw_input``.

        Examples:
            # Confirmation
            ask_user_question(prompt="Delete all temp files? This cannot be undone.")
            # Free-form input
            ask_user_question(prompt="Please provide the API endpoint URL.")
            # Pick from choices
            ask_user_question(
                prompt="Choose the output format:",
                options=["JSON", "CSV", "XML"],
            )
        """
        logger.info(f"User input requested: prompt={prompt[:100]}...")

        # The input callback is sync and may block indefinitely (the CLI's
        # prompt_toolkit callback parks on a queue until the user types). Run
        # it in a thread so the event loop stays live to service the UI.
        loop = asyncio.get_running_loop()
        raw_input = await loop.run_in_executor(
            None, lambda: self._get_input(prompt, options)
        )

        # Resolve the reply against any offered options. Falls back to the raw
        # reply (never a fabricated option) when the parse can't run.
        response = await self._parse_reply(prompt, raw_input, options)
        if response is None:
            response = raw_input

        logger.info(f"User input received: {response[:100]}...")

        # The prompt is echoed back in full: the CLI renders this result as the
        # transcript's only lasting record of the exchange (the question widget
        # is transient), and a clipped copy would hide what was actually asked.
        # ``raw_input`` preserves the user's full typed reply when it differs
        # from the resolved answer, so the model still sees the rationale.
        result = {
            "prompt": prompt,
            "response": response,
        }
        if options:
            result["options"] = list(options)
        if raw_input != response:
            result["raw_input"] = raw_input
        return json.dumps(result, ensure_ascii=False)


class AskUserQuestionRequired(StopAgentRun):
    """Raised when user input is required but not available — pauses the agent."""

    def __init__(self, prompt: str, options: Optional[List[str]] = None):
        self.prompt = prompt
        self.options = options

        user_message = Message(
            role="assistant",
            content=f"I need your input to proceed:\n\n{prompt}"
        )
        super().__init__(
            exc=f"User input required: {prompt}",
            user_message=user_message,
        )


if __name__ == "__main__":
    tool = AskUserQuestionTool()
    print("Testing AskUserQuestionTool...")
    print("\n--- Confirmation ---")
    print(tool.ask_user_question(prompt="Proceed with the test?"))
    print("\n--- Text input ---")
    print(tool.ask_user_question(prompt="Please enter your name:"))
    print("\n--- Selection ---")
    print(tool.ask_user_question(
        prompt="Choose your preferred language:",
        options=["Python", "JavaScript", "Go", "Rust"],
    ))
