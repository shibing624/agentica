# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Ask User Question Tool - Human-in-the-loop tool for agent interactions

The agent asks the user a question and waits for the reply. The reply is handed
back as text, exactly as typed — nothing between the keystrokes and the model,
not an auxiliary LLM "resolving" the wording and not a rule mapping "3" to an
option. The question and the options travel with the answer, so the model reads
"3", "C", "the last one" and "the cheap one" off the same list the user saw,
with all the context that made it ask.

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

    # Session-context only. The call shape lives on the tool schema; repeating
    # it here taught models to copy a Python list as a string.
    ASK_USER_QUESTION_SYSTEM_PROMPT = """## `ask_user_question`

This prompt renders in YOUR terminal. Work handed to you by another agent
session must go back with `send_message` — this box never reaches that person
and only blocks until it times out."""

    _ASK_PARAMETERS = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The question to show the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Choices as a JSON array of strings, not a string. "
                    'Example: ["Keep current (recommended)", "Rewrite", "Skip"]. '
                    "Omit for a free-form answer. When you recommend one, put it "
                    "first AND mark it in the label — the user picks it by position, "
                    'and only the label says which one you meant, e.g. "（推荐）" '
                    'or "(recommended)".'
                ),
            },
        },
        "required": ["prompt"],
    }

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

        self.register(self.ask_user_question, parameters_override=self._ASK_PARAMETERS)
        # Human-in-the-loop: wait indefinitely for the user (like CC/Cursor),
        # don't let the outer ~120s tool-executor timeout auto-pass the prompt
        # and silently continue without an answer.
        self.functions["ask_user_question"].manages_own_timeout = True

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

    async def ask_user_question(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
    ) -> str:
        """Ask the user a question and wait for their reply.

        Use when a choice or confirmation is needed before continuing.
        ``options`` is a JSON array of strings, never a stringified array.
        Omit it for a free-form answer. The reply comes back verbatim.
        """
        logger.info(f"User input requested: prompt={prompt[:100]}...")

        # The input callback is sync and may block indefinitely (the CLI's
        # prompt_toolkit callback parks on a queue until the user types). Run
        # it in a thread so the event loop stays live to service the UI.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self._get_input(prompt, options)
        )

        logger.info(f"User input received: {response[:100]}...")

        # The prompt is echoed back in full: the CLI renders this result as the
        # transcript's only lasting record of the exchange (the question widget
        # is transient), and a clipped copy would hide what was actually asked.
        # The options go with it — without them the answer is an orphan label.
        result = {
            "prompt": prompt,
            "response": response,
        }
        if options:
            result["options"] = list(options)
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
