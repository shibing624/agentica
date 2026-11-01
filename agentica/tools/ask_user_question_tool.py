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

# Handed to the model at the moment it decides what to do without an answer.
TIMED_OUT_NOTE = (
    "Nobody answered. Silence is not consent. If the question was a choice "
    "that stays local and reversible, go on with your recommended option and "
    "say what you chose. If it was about something hard to undo or visible "
    "outside this machine (push, deploy, send, delete), do not do that step: "
    "finish the rest and list it in your reply as waiting for the user."
)


class AskUserQuestionTimeout(Exception):
    """Raised by an input callback when nobody answered within its deadline.

    The callback owns the deadline because it owns the prompt: it is the one
    that can take the question off the screen and drop a late answer. The tool
    turns this into a ``timed_out`` result instead of an answer, so the model —
    not this layer — decides what to do without one.
    """

    def __init__(self, waited_seconds: float):
        super().__init__(f"no answer within {waited_seconds:.0f}s")
        self.waited_seconds = waited_seconds


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

    The tool itself has no deadline. A callback may give up on a question
    (the interactive CLI does after ``settings.human_wait_timeout``) by
    raising :class:`AskUserQuestionTimeout`; the result then says ``timed_out``
    with no response and tells the model what silence does and does not
    license. Nothing here invents an answer.
    Agents with nobody to answer at all must not be given this tool: see
    ``include_ask_user_question`` in ``agentica/tools/builtin`` and
    ``agentica/cli/runtime.create_agent``.

    Attributes:
        input_callback: Custom callback for getting user input.
            Signature: (prompt: str, options: Optional[List[str]]) -> str

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
session must go back with `send_message` — this box never reaches that person,
and it blocks until someone answers it.

Ask only when no reasonable default exists or the step is hard to undo. For an
ambiguity you can resolve with a recommendation, take it and tell the user
afterwards. Do not ask whether you may run a command or edit a file: call the
tool, and the permission system prompts when it needs to. A question nobody
answers in time returns `"timed_out": true` with a note on how to proceed."""

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
    ):
        """
        Initialize AskUserQuestionTool.

        Args:
            input_callback: Custom callback function for getting user input.
                           If None, uses console input.
        """
        super().__init__(name="ask_user_question_tool")
        self.input_callback = input_callback

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
            except AskUserQuestionTimeout:
                raise
            except Exception as e:
                logger.error(f"Error in input callback: {e}")
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
            # Nobody on stdin (a pipe, a closed terminal). "" is the honest
            # answer here — the caller-visible result is "no answer", not a
            # substitute for one. Mounting this tool where no user can answer is
            # the real mistake; see the class docstring.
            logger.warning("Non-interactive environment detected, using default")
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
        try:
            response = await loop.run_in_executor(
                None, lambda: self._get_input(prompt, options)
            )
        except AskUserQuestionTimeout as e:
            logger.info(f"User input timed out after {e.waited_seconds:.0f}s")
            result = {
                "prompt": prompt,
                "response": "",
                "timed_out": True,
                "waited_seconds": round(e.waited_seconds),
                "note": TIMED_OUT_NOTE,
            }
            if options:
                result["options"] = list(options)
            return json.dumps(result, ensure_ascii=False)

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
