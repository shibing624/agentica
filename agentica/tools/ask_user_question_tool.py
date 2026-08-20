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
a rationale, or none of the above — and you get that reply back verbatim next to
the question. Read it in context: a bare number means that numbered option,
anything else means what it says.

### Recommending an option:
When one choice is your recommendation, put it first in `options` and say so in
its label — e.g. "全量重跑（推荐）" or "Rerun everything (recommended)". The
user then picks it with one keystroke.

### Who this reaches:
This renders in YOUR terminal. If the work you are doing was handed to you by
another agent session, the person who asked for it is sitting at that session,
not this one — this prompt never reaches them, it only blocks until it times
out. Send the question back to that session with `send_message` and end your
turn instead. Keep using this tool for the user who is actually here.

### Best Practices:
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

        self.register(self.ask_user_question)
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
        """
        Request input from the user during agent execution.

        This pauses the agent and waits for the user's reply. Use it when you
        need clarification, confirmation, or additional information to proceed.

        Args:
            prompt: Clear description of what input is needed and why. For a
                confirmation, describe the action that will be taken.
            options: Optional list of choices to present. When given, the user
                picks one (in any wording); when omitted, the reply is
                free-form. Put the choice you recommend first and say so in its
                label, e.g. "Rerun everything (recommended)".

        Returns:
            str: JSON with ``prompt``, the user's ``response`` verbatim, and the
            ``options`` that were offered, if any.

        Examples:
            # Confirmation
            ask_user_question(prompt="Delete all temp files? This cannot be undone.")
            # Free-form input
            ask_user_question(prompt="Please provide the API endpoint URL.")
            # Pick from choices, recommendation first
            ask_user_question(
                prompt="Choose the output format:",
                options=["JSON (recommended)", "CSV", "XML"],
            )
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
