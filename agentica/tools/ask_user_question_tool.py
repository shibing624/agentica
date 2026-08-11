# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Ask User Question Tool - Human-in-the-loop tool for agent interactions

This tool enables agents to request input or confirmation from users during execution.
It supports various interaction modes:
- Confirmation: Yes/No questions for critical operations
- Text Input: Free-form text input from users
- Selection: Choose from predefined options

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
import re
from typing import Optional, List, Callable, Literal

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
    Human-in-the-loop tool that allows agents to request user input during execution.
    
    This tool provides several interaction modes:
    1. **Confirmation**: Ask yes/no questions for critical decisions
    2. **Text Input**: Request free-form text from users
    3. **Selection**: Present options for users to choose from
    
    The tool uses a callback mechanism to get user input. If no callback is provided,
    it defaults to console input (useful for CLI applications).
    
    Attributes:
        input_callback: Custom callback function for getting user input.
                       Signature: (prompt: str, options: Optional[List[str]]) -> str
        timeout: Timeout in seconds for waiting for user input (default: 300)
        default_on_timeout: Default value to return if timeout occurs
    
    Example:
        ```python
        # Basic usage with console input
        tool = AskUserQuestionTool()
        
        # With custom callback (e.g., for web applications)
        def web_input_callback(prompt: str, options: Optional[List[str]] = None) -> str:
            # Send prompt to frontend and wait for response
            return frontend_api.get_user_input(prompt, options)
        
        tool = AskUserQuestionTool(input_callback=web_input_callback)
        ```
    """
    
    # System prompt for user input tool usage guidance
    ASK_USER_QUESTION_SYSTEM_PROMPT = """## `ask_user_question` (Human-in-the-loop)

You have access to the `ask_user_question` tool to request input or confirmation from the user during execution.

### When to use:
1. **Critical Operations**: Before performing irreversible actions (delete files, send emails, make purchases)
2. **Ambiguous Requests**: When the user's intent is unclear and you need clarification
3. **Sensitive Information**: When you need passwords, API keys, or personal information
4. **Decision Points**: When multiple valid approaches exist and user preference matters
5. **Offer Choices**: To let the user choose a direction when multiple options are reasonable

### Interaction Modes:
- `confirm`: Yes/No questions — use for binary decisions
- `text`: Free-form input — use when you need detailed information
- `select`: Multiple choice — use when there are specific options to choose from

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

        # Register the ask_user_question function
        self.register(self.ask_user_question)
        self.register(self.confirm)
        # Human-in-the-loop: wait indefinitely for the user (like CC/Cursor),
        # don't let the outer ~120s tool-executor timeout auto-pass the prompt
        # and silently continue without an answer.
        self.functions["ask_user_question"].manages_own_timeout = True
        self.functions["confirm"].manages_own_timeout = True

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
        """
        Internal method to get user input using callback or console.
        
        Args:
            prompt: The prompt to display to the user
            options: Optional list of valid options
            
        Returns:
            User's input as a string
        """
        # Prefer the instance callback; fall back to the process-wide default
        # registered by the TUI (covers callback-less instances created by
        # subagents / cron / regressions). Only when BOTH are None do we use
        # bare input() — which is only safe in a non-interactive script where
        # nothing else owns stdin.
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

        # Default console input
        print("\n" + "=" * 60)
        print("🤖 Agent is requesting your input:")
        print("-" * 60)
        print(prompt)
        
        if options:
            print("\nAvailable options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            print(f"\nEnter option number (1-{len(options)}) or type your choice:")
        
        print("-" * 60)
        
        try:
            user_input = input("Your response: ").strip()
            print("=" * 60 + "\n")
            return user_input
        except EOFError:
            # Handle non-interactive environments
            logger.warning("Non-interactive environment detected, using default")
            if self.default_on_timeout:
                return self.default_on_timeout
            return ""
    
    def _resolve_parse_model(self):
        """Get a cheap model for parsing free-form replies, or None.

        Uses the parent agent's auxiliary model (the same cheap tier that
        memory extraction / compression run on). Returns None when the tool
        was never bound to an agent (SDK callers, cron jobs) — callers must
        then fall back to returning the raw reply rather than guessing.
        """
        agent = self._parent_agent
        if agent is None:
            return None
        try:
            return agent.resolve_auxiliary_model("ask_user_question")
        except Exception:
            return None

    async def _llm_parse(self, model, instruction: str) -> Optional[str]:
        """One-shot LLM call returning the model's text, or None on failure.

        Runs on the agent's event loop (this tool is async, so ``invoke`` is
        awaited directly — no loop juggling). A short hard timeout bounds a
        slow/broken auxiliary model so the user's turn never hangs on the
        parse after they already answered.
        """
        try:
            resp = await asyncio.wait_for(
                model.invoke([Message(role="user", content=instruction)]),
                timeout=30.0,
            )
        except Exception as e:
            logger.warning(f"ask_user_question LLM parse failed: {e}")
            return None
        # Extract text from common response shapes (mirrors compression manager).
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
        return text

    async def _parse_select(self, prompt: str, options: List[str], raw_input: str) -> str:
        """Map a free-form reply to one of ``options`` via the auxiliary LLM.

        The user's reply is never enumerable: "3, because workers=10 is ok",
        "第二个吧", "嗯用那个便宜的", a conditional, a typo. Rule matching
        (isdigit / startswith) is exactly what fabricated a choice the user
        never made. The LLM reads the reply in context and returns an option
        index, or null when the reply does not clearly pick one — in which
        case we return the raw reply so the model sees what the user said.

        Falls back to the raw reply (no guessing) when no model is available
        or the call fails.
        """
        if not options or raw_input in options:
            return raw_input
        model = self._resolve_parse_model()
        if model is None:
            logger.info("ask_user_question: no auxiliary model; returning raw reply")
            return raw_input

        lines = [f"The user was asked a question and given numbered options."]
        lines.append(f"Question: {prompt}")
        lines.append("Options:")
        for i, opt in enumerate(options, 1):
            lines.append(f"{i}. {opt}")
        lines.append("")
        lines.append(f"The user's reply: {raw_input}")
        lines.append("")
        lines.append(
            "Map the reply to the option the user intended by meaning, not "
            "wording. The reply may be a number, a paraphrase, a partial "
            "description, or carry a rationale after the choice. Reply with "
            'ONLY a JSON object: {"option_index": <1-based number of the chosen '
            'option, or null if the reply does not clearly choose any>}. No '
            "other text."
        )
        text = await self._llm_parse(model, "\n".join(lines))
        if not text:
            return raw_input
        # The model may wrap JSON in prose or fences; scan for the first
        # {"option_index": ...} object, then fall back to the first integer.
        m = re.search(r'"option_index"\s*:\s*(\d+|null)', text)
        if m:
            val = m.group(1)
            if val == "null":
                return raw_input
            idx = int(val) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return raw_input
        m = re.search(r"\b(\d+)\b", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]
        return raw_input

    async def _parse_confirm(self, prompt: str, raw_input: str) -> str:
        """Resolve a yes/no reply via the auxiliary LLM.

        A short keyword fast-path covers the unambiguous common cases so a
        reachable model isn't taxed for "yes"/"no"/"是". Anything else
        ("嗯行吧可以", "不要", "go ahead") goes to the LLM. Without a model,
        an unrecognised reply defaults to "no" (the safe refusal) rather than
        fabricating consent.
        """
        lower = raw_input.lower().strip()
        if lower in ("yes", "y", "是", "确认", "ok", "确定", "好", "好的", "可以"):
            return "yes"
        if lower in ("no", "n", "否", "取消", "cancel", "不要", "不行"):
            return "no"
        model = self._resolve_parse_model()
        if model is None:
            logger.warning(f"Unrecognised confirm reply, no model to parse: {raw_input}; defaulting to 'no'")
            return "no"
        instruction = (
            f"The user was asked a yes/no question. Map their reply to yes or no.\n"
            f"Question: {prompt}\n"
            f"User's reply: {raw_input}\n\n"
            'Reply with ONLY a JSON object: {"answer": "yes" | "no"}. No other text.'
        )
        text = await self._llm_parse(model, instruction)
        if not text:
            return "no"
        m = re.search(r'"answer"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        if re.search(r"\byes\b", text, re.IGNORECASE):
            return "yes"
        if re.search(r"\bno\b", text, re.IGNORECASE):
            return "no"
        return "no"

    async def ask_user_question(
        self,
        prompt: str,
        mode: Literal["confirm", "text", "select"] = "text",
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Request input from the user during agent execution.
        
        This function pauses agent execution and waits for user input.
        Use this when you need clarification, confirmation, or additional
        information from the user to proceed with a task.
        
        Args:
            prompt: Clear description of what input is needed and why.
                   For confirmations, describe the action that will be taken.
            mode: Type of input requested:
                - "confirm": Yes/No question (returns "yes" or "no")
                - "text": Free-form text input
                - "select": Choose from options (requires options parameter)
            options: List of valid options for "select" mode.
                    Each option should be a clear, concise description.
        
        Returns:
            str: User's response:
                - For "confirm": "yes" or "no"
                - For "text": The user's input text
                - For "select": The selected option
        
        Examples:
            # Confirmation
            response = ask_user_question(
                prompt="Delete all temporary files? This action cannot be undone.",
                mode="confirm"
            )
            
            # Text input
            response = ask_user_question(
                prompt="Please provide the API endpoint URL:",
                mode="text"
            )
            
            # Selection
            response = ask_user_question(
                prompt="Choose the output format:",
                mode="select",
                options=["JSON", "CSV", "XML", "Plain Text"]
            )
        """
        logger.info(f"User input requested: mode={mode}, prompt={prompt[:100]}...")

        # Validate mode and options
        if mode == "select" and not options:
            return json.dumps({
                "error": "Options required for select mode",
                "prompt": prompt,
            }, ensure_ascii=False)

        # Build the full prompt based on mode
        if mode == "confirm":
            full_prompt = f"{prompt}\n\nPlease respond with 'yes' or 'no'."
            valid_options = ["yes", "no", "y", "n"]
        elif mode == "select":
            full_prompt = prompt
            valid_options = options
        else:  # text mode
            full_prompt = prompt
            valid_options = None

        # The input callback is sync and may block indefinitely (the CLI's
        # prompt_toolkit callback parks on a queue until the user types).
        # Run it in a thread so the event loop stays live to service the UI
        # that will deliver the answer.
        loop = asyncio.get_running_loop()
        raw_input = await loop.run_in_executor(
            None, lambda: self._get_input(full_prompt, valid_options if mode == "select" else None)
        )

        # Parse the reply. The user's wording is never enumerable, so select
        # and confirm go through the auxiliary LLM. Text mode is the user's
        # verbatim reply — nothing to resolve.
        if mode == "confirm":
            response = await self._parse_confirm(prompt, raw_input)
        elif mode == "select":
            response = await self._parse_select(prompt, options or [], raw_input)
        else:
            response = raw_input

        logger.info(f"User input received: {response[:100]}...")

        # The prompt is echoed back in full: the CLI renders this result as the
        # transcript's only lasting record of the exchange (the question widget
        # is transient), and a clipped copy would hide what was actually asked.
        # ``raw_input`` preserves the user's full typed reply when it was
        # resolved to an option (e.g. "3, because workers=10 is ok" → option 3),
        # so the model still sees the rationale the user attached.
        result = {
            "mode": mode,
            "prompt": prompt,
            "response": response,
        }
        if mode in ("select", "confirm") and raw_input != response:
            result["raw_input"] = raw_input
        return json.dumps(result, ensure_ascii=False)

    async def confirm(self, prompt: str) -> str:
        """
        Quick confirmation method - shorthand for ask_user_question with mode="confirm".
        
        Use this for simple yes/no questions before critical operations.
        
        Args:
            prompt: Description of the action to confirm.
                   Should clearly state what will happen if confirmed.
        
        Returns:
            str: JSON with "response" field containing "yes" or "no"
        
        Example:
            result = confirm("Proceed with deploying to production?")
            # Returns: {"mode": "confirm", "prompt": "...", "response": "yes"}
        """
        return self.ask_user_question(prompt=prompt, mode="confirm")


class AskUserQuestionRequired(StopAgentRun):
    """
    Exception raised when user input is required but not available.
    
    This can be used to signal that the agent should pause and wait
    for user input before continuing.
    """
    
    def __init__(
        self,
        prompt: str,
        mode: str = "text",
        options: Optional[List[str]] = None,
    ):
        """
        Initialize AskUserQuestionRequired exception.
        
        Args:
            prompt: The prompt to show to the user
            mode: Type of input needed
            options: Options for select mode
        """
        self.prompt = prompt
        self.mode = mode
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
    # Test the AskUserQuestionTool
    tool = AskUserQuestionTool()
    
    print("Testing AskUserQuestionTool...")
    
    # Test confirmation
    print("\n--- Test 1: Confirmation ---")
    result = tool.confirm("Do you want to proceed with the test?")
    print(f"Result: {result}")
    
    # Test text input
    print("\n--- Test 2: Text Input ---")
    result = tool.ask_user_question(
        prompt="Please enter your name:",
        mode="text"
    )
    print(f"Result: {result}")
    
    # Test selection
    print("\n--- Test 3: Selection ---")
    result = tool.ask_user_question(
        prompt="Choose your preferred programming language:",
        mode="select",
        options=["Python", "JavaScript", "Go", "Rust"]
    )
    print(f"Result: {result}")
