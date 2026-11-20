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
from typing import Awaitable, Callable, List, Optional

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
    "finish the rest and list it in your reply as waiting for the user. You "
    "may ask again later if the question still matters."
)

SKIPPED_NOTE = (
    "Not asked: the earlier question in this batch got no answer, so nobody is "
    "at the keyboard. Silence is not consent. Pick your recommended option for "
    "anything local and reversible and say what you chose; for anything hard to "
    "undo or visible outside this machine (push, deploy, send, delete), leave "
    "that step undone and list it in your reply as waiting for the user."
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

This is the only way to get a choice or an answer back from the user: it shows
the question where the user is (the terminal, a Web/Desktop card, or an IM
message with a tap-to-answer link) and blocks until they reply. A page from
`publish_html` / `publish_markdown` can only be read — never use one to ask,
approve, or collect input. Work handed to you by another agent session must go
back in this session's answer; this question never reaches that person.

Ask only when no reasonable default exists or the step is hard to undo. For an
ambiguity you can resolve with a recommendation, take it and tell the user
afterwards. Do not ask whether you may run a command or edit a file: call the
tool, and the permission system prompts when it needs to. A question nobody
answers in time returns `"timed_out": true` with a note on how to proceed.

**Ask everything you already know you need in one call.** Before starting work,
read the request and list every genuine ambiguity it leaves — the entry point,
the target file, the naming, which of two approaches, whether to include the
optional piece. Pass them together as `questions`, one object per item with its
own `id`. The user answers them in one sitting; asking one now and a second one
three minutes later turns a single interruption into two, and the second one
usually arrives when they have moved on to something else. Do not batch
questions you have not thought of yet: ask the ones that block the whole plan,
and make a recommendation for the ones that do not."""

    _ASK_PARAMETERS = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The question to show the user. Use this for a single "
                    "question; use ``questions`` when you have several."
                ),
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
            "questions": {
                "type": "array",
                "description": (
                    "Several questions to ask in one call. Use this instead of "
                    "``prompt`` when you have more than one ambiguity to settle "
                    "before starting work — the user answers them in one sitting "
                    "instead of being interrupted again a minute later. Each item "
                    "carries its own ``id``, and the answer echoes that id."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Stable id for this question; echoed in the answer.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The specific question to ask the user.",
                        },
                        "header": {
                            "type": "string",
                            "description": 'Optional short heading, such as "Confirm" or "Choose Mode".',
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Choices for this question. Put the recommended one "
                                'first and mark it in the label, e.g. "(recommended)".'
                            ),
                        },
                        "multi_select": {
                            "type": "boolean",
                            "description": (
                                "Whether the user may pick more than one option. "
                                "A hint only: the answer is still whatever they type."
                            ),
                        },
                    },
                    "required": ["id", "question"],
                },
            },
        },
        # Neither is required on its own: one call asks either one question
        # (``prompt``) or several (``questions``). Enforced in the body, where a
        # missing or contradictory pair can say what to fix.
        "required": [],
    }

    def __init__(
        self,
        input_callback: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        batch_input_callback: Optional[Callable[[list[dict]], Awaitable[dict[str, str] | None]]] = None,
    ):
        """
        Initialize AskUserQuestionTool.

        Args:
            input_callback: Custom callback function for getting user input.
                           If None, uses console input.
            batch_input_callback: Present all questions at once. Return None to
                                  fall back to sequential prompts on that surface.
        """
        super().__init__(name="ask_user_question_tool")
        self.input_callback = input_callback
        self.batch_input_callback = batch_input_callback

        self.register(self.ask_user_question, parameters_override=self._ASK_PARAMETERS)
        # Human-in-the-loop: wait indefinitely for the user (like CC/Cursor),
        # don't let the outer ~120s tool-executor timeout auto-pass the prompt
        # and silently continue without an answer.
        self.functions["ask_user_question"].manages_own_timeout = True

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for user input tool usage guidance."""
        return self.ASK_USER_QUESTION_SYSTEM_PROMPT

    def _get_input(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
        multi_select: bool = False,
    ) -> str:
        """Get user input via the callback or, as a last resort, bare input().

        ``multi_select`` is passed as a third argument **only** when it is set.
        Every surface that predates multi-select takes ``(prompt, options)``, and
        a single-select question calls it exactly as it always did — so nothing
        that works today changes. A callback that cannot take the third argument
        raises TypeError on a multi-select question, which is the honest
        outcome: it asked for a shape this callback does not implement.
        """
        callback = (
            self.input_callback
            if self.input_callback is not None
            else _default_callback_holder[0]
        )
        if callback is not None:
            try:
                if multi_select:
                    return callback(prompt, options, multi_select)
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
        prompt: Optional[str] = None,
        options: Optional[List[str]] = None,
        questions: Optional[List[dict]] = None,
    ) -> str:
        """Ask the user one question, or several, and wait for the replies.

        Two shapes, because the two situations differ. ``prompt`` (+ optional
        ``options``) asks one thing. ``questions`` asks a batch — the point being
        that the model settles every ambiguity it already knows about in one
        sitting, instead of interrupting the user again a minute later.

        A surface can present the batch together; otherwise it asks one
        question at a time through the same callback a single question uses.

        The reply comes back verbatim. A timed-out item carries ``timed_out``;
        the rest of the batch still returns, so a partial answer is usable
        instead of being thrown away.
        """
        items, error = self._normalize_questions(prompt, options, questions)
        if error is not None:
            return json.dumps({"error": error}, ensure_ascii=False)

        if len(items) == 1 and questions is None:
            # The single-question result shape is unchanged: existing callers
            # (and the transcript rendering) read ``prompt``/``response``.
            return await self._ask_one(
                items[0]["question"], items[0].get("options"), key=None, header=None,
                multi_select=bool(items[0].get("multi_select")),
            )

        if self.batch_input_callback is not None:
            try:
                replies = await self.batch_input_callback(items)
            except AskUserQuestionTimeout as e:
                return json.dumps({"answers": [
                    {
                        **self._result_shape(item, key=item["id"]), "response": "", "timed_out": True,
                        "waited_seconds": round(e.waited_seconds), "note": TIMED_OUT_NOTE,
                    }
                    for item in items
                ]}, ensure_ascii=False)
            if replies is not None:
                answers = []
                for item in items:
                    response = replies[item["id"]]
                    result = {**self._result_shape(item, key=item["id"]), "response": response}
                    if item.get("multi_select"):
                        result["multi_select"] = True
                    if not response:
                        result["skipped"] = True
                        result["note"] = "The user skipped this question. This is not consent."
                    answers.append(result)
                return json.dumps({"answers": answers}, ensure_ascii=False)

        answers = []
        for item in items:
            try:
                raw = await self._ask_one(
                    item["question"], item.get("options"),
                    key=item["id"], header=item.get("header"),
                    multi_select=bool(item.get("multi_select")),
                )
            except Exception as e:
                # One question failing (the Web card's future was cancelled, the
                # IM relay dropped) must not discard the answers already given.
                # Report it as this question's outcome and let the model decide.
                logger.warning(f"ask_user_question failed for {item['id']}: {e}")
                answers.append({
                    **self._result_shape(item, key=item["id"]),
                    "response": "",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            answers.append(json.loads(raw))
            # A timeout means nobody is at the keyboard. Asking the remaining
            # questions would arm one prompt after another that nobody will
            # answer, so the rest are reported as skipped and the model decides.
            if answers[-1].get("timed_out"):
                for remaining in items[len(answers):]:
                    answers.append({
                        **self._result_shape(remaining, key=remaining["id"]),
                        "response": "",
                        "timed_out": True,
                        "skipped": True,
                        "note": SKIPPED_NOTE,
                    })
                break
        return json.dumps({"answers": answers}, ensure_ascii=False)

    @staticmethod
    def _result_shape(item: dict, *, key: Optional[str]) -> dict:
        """The fields every batch answer carries, so one key name serves all.

        A skipped or failed item used to be shaped `{id, question, response}`
        while an answered one was `{prompt, response, id, options}` — a consumer
        reading `prompt` by position saw nothing. Both now carry `prompt`.
        """
        result = {"prompt": item["question"]}
        if key is not None:
            result["id"] = key
        if item.get("options"):
            result["options"] = list(item["options"])
        return result

    @classmethod
    def _normalize_questions(
        cls,
        prompt: Optional[str],
        options: Optional[List[str]],
        questions: Optional[List[dict]],
    ) -> tuple[list[dict], Optional[str]]:
        """Validate the two call shapes into one list, or explain what is wrong.

        Returning an error instead of raising keeps this a normal tool result:
        the model reads what to fix and retries, rather than the run dying on an
        argument it can correct.
        """
        if questions is not None:
            if prompt is not None:
                return [], (
                    "Pass either `prompt` (one question) or `questions` (several), "
                    "not both."
                )
            if not isinstance(questions, list) or not questions:
                return [], "`questions` must be a non-empty JSON array."
            normalized: list[dict] = []
            seen: set[str] = set()
            for index, raw in enumerate(questions):
                if not isinstance(raw, dict):
                    return [], f"questions[{index}] must be an object."
                text = str(raw.get("question") or "").strip()
                if not text:
                    return [], f"questions[{index}] is missing `question`."
                qid = str(raw.get("id") or "").strip() or f"q{index + 1}"
                if qid in seen:
                    # Ids are the answer's only handle; a duplicate would make
                    # two answers indistinguishable.
                    return [], f"questions[{index}] reuses id {qid!r}."
                seen.add(qid)
                item = {"id": qid, "question": text}
                header = str(raw.get("header") or "").strip()
                if header:
                    item["header"] = header
                opts = raw.get("options")
                if opts is not None and opts != []:
                    coerced, why = cls._coerce_options(opts, f"questions[{index}].options")
                    if why is not None:
                        return [], why
                    if coerced:
                        item["options"] = coerced
                if raw.get("multi_select") is True:
                    item["multi_select"] = True
                normalized.append(item)
            return normalized, None

        text = str(prompt or "").strip()
        if not text:
            return [], (
                "Pass `prompt` for one question, or `questions` for several. "
                "Asking nothing is not a question."
            )
        item = {"id": "q1", "question": text}
        if options is not None and options != []:
            coerced, why = cls._coerce_options(options, "options")
            if why is not None:
                return [], why
            if coerced:
                item["options"] = coerced
        return [item], None

    @staticmethod
    def _coerce_options(raw: object, where: str) -> tuple[list[str], Optional[str]]:
        """Turn the options the model sent into a list of strings, or say why not.

        Models routinely serialize the array to a JSON string when the item
        schema is nested (`questions[].options`), and dropping it silently is
        the worst outcome: the caller asked a multiple-choice question and got a
        free-form prompt instead, with nothing in the result explaining why.
        """
        value = raw
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                return [], (
                    f"`{where}` is a string that is not JSON ({e}); pass a JSON "
                    'array of strings, e.g. ["A", "B"].'
                )
        if not isinstance(value, list):
            return [], (
                f"`{where}` must be a JSON array of strings, got "
                f"{type(value).__name__}."
            )
        return [str(option) for option in value], None

    async def _ask_one(
        self,
        question: str,
        options: Optional[List[str]],
        *,
        key: Optional[str],
        header: Optional[str],
        multi_select: bool = False,
    ) -> str:
        """Ask one question through the shared callback and shape one result."""
        shown = f"[{header}] {question}" if header else question
        logger.info(f"User input requested: prompt={shown[:100]}...")

        # The input callback is sync and may block indefinitely (the CLI's
        # prompt_toolkit callback parks on a queue until the user types). Run it
        # in a thread so the event loop stays live to service the UI.
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self._get_input(shown, options, multi_select)
            )
        except AskUserQuestionTimeout as e:
            logger.info(f"User input timed out after {e.waited_seconds:.0f}s")
            result = {
                "prompt": question,
                "response": "",
                "timed_out": True,
                "waited_seconds": round(e.waited_seconds),
                "note": TIMED_OUT_NOTE,
            }
            if key is not None:
                result["id"] = key
            if options:
                result["options"] = list(options)
            return json.dumps(result, ensure_ascii=False)

        logger.info(f"User input received: {response[:100]}...")

        # The prompt is echoed back in full: the CLI renders this result as the
        # transcript's only lasting record of the exchange (the question widget
        # is transient), and a clipped copy would hide what was actually asked.
        # The options go with it — without them the answer is an orphan label.
        result = {
            "prompt": question,
            "response": response,
        }
        if multi_select:
            # The answer is the user's line, verbatim, exactly like a single
            # answer — this flag is what tells the model it may hold several
            # labels and that it should read them against `options`.
            result["multi_select"] = True
        if key is not None:
            result["id"] = key
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
