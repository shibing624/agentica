# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Hand a self-contained task to a separate agentica CLI process.

Between the two delegation mechanisms already here:

- ``task`` (``agentica.subagents``) spawns an in-process subagent. Cheap, shares
  this process, and its result comes straight back as the tool result.
- ``delegate`` starts a *whole other* ``agentica`` run in its own OS process,
  with its own model, context window, session log and working directory.

Use ``delegate`` when the work is big enough to deserve its own context window
or must run somewhere else; use ``task`` for everything smaller. The child is a
one-shot ``agentica --query ... --print``: it has no terminal, so it cannot ask
anyone anything, and it does not appear in ``list_agents`` (only an interactive
session publishes a peer record). Its final answer is its stdout, delivered
back through the same background-process reporting that ``execute(background=True)``
uses.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from agentica.model.defaults import provider_env_var
from agentica.global_config import get_profiles
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.base import Tool
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.model.base import Model

# Depth of the current process in a delegation chain: 0 for the session the user
# started, 1 for anything it delegated to. The env var travels to the child, and
# the tool is simply not built past MAX_DEPTH — a tree of agents spawning agents
# spends the user's money in a way nobody is watching.
DEPTH_ENV_VAR = "AGENTICA_DELEGATE_DEPTH"
MAX_DEPTH = 1


def profile_for_model(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    profiles: Optional[dict] = None,
) -> Optional[str]:
    """Name of the one config.yaml profile that uniquely describes this model.

    A delegated worker cannot receive credentials on the command line, and bare
    ``--model_provider``/``--model_name`` flags cannot leave the child's active
    endpoint (see ``resolve_model_config``). Launching the child on a *profile*
    is the only way a model crosses endpoints with its base_url, api_key and
    tuning intact.

    Matching on ``model_name`` alone and taking the first hit is how a worker
    used to land on the wrong endpoint when two profiles shared a name. When
    ``provider`` and/or ``base_url`` are given they must match; if several
    profiles still qualify, return None rather than guess.
    """
    items = profiles if profiles is not None else get_profiles()
    matches: List[str] = []
    for name, profile in items.items():
        if not isinstance(profile, dict) or profile.get("model_name") != model_name:
            continue
        if provider is not None and profile.get("model_provider") != provider:
            continue
        if base_url is not None and profile.get("base_url") != base_url:
            continue
        matches.append(name)
    return matches[0] if len(matches) == 1 else None


# Model class name → provider key. Only the base classes appear: the
# agentica.DeepSeekChat / MoonshotChat / ... factories all return plain
# OpenAIChat instances (with their own base_url), so every OpenAI-compatible
# provider lands on "openai" and the child gets the base_url + api_key as one
# pair. AzureOpenAIChat subclasses OpenAIChat but stands EARLIER in the MRO,
# so it is detected first and correctly refused (Azure credentials have no
# environment variable the child could read). A third-party Model class that
# agentica does not know maps to nothing; the worker then resolves its own
# model from config.yaml / the environment.
_MODEL_CLASS_PROVIDERS = {
    "OpenAIChat": "openai",
    "AzureOpenAIChat": "azure",
    "Claude": "anthropic",
}


def provider_for_model(model: "Model") -> Optional[str]:
    """The provider key a Model instance belongs to, or None if unrecognized."""
    for klass in type(model).__mro__:
        provider = _MODEL_CLASS_PROVIDERS.get(klass.__name__)
        if provider:
            return provider
    return None

# Same ceiling as SubagentRegistry.MAX_CONCURRENT: three parallel workers is
# already more than a person can follow, and each one here is a full model.
MAX_CONCURRENT_DELEGATES = 3

# The child has no user behind it. Saying so is the difference between a worker
# that decides and one that ends its turn asking a question nobody will read.
DELEGATE_PREAMBLE = (
    "You are running headless as a delegated worker for another agentica session. "
    "There is no user at this terminal: nobody can answer a question, approve a "
    "choice, or read anything you print along the way. Decide, act, and finish. "
    "Your final message is the only thing the caller receives, so make it a short "
    "self-contained report: what you did, what you found, and anything the caller "
    "must know to continue.\n\nTask:\n"
)


def agentica_command() -> List[str]:
    """How to start another agentica CLI from this process.

    The console script next to the running interpreter is the same installation
    this session came from, and it reads as itself in a log. ``python -c`` is the
    fallback for a source checkout that was never installed. ``-m agentica.cli.main``
    is deliberately not used: runpy warns on stderr that the module is already
    imported, and with the worker's stderr folded into its log that warning would
    end up inside the report the caller reads.
    """
    bin_dir = Path(sys.executable).parent
    for name in ("agentica", "agentica.exe"):
        script = bin_dir / name
        if script.is_file() and os.access(script, os.X_OK):
            return [str(script)]
    return [sys.executable, "-c", "from agentica.cli.main import main; main()"]


def delegation_depth() -> int:
    """How deep in a delegation chain this process is (0 = started by the user)."""
    try:
        return max(0, int(os.getenv(DEPTH_ENV_VAR, "0") or 0))
    except ValueError:
        return 0


class BuiltinDelegateTool(Tool):
    """Expose ``delegate`` — run a task in a separate agentica CLI process."""

    def __init__(
        self,
        *,
        background_process_registry: BackgroundProcessRegistry,
        permission_mode: Callable[[], str],
        work_dir: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        model: Optional["Model"] = None,
        session_profile: Optional[str] = None,
        profile_lookup: Optional[Callable[..., Optional[str]]] = None,
    ):
        """
        Args:
            background_process_registry: the CLI session's shared registry. The
                child is tracked there like any background command, which is what
                makes /ps, /stop, `wait` and the completion report work on it.
            permission_mode: reads the *live* permission tier of the calling
                session. A callable rather than a string because /permissions
                switches the mode in place without rebuilding the agent, and the
                child must be started under the mode in effect right now.
            work_dir: default working directory for children.
            model_provider, model_name: what the caller itself runs on. Children
                inherit it so delegated work does not silently change model.
            model: the caller's own Model object (the SDK path — no CLI config
                involved). Its provider/id/base_url/api_key describe the worker's
                model; the key travels to the child in its environment, never
                on the command line. Supplied by the caller when it has a real
                model; model_provider/model_name are the CLI's split form of
                the same thing.
            session_profile: the config.yaml profile this session is actually
                running on. An omitted ``model`` launches the child on this
                profile, not on the first profile that happens to share a
                model_name.
            profile_lookup: maps a model name to the config.yaml profile that
                runs it; injectable so tests do not read the real config.
                API keys are never passed on the command line — the child reads
                config.yaml / the environment like any other agentica run.
        """
        super().__init__(name="builtin_delegate_tool")
        self._registry = background_process_registry
        self._permission_mode = permission_mode
        self._work_dir = work_dir
        self._model_provider = model_provider
        self._model_name = model_name
        self._model = model
        self._session_profile = session_profile
        self._profile_lookup = profile_lookup or profile_for_model
        self.register(self.delegate, is_destructive=True)

    async def delegate(self, task: str, label: str = "", work_dir: str = "", model: str = "") -> str:
        """Runs a task in a separate agentica session, in parallel with your own work.

        The worker is a full agentica run in its own OS process: its own context
        window, its own model, its own tools. It starts immediately and you are
        not blocked — start several, keep working, and each one's report arrives
        when it finishes. Call wait(id="term_N") when your next step
        actually needs a worker's answer.

        Use it for a chunk of work that is independent of what you are doing and
        big enough to deserve its own context window: build and verify a feature
        in another checkout, run a long migration end to end, do the same review
        against three services at once.

        Do NOT use it for: anything you can do in a few tool calls yourself;
        a cheap in-process search; or work that needs to ask the user
        something — the worker has no terminal and cannot ask anyone anything.

        The task text is all the worker gets. It cannot see your conversation, so
        state the goal, the files or directory involved, and what "done" means.

        Args:
            task: The complete self-contained instruction for the worker.
            label: A few words naming the task, for the user's process list and
                the completion report. Defaults to the start of the task.
            work_dir: Directory the worker runs in. Defaults to yours — set it
                when the point of delegating is to work on another checkout.
            model: Model for the worker, as "provider/name" (e.g.
                "deepseek/deepseek-chat") or just a name to keep your provider.
                A value that is already your model id — including ids that
                contain a slash, like "openai/glm-5" — is kept whole; only
                when that string is not an id does it split on the first "/".
                A model that exactly one config.yaml profile runs is delegated
                on that profile so its endpoint and key come along. A provider
                nothing on this machine is configured for is refused instead of
                launched into an authentication failure. Defaults to your own
                model (and your session profile, when the CLI has one).

        Returns:
            str: The worker's id and how to reach its result, or why it was refused.
        """
        instruction = (task or "").strip()
        if not instruction:
            return "Nothing delegated: the task text is empty."

        running = self._registry.list(kind="delegate")
        if len(running) >= MAX_CONCURRENT_DELEGATES:
            busy = ", ".join(f'{item.id} ("{item.label}")' for item in running)
            return (
                f"Nothing delegated: {len(running)} delegated tasks are already running "
                f"({busy}), which is the limit of {MAX_CONCURRENT_DELEGATES}. Wait for one "
                f'to finish — wait(id="{running[0].id}") — or do this one yourself.'
            )

        name = (label or " ".join(instruction.split())[:60]).strip()
        # Only a directory the caller named is checked; the session's own
        # work_dir is where this process already runs.
        target_dir = self._work_dir
        if work_dir.strip():
            target_dir = os.path.expanduser(work_dir.strip())
            if not os.path.isdir(target_dir):
                return f"Nothing delegated: work_dir {target_dir!r} is not a directory."

        argv = [
            *agentica_command(),
            "--query",
            DELEGATE_PREAMBLE + instruction,
            "--print",
            # The worker acts under the tier the calling session is under right
            # now. Delegating must not be a way to get work done at a permission
            # level the user did not agree to — in either direction.
            "--permissions",
            self._permission_mode(),
        ]
        child_env: dict = {}
        provider, model_name = self._resolve_model(model)
        inherited = (provider, model_name) == (self._model_provider, self._model_name)
        if inherited and self._session_profile:
            argv += ["--profile", self._session_profile]
            if model_name:
                argv += ["--model_name", model_name]
        elif provider and model_name:
            if inherited:
                parent_url = self._model.base_url if self._model is not None else None
                profile = self._profile_lookup(model_name, provider=None, base_url=parent_url)
            elif (model or "").strip() == model_name:
                profile = self._profile_lookup(model_name, provider=None, base_url=None)
            else:
                profile = self._profile_lookup(model_name, provider=provider, base_url=None)
            if profile:
                argv += ["--profile", profile]
            elif self._model is not None:
                # SDK caller: its model object IS the configuration (no
                # config.yaml involved). The flags describe the model — the
                # base_url is not a secret and rides along as a flag — while
                # the api_key reaches the child through the environment, never
                # the command line where `ps` would show it.
                api_key = self._model.api_key
                key_var = provider_env_var(provider)
                if api_key and not key_var:
                    return (
                        f"Nothing delegated: provider '{provider}' has no environment variable a "
                        f"worker could take an API key from, and credentials never travel the "
                        f"command line. Run such work in-process instead."
                    )
                argv += ["--model_provider", provider, "--model_name", model_name]
                base_url = self._model.base_url
                if base_url:
                    argv += ["--base_url", str(base_url)]
                if api_key and key_var:
                    child_env[key_var] = api_key
            elif provider != self._model_provider:
                # No profile runs this model and the provider is not the one
                # this session runs on: the child would fall back to that
                # provider's public endpoint plus an env key that likely does
                # not exist — a guaranteed 401 six seconds later.
                return (
                    f"Nothing delegated: no config.yaml profile runs '{model_name}' on provider "
                    f"'{provider}', and a worker never receives credentials on the command line, "
                    f"so it would fail authentication. Pick a model one of your profiles runs, "
                    f"or add a profile for it first."
                )
            else:
                argv += ["--model_provider", provider, "--model_name", model_name]
        if target_dir:
            argv += ["--work_dir", target_dir]

        command = " ".join(shlex.quote(part) for part in argv)
        item = self._registry.start(
            command,
            cwd=target_dir,
            env={DEPTH_ENV_VAR: str(delegation_depth() + 1), **child_env},
            kind="delegate",
            label=name,
        )
        logger.info(f"delegated '{name}' to {item.id} (pid {item.pid})")
        return (
            f'Delegated "{name}" to a separate agentica session: {item.id} (PID {item.pid}).\n'
            f"Log: {item.log_path}\n"
            f"It is running now and you are not blocked. Its report is delivered to this "
            f'conversation when it finishes; call wait(id="{item.id}") only if your next '
            f"step needs the answer before you can continue. Do not poll it any other way.\n"
            f"To stop it: the user runs /stop {item.id}."
        )

    def _resolve_model(self, model: str) -> tuple[Optional[str], Optional[str]]:
        """Resolve a caller-supplied model against the session defaults.

        A slash is first treated as part of a model id (Venus-style
        ``openai/glm-5``, or the environment-context form
        ``provider/<id>``). Only when that string is not the caller's own
        id and not a profile's ``model_name`` does it split as
        ``provider/name``.
        """
        choice = (model or "").strip()
        if not choice:
            return self._model_provider, self._model_name
        if choice == self._model_name:
            return self._model_provider, self._model_name
        if (
            self._model_provider
            and self._model_name
            and choice == f"{self._model_provider}/{self._model_name}"
        ):
            return self._model_provider, self._model_name
        if self._profile_lookup(choice, provider=None, base_url=None):
            return self._model_provider, choice
        provider, _, name = choice.partition("/")
        if name:
            return provider, name
        return self._model_provider, choice
