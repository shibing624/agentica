# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `agentica --query ... --print` — the machine-readable one-shot mode.

This is what a delegating session runs, so its stdout has to be the answer and
only the answer.
"""
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentica.cli.main import main
from agentica.cli.runtime import parse_args

# `agentica.cli.main` as a dotted patch target resolves to the `main` function
# re-exported by `agentica.cli` on Python 3.10; patch the module object instead.
cli_main = importlib.import_module("agentica.cli.main")
from agentica.model.usage import Usage
from agentica.run.response import RunResponse


def _run_one_shot(chunks, *, print_mode, query="say hi", stream=None, captured=None):
    argv = [
        "agentica",
        "--query",
        query,
        "--model_provider",
        "openai",
        "--model_name",
        "gpt-4o-mini",
        "--no-workspace",
        "--no-experience",
    ]
    if print_mode:
        argv.append("--print")
    with patch.object(sys, "argv", argv):
        args = parse_args()

    def default_stream(_query):
        for chunk in chunks:
            yield RunResponse(content=chunk)

    stream = stream or default_stream
    agent = SimpleNamespace(
        run_stream_sync=stream,
        model=SimpleNamespace(usage=Usage()),
        session_id="session-1",
        _session_log=SimpleNamespace(exists=lambda: True),
    )
    resolved = {"model_provider": "openai", "model_name": "gpt-4o-mini", "base_url": None}

    def fake_create_agent(*args, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        return agent

    with (
        patch.object(cli_main, "parse_args", return_value=args),
        patch.object(cli_main, "_enable_cli_file_logging"),
        patch.object(cli_main, "refresh_model_catalog_in_background"),
        patch.object(cli_main, "resolve_model_config", return_value=resolved),
        patch.object(cli_main, "create_agent", side_effect=fake_create_agent),
    ):
        main()


class TestPrintMode:
    def test_stdout_is_the_answer_and_nothing_else(self, capsys):
        _run_one_shot(["Ported the parser.", " The v1 shim is gone."], print_mode=True)

        assert capsys.readouterr().out == "Ported the parser. The v1 shim is gone.\n"

    def test_without_it_the_run_still_announces_itself(self, capsys):
        _run_one_shot(["Ported the parser."], print_mode=False)

        out = capsys.readouterr().out
        assert "Running query" in out
        assert "gpt-4o-mini" in out

    def test_brackets_in_the_answer_survive(self, capsys):
        # Rich would read [bold] as markup and print nothing for it, quietly
        # corrupting an answer that talks about, say, a log line.
        _run_one_shot(["the log says [warn] retrying"], print_mode=True)

        assert capsys.readouterr().out == "the log says [warn] retrying\n"

    def test_a_failed_run_exits_non_zero(self):
        def exploding_stream(_query):
            raise RuntimeError("model refused")
            yield

        with pytest.raises(SystemExit) as exit_info:
            _run_one_shot([], print_mode=True, stream=exploding_stream)

        # The delegating caller decides what to do next from this status.
        assert exit_info.value.code == 1

    def test_a_one_shot_run_is_not_given_the_question_tool(self, capsys):
        """`--query` has no TUI, so there is nobody to answer a question.

        ``ask_user_question`` waits for a person by design, so a run with no
        person must not be handed it. The previous arrangement mounted it and
        had a canned "no user is available" string answer in the user's place —
        the model asked, and got words the user never said.

        This is also the path a ``delegate`` worker takes (``agentica --query
        ... --print``), so the worker inherits the absence rather than needing
        its own wiring.
        """
        captured = {}
        _run_one_shot(["done"], print_mode=True, captured=captured)
        capsys.readouterr()

        assert captured["include_ask_user_question"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
