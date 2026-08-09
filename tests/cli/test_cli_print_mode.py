# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `agentica --query ... --print` — the machine-readable one-shot mode.

This is what a delegating session runs, so its stdout has to be the answer and
only the answer.
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentica.cli.main import main
from agentica.cli.runtime import parse_args
from agentica.model.usage import Usage
from agentica.run_response import RunResponse


def _run_one_shot(chunks, *, print_mode, query="say hi", stream=None):
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
    with (
        patch("agentica.cli.main.parse_args", return_value=args),
        patch("agentica.cli.main._enable_cli_file_logging"),
        patch("agentica.cli.main.resolve_model_config", return_value=resolved),
        patch("agentica.cli.main.create_agent", return_value=agent),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
