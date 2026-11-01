# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for memory prompt loading semantics.
"""


def test_recall_select_prompt_uses_import_time_templates(monkeypatch):
    """Per-turn prompt formatting must not read markdown from disk again.

    Other memory prompts are imported as module constants. The recall selector
    should be the same so a long-running process cannot combine old Python code
    with newly-written markdown templates on the next user turn.
    """
    from agentica.prompts import memory as prompts

    def fail_if_called(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("recall selector should use import-time templates")

    monkeypatch.setattr(prompts, "_load", fail_if_called)

    without_work = prompts.get_memory_recall_select_prompt(limit=3)
    assert "Return at most 3 filenames" in without_work
    assert '{"files": ["<filename>", ...]}' in without_work
    assert '"work"' not in without_work

    with_work = prompts.get_memory_recall_select_prompt(limit=4, work_limit=2)
    assert "Return at most 4 filenames" in with_work
    assert "Return at most 2 earlier-work ids" in with_work
    assert '{"files": ["<filename>", ...], "work": [<id>, ...]}' in with_work
