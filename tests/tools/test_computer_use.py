import asyncio
import base64
import sys

import pytest

from agentica.agent.approvals import PendingApproval, SessionGrants, classify, describe_approval
from agentica.cli.approvals import format_approval_prompt
from agentica.cli.runtime import parse_args
from agentica.tools.base import Function, FunctionCall


def build_tool():
    from agentica.tools.builtin.computer_use_tool import ComputerUseTool

    return ComputerUseTool('/opt/homebrew/bin/peekaboo', 'com.apple.TextEdit', 42)


@pytest.mark.parametrize('mode', ['read-only', 'auto', 'allow-all'])
def test_computer_use_requires_human_approval_even_in_allow_all(mode):
    call = FunctionCall(function=build_tool().functions['computer_use'], arguments={'action': 'observe'})
    assert classify(mode, call, SessionGrants(), work_dir=None) == 'ask'


def test_third_party_tools_do_not_inherit_local_only_approval_from_their_name():
    call = FunctionCall(function=Function(name='computer_use'), arguments={'action': 'observe'})
    assert classify('allow-all', call, SessionGrants(), work_dir=None) == 'allow'


def test_approval_displays_host_selected_target_not_model_supplied_target():
    tool = build_tool()
    call = FunctionCall(function=tool.functions['computer_use'], arguments={'action': 'observe'})
    question, preview = describe_approval(call)
    assert 'com.apple.TextEdit' in question and '42' in question
    assert 'model' in preview and 'observe' in preview
    assert 'window_id' not in call.function.parameters['properties']
    prompt = format_approval_prompt(PendingApproval(
        'call1', 'computer_use', call.arguments, question, preview, options=('allow', 'deny'),
        local_approval_context='com.apple.TextEdit, window 42',
    ))
    assert 'com.apple.TextEdit, window 42' in prompt
    assert "don't ask again" not in prompt


@pytest.mark.parametrize('has_approval_callback', [False, True])
@pytest.mark.asyncio
async def test_model_pipeline_denial_never_spawns_desktop_driver(monkeypatch, has_approval_callback):
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat

    tool = build_tool()
    calls = []

    async def approve(call):
        calls.append(call.function.name)
        return 'deny'

    async def unexpected_driver(*args):
        pytest.fail(f'Driver must not run after denial, got {args}')

    agent = Agent(
        model=OpenAIChat(id='gpt-4o-mini', api_key='unused'), tools=[tool],
        approve=approve if has_approval_callback else None,
    )
    active_tool = next(kit for kit in agent.tools if isinstance(kit, type(tool)))
    monkeypatch.setattr(active_tool, '_run', unexpected_driver)
    agent.update_model()
    fc = FunctionCall(function=agent.model.functions['computer_use'], arguments={'action': 'observe'}, call_id='need-consent')
    results = []
    async for _ in agent.model.run_function_calls([fc], results):
        pass
    assert calls == (['computer_use'] if has_approval_callback else [])
    assert results[0].tool_call_id == 'need-consent' and results[0].tool_call_error


def test_subagent_cannot_inherit_a_local_terminal_capability():
    from agentica.subagents.runtime import SubagentConfig, SubagentRegistry

    tool = build_tool()
    config = SubagentConfig(type='explore', name='test_local_approval', description='test', system_prompt='test')
    assert not SubagentRegistry()._select_child_tools([tool, tool.functions['computer_use']], config)


@pytest.mark.parametrize('extra', [['--query', 'test'], ['acp'], ['--print']])
def test_computer_use_rejects_noninteractive_arguments(monkeypatch, capsys, extra):
    monkeypatch.setattr(sys, 'argv', ['agentica', '--computer-use', 'com.apple.TextEdit', '42', *extra])
    with pytest.raises(SystemExit) as failure:
        parse_args()
    assert failure.value.code == 2
    assert '--computer-use requires the interactive CLI' in capsys.readouterr().err


@pytest.mark.asyncio
async def test_observation_binds_action_and_sends_pixels_as_image(monkeypatch):
    tool = build_tool()
    commands = []
    png = b'\x89PNG\r\n\x1a\nfixture'

    async def driver(*argv):
        commands.append(argv)
        if argv[0] == 'see':
            from pathlib import Path

            Path(argv[argv.index('--path') + 1]).write_bytes(png)
            return {
                'snapshot_id': 'ps1_fixture', 'snapshot_reusable': True,
                'mutation_targeting_available': True,
                'ui_elements': [{'id': 'B1', 'role': 'button', 'is_actionable': True}],
            }
        return {}

    monkeypatch.setattr(tool, '_run', driver)
    with pytest.raises(ValueError, match='Observe'):
        await tool.computer_use('click', element_id='B1')
    observe = await tool.computer_use('observe')
    assert observe['images'] == [{'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}]
    assert base64.b64encode(png).decode() not in observe['text_summary']
    with pytest.raises(ValueError, match='element'):
        await tool.computer_use('click', element_id='another button')
    await tool.computer_use('click', element_id='B1')
    assert commands[1] == (
        'click', '--app=com.apple.TextEdit', '--window-id=42',
        '--snapshot=ps1_fixture', '--on=B1',
    )
    with pytest.raises(ValueError, match='Observe'):
        await tool.computer_use('click', element_id='B1')
    assert len(commands) == 2


@pytest.mark.asyncio
async def test_set_value_addresses_only_the_exact_window_snapshot(monkeypatch):
    tool = build_tool()
    tool._snapshot = 'ps1_fixture'
    tool._elements = {'F1': {'is_value_settable': True}}
    commands = []

    async def driver(*argv):
        commands.append(argv)
        return {}

    monkeypatch.setattr(tool, '_run', driver)
    await tool.computer_use('set_value', element_id='F1', text='--foreground')
    assert commands == [('set-value', '--snapshot=ps1_fixture', '--on=F1', '--value=--foreground')]


@pytest.mark.asyncio
async def test_cancelled_driver_is_killed_and_reaped(monkeypatch):
    tool = build_tool()
    from agentica.tools.builtin import computer_use_tool

    started = asyncio.Event()
    processes = []
    spawn = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        proc = await spawn(sys.executable, '-c', 'import time; time.sleep(60)', **kwargs)
        processes.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(computer_use_tool.asyncio, 'create_subprocess_exec', launch)
    task = asyncio.create_task(tool._run('see'))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert processes[0].returncode == -9
