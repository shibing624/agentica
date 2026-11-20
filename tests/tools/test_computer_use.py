import asyncio
import base64
import json
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentica.agent.approvals import ApprovalRegistry, SessionGrants, classify, make_approve
from agentica.cli.approvals import format_approval_prompt
from agentica.cli.runtime import parse_args
from agentica.tools.base import Function, FunctionCall
from agentica.tools.builtin.computer_use_tool import ComputerUseTool

EDITOR = 'com.example.Editor'
MUSIC = 'com.example.Music'
TERMINAL = 'com.example.Terminal'
PNG = b'\x89PNG\r\n\x1a\nfixture'


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    tool = ComputerUseTool('/opt/homebrew/bin/peekaboo', TERMINAL)
    commands = []
    bundle = tmp_path / 'Music.app'
    info = bundle / 'Contents' / 'Info.plist'
    info.parent.mkdir(parents=True)
    info.write_bytes(plistlib.dumps({'CFBundleIdentifier': MUSIC, 'CFBundleDisplayName': '音乐'}))
    windows = []
    for window_id, title in ((42, '测试文档'), (43, '另一个窗口')):
        windows.append({'window_id': window_id, 'window_title': title, 'observation_capability': 'combined_eligible'})

    async def installed():
        return [bundle]

    async def driver(*argv):
        commands.append(argv)
        if argv[:2] == ('app', 'list'):
            return {'apps': [
                {'bundle_id': EDITOR, 'name': '编辑器', 'pid': 100},
                {'bundle_id': TERMINAL, 'name': '终端', 'pid': 200},
                {'bundle_id': 'com.apple.systempreferences', 'name': '系统设置', 'pid': 300},
            ]}
        if argv[:2] == ('window', 'list'):
            return {'target_application_info': {'bundle_id': EDITOR, 'pid': 100}, 'windows': windows}
        if argv[0] == 'see':
            Path(argv[argv.index('--path') + 1]).write_bytes(PNG)
            return {
                'window_title': '测试文档', 'snapshot_id': 'ps1_fixture', 'snapshot_reusable': True,
                'mutation_targeting_available': True,
                'ui_elements': [{'id': 'B1', 'is_actionable': True}, {'id': 'F1', 'is_value_settable': True}],
            }
        return {}

    monkeypatch.setattr(tool, '_installed_app_paths', installed)
    monkeypatch.setattr(tool, '_run', driver)
    return tool, commands, windows


def make_call(tool, action, **args):
    return FunctionCall(function=tool.functions['computer_use'], arguments={'action': action, **args}, call_id='cua')


@pytest.mark.parametrize('mode,screen_route', [('read-only', 'ask'), ('auto', 'ask'), ('allow-all', 'allow')])
def test_computer_use_follows_cli_mode(desktop, mode, screen_route):
    tool, _, _ = desktop
    grants = SessionGrants()
    assert classify(mode, make_call(tool, 'list_apps'), grants, work_dir=None) == 'allow'
    for action in ('list_windows', 'launch', 'observe', 'click', 'set_value'):
        assert classify(mode, make_call(tool, action, app_id=EDITOR), grants, work_dir=None) == screen_route


def test_third_party_tools_do_not_inherit_local_only_approval_from_their_name():
    call = FunctionCall(function=Function(name='computer_use'), arguments={'action': 'observe'})
    assert classify('auto', call, SessionGrants(), work_dir=None) == 'allow'


def approval_callback(mode, registry, cards):
    async def unexpected_review(_):
        pytest.fail('A reviewer must not grant terminal Computer Use')

    return make_approve(
        get_mode=lambda: mode, get_grants=SessionGrants, get_registry=lambda: registry,
        get_work_dir=lambda: None, publish=cards.append, apply_path_grant=lambda *_: None,
        review=unexpected_review,
    )


@pytest.mark.asyncio
async def test_approval_displays_discovered_window_and_tracks_permission_switch(desktop):
    tool, commands, _ = desktop
    await tool.computer_use('list_apps')
    await tool.computer_use('list_windows', app_id=EDITOR)
    registry = ApprovalRegistry()
    cards = []
    approve = approval_callback('auto', registry, cards)
    call = make_call(tool, 'observe', app_id=EDITOR, window_id=42)
    task = asyncio.create_task(approve(call))
    await asyncio.sleep(0)
    pending, = cards
    prompt = format_approval_prompt(pending)
    assert all(value in prompt for value in ('编辑器', EDITOR, '测试文档', '42', 'model', 'observe'))
    assert pending.options == ('allow', 'deny')
    from agentica.agent.approvals import classify_pending

    assert classify_pending('allow-all', pending, SessionGrants(), work_dir=None) == 'allow'
    assert classify_pending('read-only', pending, SessionGrants(), work_dir=None) == 'ask'
    registry.decide('cua', 'deny')
    assert await task == 'deny'
    invalid = make_call(tool, 'observe', app_id=EDITOR, window_id=999)
    assert await approve(invalid) == 'deny'
    assert 'list_windows' in invalid.denied_result
    assert len(cards) == 1
    assert len(commands) == 2


@pytest.mark.asyncio
async def test_full_access_still_requires_local_approval_registry(desktop):
    tool, _, _ = desktop
    cards = []
    call = make_call(tool, 'list_apps')
    assert await approval_callback('allow-all', None, cards)(call) == 'deny'
    assert await approval_callback('allow-all', ApprovalRegistry(), cards)(call) == 'allow'
    assert not cards


@pytest.mark.asyncio
async def test_interactive_allow_all_does_not_authorize_automated_turns(desktop):
    from agentica.cli.approvals import build_interactive_approve
    from agentica.cli.interactive.session_state import SessionState

    tool, _, _ = desktop
    agent = SimpleNamespace(
        _cancelled=False, _run_automated_input=True, user_id=None, work_dir=None,
        tool_config=SimpleNamespace(permission_mode='allow-all'),
    )
    state = SessionState(current_agent=agent, agent_running=True)
    approve = build_interactive_approve(state, {})
    assert await approve(make_call(tool, 'observe', app_id=EDITOR, window_id=42)) == 'deny'
    agent._run_automated_input = False
    assert await approve(make_call(tool, 'list_apps')) == 'allow'


@pytest.mark.parametrize('has_approval_callback', [False, True])
@pytest.mark.asyncio
async def test_model_pipeline_denial_never_spawns_desktop_driver(monkeypatch, has_approval_callback):
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat

    tool = ComputerUseTool('/opt/homebrew/bin/peekaboo', TERMINAL)
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
    fc = FunctionCall(function=agent.model.functions['computer_use'], arguments={'action': 'list_apps'}, call_id='need-consent')
    results = []
    async for _ in agent.model.run_function_calls([fc], results):
        pass
    assert calls == (['computer_use'] if has_approval_callback else [])
    assert results[0].tool_call_id == 'need-consent' and results[0].tool_call_error


def test_subagent_cannot_inherit_a_local_terminal_capability(desktop):
    from agentica.subagents.runtime import SubagentConfig, SubagentRegistry

    tool, _, _ = desktop
    config = SubagentConfig(type='explore', name='test_local_approval', description='test', system_prompt='test')
    assert not SubagentRegistry()._select_child_tools([tool, tool.functions['computer_use']], config)


@pytest.mark.parametrize('extra', [['--query', 'test'], ['acp'], ['--print']])
def test_computer_use_rejects_noninteractive_arguments(monkeypatch, capsys, extra):
    monkeypatch.setattr(sys, 'argv', ['agentica', '--computer-use', *extra])
    with pytest.raises(SystemExit) as failure:
        parse_args()
    assert failure.value.code == 2
    assert '--computer-use requires the interactive CLI' in capsys.readouterr().err


@pytest.mark.asyncio
async def test_apps_by_display_name_include_stopped_apps_but_not_terminal_or_system_settings(desktop):
    tool, commands, _ = desktop
    assert json.loads(await tool.computer_use('list_apps')) == [
        {'app_id': EDITOR, 'name': '编辑器', 'running': True},
        {'app_id': MUSIC, 'name': '音乐', 'running': False},
    ]
    for app_id in (TERMINAL, 'com.apple.systempreferences', '--open=/tmp/unapproved'):
        with pytest.raises(ValueError, match='list_apps'):
            await tool.computer_use('launch', app_id=app_id)
    await tool.computer_use('launch', app_id=MUSIC)
    assert commands == [
        ('app', 'list', '--include-hidden'),
        ('app', 'launch', f'--bundle-id={MUSIC}', '--foreground', '--wait-ready', '--wait-for-window'),
    ]


@pytest.mark.asyncio
async def test_observation_is_single_use_and_cannot_be_retargeted(desktop):
    tool, commands, _ = desktop
    await tool.computer_use('list_apps')
    await tool.computer_use('list_windows', app_id=EDITOR)
    with pytest.raises(ValueError, match='Observe'):
        await tool.computer_use('click', app_id=EDITOR, window_id=42, element_id='B1')
    observe = await tool.computer_use('observe', app_id=EDITOR, window_id=42)
    assert observe['images'] == [{'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}]
    assert json.loads(observe['text_summary'])['window_id'] == 42
    with pytest.raises(ValueError, match='Observe'):
        await tool.computer_use('click', app_id=EDITOR, window_id=43, element_id='B1')
    with pytest.raises(ValueError, match='element'):
        await tool.computer_use('click', app_id=EDITOR, window_id=42, element_id='another button')
    await tool.computer_use('click', app_id=EDITOR, window_id=42, element_id='B1')
    assert commands[-1] == ('click', '--pid=100', '--window-id=42', '--snapshot=ps1_fixture', '--on=B1')
    with pytest.raises(ValueError, match='Observe'):
        await tool.computer_use('click', app_id=EDITOR, window_id=42, element_id='B1')
    assert sum(row[0] == 'click' for row in commands) == 1


@pytest.mark.asyncio
async def test_set_value_addresses_only_the_exact_window_snapshot(desktop):
    tool, commands, _ = desktop
    await tool.computer_use('list_apps')
    await tool.computer_use('list_windows', app_id=EDITOR)
    await tool.computer_use('observe', app_id=EDITOR, window_id=42)
    await tool.computer_use('set_value', app_id=EDITOR, window_id=42, element_id='F1', text='--foreground')
    assert commands[-1] == ('set-value', '--snapshot=ps1_fixture', '--on=F1', '--value=--foreground')


@pytest.mark.asyncio
async def test_window_replaced_while_waiting_for_approval_is_not_captured(desktop):
    tool, commands, windows = desktop
    await tool.computer_use('list_apps')
    await tool.computer_use('list_windows', app_id=EDITOR)
    windows[0]['window_title'] = '不同的文档'
    with pytest.raises(ValueError, match='list_windows'):
        await tool.computer_use('observe', app_id=EDITOR, window_id=42)
    assert not any(row[0] == 'see' for row in commands)


@pytest.mark.asyncio
async def test_cancelled_driver_is_killed_and_reaped(monkeypatch, desktop):
    tool, _, _ = desktop
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
    task = asyncio.create_task(ComputerUseTool._run(tool, 'see'))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert processes[0].returncode == -9
