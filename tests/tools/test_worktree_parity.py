from agentica.cli.worktree_binding import WorktreeBinder
from agentica.tools.worktree_tool import WorktreeTool


class _Service:
    def __init__(self):
        self.work_dir = "/repo"
        self.invalidated = []

    def get_session_work_dir(self, session_id, owner=None):
        return self.work_dir

    def set_session_work_dir(self, session_id, work_dir, owner=None):
        self.work_dir = work_dir

    def invalidate_session_agent(self, session_id, owner=None):
        self.invalidated.append((session_id, owner))


def test_create_agent_tools_includes_worktree_for_every_product_surface():
    from agentica.tools.product import build_product_tools

    tools = build_product_tools(
        surface="gateway",
        work_dir="/repo",
        owner="u1",
        session_id="s1",
        service=_Service(),
    )
    names = sorted(tool.name for tool in tools)
    assert "worktree_tool" in names


def test_worktree_tool_is_bound_to_gateway_session_work_dir():
    from agentica.tools.product import build_product_tools

    service = _Service()
    tools = build_product_tools(
        surface="gateway",
        work_dir="/repo",
        owner="u1",
        session_id="s1",
        service=service,
    )
    tool = next(tool for tool in tools if tool.name == "worktree_tool")
    assert tool._binder.work_dir() == "/repo"


def test_worktree_skill_describes_all_surfaces_without_python_scripts():
    from pathlib import Path

    text = Path("agentica/skills/bundled/worktree/SKILL.md").read_text()

    assert "CLI, Web, Desktop, IM" in text
    assert "python3" not in text
