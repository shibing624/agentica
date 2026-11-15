from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
assert (ROOT / "pyproject.toml").is_file()
BUNDLED = ROOT / "agentica" / "skills" / "bundled"


def test_worktree_skill_describes_every_surface_and_forbids_python_fallback():
    text = (BUNDLED / "worktree" / "SKILL.md").read_text()
    for surface in ("CLI", "Web", "Desktop", "IM", "headless"):
        assert surface in text
    assert "all have the `worktree` tool" in text
    assert "Do not call `git worktree add` through `execute`" in text
    assert "Do not write Python inside a product session" in text
    assert "agentica.git.worktree.ensure" in text
    assert "Python SDK callers may use `agentica.git.worktree.ensure`" in text
    assert "multi-agent" not in text


def test_agentica_skill_does_not_assume_cli_only():
    text = (BUNDLED / "agentica" / "SKILL.md").read_text()
    assert "CLI, Web, Desktop, IM and headless surfaces" in text
    assert "shared; a surface only changes how a human invokes it" in text
    assert "almost certainly running inside an agentica CLI session" not in text
