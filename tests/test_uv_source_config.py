"""uv must not universal-resolve this project's optional extras."""
import tomllib
from pathlib import Path


def test_uv_run_does_not_manage_project_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["uv"]["managed"] is False
    assert data["project"]["optional-dependencies"]