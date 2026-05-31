# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill curator - lint, catalog, and lifecycle-check installed skills.

A small maintenance tool over the skill directories that ``SkillLoader`` knows
about. It surfaces common problems before they bite at runtime:

  - SKILL.md that fails to parse (missing name/description) -> broken
  - description too long / missing when_to_use -> won't trigger well
  - trigger that doesn't start with "/"
  - bundled resource dirs referenced in the body but missing on disk
  - duplicate skill names or triggers across the install set

It also generates a Markdown catalog of all discovered skills.

Library use:
    from agentica.skills import SkillCurator
    curator = SkillCurator()
    reports = curator.scan()
    print(curator.generate_catalog(reports))

CLI use:
    python -m agentica.skills.curator            # lint + summary
    python -m agentica.skills.curator --catalog  # print Markdown catalog
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agentica.skills.skill import Skill
from agentica.skills.skill_loader import SkillLoader
from agentica.utils.log import logger

# Soft limits / well-known bundled resource directories.
_MAX_DESCRIPTION_CHARS = 1024
_MAX_BODY_CHARS = 15_000
_RESOURCE_DIRS = ("scripts", "references", "assets", "templates")

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_BROKEN = "broken"


@dataclass
class SkillIssue:
    """A single problem found with a skill."""
    level: str   # "error" | "warning"
    code: str
    message: str


@dataclass
class SkillReport:
    """Validation result for one skill directory."""
    name: str
    path: str
    location: str
    status: str = STATUS_OK
    issues: List[SkillIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class SkillCurator:
    """Lint, catalog, and lifecycle-check the installed skills."""

    def __init__(self, project_root: Optional[str] = None):
        self.loader = SkillLoader(Path(project_root) if project_root else None)

    # ── scanning ─────────────────────────────────────────────────────────
    def scan(self) -> List[SkillReport]:
        """Discover all skills and validate each. Returns one report per skill."""
        discovered: List[tuple] = []  # (skill_or_none, md_path, location)
        for search_path, location in self.loader.get_search_paths():
            for md_path in self.loader.discover_skills(Path(search_path).expanduser()):
                skill = Skill.from_skill_md(md_path, location)
                discovered.append((skill, md_path, location))

        reports = [self._validate_one(skill, md_path, location)
                   for skill, md_path, location in discovered]
        self._flag_duplicates(reports, discovered)
        for r in reports:
            self._finalize_status(r)
        return reports

    def _validate_one(self, skill: Optional[Skill], md_path: Path, location: str) -> SkillReport:
        if skill is None:
            return SkillReport(
                name=md_path.parent.name,
                path=str(md_path.parent),
                location=location,
                status=STATUS_BROKEN,
                issues=[SkillIssue("error", "parse_failed",
                                   "SKILL.md missing required 'name'/'description' or invalid frontmatter")],
            )

        report = SkillReport(name=skill.name, path=str(skill.path), location=location)

        if not skill.description or not skill.description.strip():
            report.issues.append(SkillIssue("error", "no_description", "description is empty"))
        elif len(skill.description) > _MAX_DESCRIPTION_CHARS:
            report.issues.append(SkillIssue(
                "warning", "long_description",
                f"description is {len(skill.description)} chars (> {_MAX_DESCRIPTION_CHARS})"))

        if skill.trigger and not skill.trigger.startswith("/"):
            report.issues.append(SkillIssue(
                "warning", "bad_trigger", f"trigger '{skill.trigger}' should start with '/'"))

        if not skill.when_to_use:
            report.issues.append(SkillIssue(
                "warning", "no_when_to_use",
                "no 'when_to_use' keywords -> skill won't auto-activate by context"))

        self._check_resources(skill, report)
        self._check_body_size(skill, report)
        return report

    def _check_resources(self, skill: Skill, report: SkillReport) -> None:
        """Warn when the body references a resource dir that doesn't exist."""
        try:
            body = skill.content
        except Exception as e:  # pragma: no cover - IO edge
            logger.debug(f"Could not read body for {skill.name}: {e}")
            return
        for res in _RESOURCE_DIRS:
            if re.search(rf"\b{res}/", body) and not (skill.path / res).is_dir():
                report.issues.append(SkillIssue(
                    "warning", "missing_resource_dir",
                    f"body references '{res}/' but directory is missing"))

    def _check_body_size(self, skill: Skill, report: SkillReport) -> None:
        try:
            if len(skill.content) > _MAX_BODY_CHARS:
                report.issues.append(SkillIssue(
                    "warning", "large_body",
                    f"SKILL.md body is large (> {_MAX_BODY_CHARS} chars); consider splitting"))
        except Exception:
            pass

    def _flag_duplicates(self, reports: List[SkillReport], discovered: List[tuple]) -> None:
        names: Dict[str, int] = {}
        triggers: Dict[str, int] = {}
        for skill, _, _ in discovered:
            if skill is None:
                continue
            names[skill.name] = names.get(skill.name, 0) + 1
            if skill.trigger:
                triggers[skill.trigger] = triggers.get(skill.trigger, 0) + 1

        for (skill, _, _), report in zip(discovered, reports):
            if skill is None:
                continue
            if names.get(skill.name, 0) > 1:
                report.issues.append(SkillIssue(
                    "error", "duplicate_name", f"skill name '{skill.name}' is defined more than once"))
            if skill.trigger and triggers.get(skill.trigger, 0) > 1:
                report.issues.append(SkillIssue(
                    "error", "duplicate_trigger", f"trigger '{skill.trigger}' is used by multiple skills"))

    @staticmethod
    def _finalize_status(report: SkillReport) -> None:
        if report.status == STATUS_BROKEN:
            return
        if any(i.level == "error" for i in report.issues):
            report.status = STATUS_BROKEN
        elif any(i.level == "warning" for i in report.issues):
            report.status = STATUS_WARNING
        else:
            report.status = STATUS_OK

    # ── catalog ──────────────────────────────────────────────────────────
    def generate_catalog(self, reports: Optional[List[SkillReport]] = None) -> str:
        """Render a Markdown catalog of all discovered skills."""
        if reports is None:
            reports = self.scan()
        if not reports:
            return "# Skill Catalog\n\n_No skills found._\n"

        lines = ["# Skill Catalog", "", f"Total skills: {len(reports)}", "",
                 "| Status | Name | Trigger | Location | Path |",
                 "|--------|------|---------|----------|------|"]
        status_icon = {STATUS_OK: "ok", STATUS_WARNING: "warn", STATUS_BROKEN: "broken"}
        for r in sorted(reports, key=lambda x: (x.status != STATUS_BROKEN, x.name)):
            skill = Skill.from_skill_md(Path(r.path) / "SKILL.md", r.location)
            trigger = skill.trigger if skill and skill.trigger else "-"
            lines.append(
                f"| {status_icon.get(r.status, r.status)} | {r.name} | {trigger} | {r.location} | {r.path} |")
        return "\n".join(lines) + "\n"

    def summary(self, reports: Optional[List[SkillReport]] = None) -> str:
        """Human-readable lint summary."""
        if reports is None:
            reports = self.scan()
        ok = sum(1 for r in reports if r.status == STATUS_OK)
        warn = sum(1 for r in reports if r.status == STATUS_WARNING)
        broken = sum(1 for r in reports if r.status == STATUS_BROKEN)
        lines = [f"Scanned {len(reports)} skill(s): {ok} ok, {warn} warning, {broken} broken", ""]
        for r in reports:
            if r.issues:
                lines.append(f"[{r.status}] {r.name} ({r.location})  {r.path}")
                for i in r.issues:
                    lines.append(f"    - {i.level}: {i.code}: {i.message}")
        return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Lint and catalog Agentica skills.")
    parser.add_argument("--root", default=None, help="Project root to scan from")
    parser.add_argument("--catalog", action="store_true", help="Print a Markdown catalog instead of the lint summary")
    args = parser.parse_args(argv)

    curator = SkillCurator(project_root=args.root)
    reports = curator.scan()
    print(curator.generate_catalog(reports) if args.catalog else curator.summary(reports))
    return 1 if any(r.status == STATUS_BROKEN for r in reports) else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
