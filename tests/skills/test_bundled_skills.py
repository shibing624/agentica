# -*- coding: utf-8 -*-
"""Tests for the skills shipped inside the package.

An agentica install should know how to drive agentica, so bundled skills travel
with the code. They are ordinary skills in every other respect — in particular
a user who writes their own of the same name must win, and slash commands
come from ``name`` (no ``trigger`` field; that is not part of the standard
Agent Skills format). The CLI ``/worktree`` and ``/cron`` commands are
registered first, so they win over the skills' auto-commands of the same slug.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentica.skills.skill import Skill
from agentica.skills.skill_loader import SkillLoader
from agentica.skills.skill_registry import SkillRegistry

BUNDLED = ("agentica", "cron", "multi-agent", "worktree")


class TestBundledSkillsShip(unittest.TestCase):
    def test_the_bundled_directory_lives_inside_the_package(self):
        # Not a fixture path: this is what gets packaged, and package-data
        # picks it up through the agentica/**/*.md glob.
        self.assertTrue(SkillLoader.BUNDLED_SKILL_DIR.is_dir())
        self.assertEqual(SkillLoader.BUNDLED_SKILL_DIR.parent.name, "skills")

    def test_every_bundled_skill_parses(self):
        for name in BUNDLED:
            path = SkillLoader.BUNDLED_SKILL_DIR / name / "SKILL.md"
            skill = Skill.from_skill_md(path, location="bundled")
            self.assertIsNotNone(skill, f"{name} failed to parse")
            self.assertEqual(skill.name, name)
            self.assertTrue(skill.description.strip())
            # Standard Agent Skills have no trigger; CLI /slug comes from name.
            self.assertIsNone(skill.trigger, f"{name} must not set trigger")
            self.assertTrue(skill.content.strip())

    def test_slash_commands_come_from_name(self):
        registry = SkillRegistry()
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        for skill_md in loader.discover_skills(SkillLoader.BUNDLED_SKILL_DIR):
            registry.register(loader.load_skill(skill_md, "bundled"))
        cmds = registry.auto_commands()
        self.assertEqual(cmds["/agentica"].name, "agentica")
        self.assertEqual(cmds["/cron"].name, "cron")
        self.assertEqual(cmds["/multi-agent"].name, "multi-agent")
        self.assertEqual(cmds["/worktree"].name, "worktree")

    def test_loading_registers_exactly_these_as_bundled(self):
        """The bundled set is closed on purpose, so adding to it is a decision.

        Shipping a skill spends every user's context and cannot be uninstalled,
        so the bar is: knowledge about *agentica itself* that an install cannot
        work out on its own. General workflow skills (TDD, brainstorming, code
        review) belong in the hub or a user directory, and a capability that
        changes machine state belongs in a tool — `self_manage` stays a tool;
        the judgement about using it lives in the `agentica` skill.
        """
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        registry = SkillRegistry()
        for skill_md in loader.discover_skills(SkillLoader.BUNDLED_SKILL_DIR):
            registry.register(loader.load_skill(skill_md, "bundled"))
        self.assertEqual(sorted(s.name for s in registry.list_all()), sorted(BUNDLED))
        self.assertEqual({s.location for s in registry.list_all()}, {"bundled"})


class TestBundledSkillsArePreemptable(unittest.TestCase):
    """Shipping a skill is a default, not a decision taken away from the user."""

    def _register(self, registry, name, location):
        tmp = Path(tempfile.mkdtemp()) / name
        tmp.mkdir()
        (tmp / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: mine\n---\nmy own version",
            encoding="utf-8",
        )
        return registry.register(Skill.from_skill_md(tmp / "SKILL.md", location=location))

    def test_a_user_skill_of_the_same_name_wins(self):
        registry = SkillRegistry()
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        bundled = loader.load_skill(
            SkillLoader.BUNDLED_SKILL_DIR / "multi-agent" / "SKILL.md", "bundled",
        )
        registry.register(bundled)
        self.assertTrue(self._register(registry, "multi-agent", "user"))
        self.assertEqual(registry.get("multi-agent").location, "user")
        self.assertIn("my own version", registry.get("multi-agent").content)

    def test_a_bundled_skill_never_displaces_one_already_loaded(self):
        registry = SkillRegistry()
        self._register(registry, "multi-agent", "user")
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        bundled = loader.load_skill(
            SkillLoader.BUNDLED_SKILL_DIR / "multi-agent" / "SKILL.md", "bundled",
        )
        self.assertFalse(registry.register(bundled))
        self.assertEqual(registry.get("multi-agent").location, "user")

    def test_sdk_search_paths_omit_bundled(self):
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        locations = [loc for _, loc in loader.get_search_paths()]
        self.assertNotIn("bundled", locations)

    def test_product_search_paths_put_bundled_last(self):
        loader = SkillLoader(project_root=Path(tempfile.mkdtemp()))
        paths = loader.get_search_paths(include_system=True)
        self.assertEqual(paths[-1][1], "bundled")
        self.assertEqual(
            Path(paths[-1][0]).resolve(), SkillLoader.BUNDLED_SKILL_DIR.resolve(),
        )


class TestSystemSkillsAreProductOnly(unittest.TestCase):
    """SDK load_skills must not register or materialize bundled product skills."""

    def test_sdk_load_does_not_register_bundled_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                loader = SkillLoader(project_root=Path(tmp))
                loader.home_dir = Path(tmp)
                registry = loader.load_all(SkillRegistry())
        self.assertIsNone(registry.get("agentica"))
        self.assertIsNone(registry.get("cron"))
        self.assertIsNone(registry.get("multi-agent"))
        self.assertIsNone(registry.get("worktree"))

    def test_sdk_load_skips_leftover_system_dir(self):
        """A stale ``.system`` dir from an older CLI must not be picked up."""
        with tempfile.TemporaryDirectory() as tmp:
            leftover = Path(tmp) / ".system" / "agentica"
            leftover.mkdir(parents=True)
            (leftover / "SKILL.md").write_text(
                "---\nname: agentica\ndescription: leftover\n---\nfrom CLI\n",
                encoding="utf-8",
            )
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                loader = SkillLoader(project_root=Path(tmp))
                loader.home_dir = Path(tmp)
                registry = loader.load_all(SkillRegistry())
        self.assertIsNone(registry.get("agentica"))

    def test_load_system_skills_registers_bundled_from_package(self):
        from agentica.skills.skill_loader import load_system_skills
        from agentica.skills.skill_registry import reset_skill_registry

        with tempfile.TemporaryDirectory() as tmp:
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                reset_skill_registry()
                registry = load_system_skills(project_root=Path(tmp))
                skill = registry.get("agentica")
                self.assertIsNotNone(skill)
                self.assertEqual(skill.location, "bundled")
                self.assertEqual(
                    Path(skill.path).resolve(),
                    (SkillLoader.BUNDLED_SKILL_DIR / "agentica").resolve(),
                )
                # Nothing is materialized into $AGENTICA_HOME anymore.
                self.assertFalse((Path(tmp) / ".system").exists())

    def test_user_skill_of_the_same_name_still_wins_after_product_load(self):
        from agentica.skills.skill_loader import load_system_skills
        from agentica.skills.skill_registry import reset_skill_registry

        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "multi-agent"
            user.mkdir()
            (user / "SKILL.md").write_text(
                "---\nname: multi-agent\ndescription: mine\n---\nmy own version\n",
                encoding="utf-8",
            )
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                reset_skill_registry()
                registry = load_system_skills(project_root=Path(tmp))
            self.assertEqual(registry.get("multi-agent").location, "user")
            self.assertIn("my own version", registry.get("multi-agent").content)

    def test_skill_tool_sdk_auto_load_omits_system_skills(self):
        from agentica.skills.skill_registry import reset_skill_registry
        from agentica.tools.skill_tool import SkillTool

        with tempfile.TemporaryDirectory() as tmp:
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                reset_skill_registry()
                leftover = Path(tmp) / ".system" / "agentica"
                leftover.mkdir(parents=True)
                (leftover / "SKILL.md").write_text(
                    "---\nname: agentica\ndescription: leftover\n---\nfrom CLI\n",
                    encoding="utf-8",
                )
                tool = SkillTool(auto_load=True)
                tool.initialize()
        self.assertIsNone(tool.registry.get("agentica"))
        self.assertIsNone(tool.registry.get("cron"))
        self.assertIsNone(tool.registry.get("multi-agent"))
        self.assertIsNone(tool.registry.get("worktree"))

    def test_skill_tool_after_product_load_keeps_system_skills(self):
        """CLI create_agent preloads via load_system_skills, then SkillTool
        auto_load calls load_skills() without include_system. That must not
        drop the already-registered system skills."""
        from agentica.skills.skill_loader import load_system_skills
        from agentica.skills.skill_registry import reset_skill_registry
        from agentica.tools.skill_tool import SkillTool

        with tempfile.TemporaryDirectory() as tmp:
            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", tmp):
                reset_skill_registry()
                load_system_skills(project_root=Path(tmp))
                tool = SkillTool(auto_load=True)
                tool.initialize()
                self.assertIsNotNone(tool.registry.get("agentica"))
                self.assertEqual(tool.registry.get("agentica").location, "bundled")

class TestBundledSkillContent(unittest.TestCase):
    """These bodies are prompt text. The point of them is to send the model to
    a live source instead of to a remembered flag list, so a few load-bearing
    pointers are locked in."""

    def _body(self, name: str) -> str:
        return Skill.from_skill_md(
            SkillLoader.BUNDLED_SKILL_DIR / name / "SKILL.md", location="bundled",
        ).content

    def test_the_agentica_skill_points_at_live_sources(self):
        body = self._body("agentica")
        self.assertIn("agentica --help", body)
        self.assertIn("config.yaml", body)
        # Slash commands are typed by the user; the model cannot run them.
        self.assertIn("cannot type slash commands", body)

    def test_the_agentica_skill_sends_changes_through_the_tool(self):
        """Self-management is a capability, not a workflow: the hand is
        ``SelfManageTool`` (always mounted by the CLI), and this skill only
        carries the judgement around using it. It must not grow a copy of the
        tool's action list — that ships in the schema every turn."""
        body = self._body("agentica")
        self.assertIn("self_manage", body)
        self.assertIn("confirm=True", body)
        self.assertIn("pip", body)  # ... telling it not to roll its own
        for action in ("action='set_env'", "action='check_upgrade'"):
            self.assertNotIn(action, body)

    def test_the_multi_agent_skill_covers_all_three_mechanisms(self):
        body = self._body("multi-agent")
        for mechanism in ("`task`", "`delegate`", "tmux"):
            self.assertIn(mechanism, body)
        # Verified behaviours that are easy to get wrong.
        self.assertIn("tmux kill-session", body)
        self.assertIn("tmux attach", body)
        self.assertIn("never by session id", body)
        self.assertIn("IS your user", body)
        self.assertIn("goes back to the sender with `send_message`", body)
        self.assertIn("goes in a file and the message carries its absolute path", body)

    def test_the_cron_skill_points_at_the_tool_and_both_daemon_surfaces(self):
        body = self._body("cron")
        self.assertIn("cronjob", body)
        self.assertIn("self-contained", body)
        self.assertIn("/cron daemon on", body)
        self.assertIn("cron.enabled", body)
        self.assertIn("deployment", body)
        self.assertIn("cannot type slash commands", body)
        for action in ("action='create'", "action='list'", "action='pause'"):
            self.assertNotIn(action, body)

    def test_the_worktree_skill_forbids_execute_git_and_points_at_the_tool(self):
        body = self._body("worktree")
        self.assertIn("git worktree add", body)
        self.assertIn("worktree", body)
        self.assertIn("list_agents", body)
        self.assertIn("/worktree", body)
        self.assertIn("cannot type slash commands", body)


if __name__ == "__main__":
    unittest.main()
