import tempfile
import unittest
from pathlib import Path

from agentica.skills.installer import install_skills, list_installed_skills, remove_skill


class TestSkillInstaller(unittest.TestCase):
    """Install external skills from a local directory or cloned repo."""

    def test_install_skills_from_collection_directory(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_root = Path(source_dir)
            skill_dir = source_root / "skills" / "learn-from-experience"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: learn-from-experience\n"
                "description: Learn from feedback and corrections\n"
                "---\n"
                "# Learn From Experience\n",
                encoding="utf-8",
            )

            installed = install_skills(str(source_root), destination_dir=target_dir)

            self.assertEqual([skill.name for skill in installed], ["learn-from-experience"])
            self.assertTrue((Path(target_dir) / "learn-from-experience" / "SKILL.md").exists())

    def test_list_and_remove_installed_skill(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_root = Path(source_dir)
            skill_dir = source_root / "skills" / "learn-from-experience"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: learn-from-experience\n"
                "description: Learn from feedback and corrections\n"
                "---\n"
                "# Learn From Experience\n",
                encoding="utf-8",
            )

            install_skills(str(source_root), destination_dir=target_dir)
            installed = list_installed_skills(destination_dir=target_dir)
            self.assertEqual([skill.name for skill in installed], ["learn-from-experience"])

            remove_skill("learn-from-experience", destination_dir=target_dir)
            self.assertFalse((Path(target_dir) / "learn-from-experience").exists())

    def test_install_skills_force_replaces_existing_symlink(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir, tempfile.TemporaryDirectory() as linked_dir:
            source_root = Path(source_dir)
            skill_dir = source_root / "skills" / "learn-from-experience"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: learn-from-experience\n"
                "description: New version\n"
                "---\n"
                "# Learn From Experience\n",
                encoding="utf-8",
            )

            existing_dir = Path(linked_dir) / "learn-from-experience"
            existing_dir.mkdir(parents=True)
            (existing_dir / "SKILL.md").write_text(
                "---\n"
                "name: learn-from-experience\n"
                "description: Old version\n"
                "---\n"
                "# Old Skill\n",
                encoding="utf-8",
            )

            symlink_path = Path(target_dir) / "learn-from-experience"
            symlink_path.symlink_to(existing_dir, target_is_directory=True)

            replaced_symlinked_skills = []
            installed = install_skills(
                str(source_root),
                destination_dir=target_dir,
                force=True,
                replaced_symlinked_skills=replaced_symlinked_skills,
            )

            self.assertEqual([skill.name for skill in installed], ["learn-from-experience"])
            self.assertEqual(replaced_symlinked_skills, ["learn-from-experience"])
            self.assertTrue(symlink_path.exists())
            self.assertFalse(symlink_path.is_symlink())
            self.assertEqual((symlink_path / "SKILL.md").read_text(encoding="utf-8").splitlines()[2], "description: New version")

    def test_remove_skill_removes_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as target_dir, tempfile.TemporaryDirectory() as linked_dir:
            existing_dir = Path(linked_dir) / "learn-from-experience"
            existing_dir.mkdir(parents=True)
            (existing_dir / "SKILL.md").write_text(
                "---\n"
                "name: learn-from-experience\n"
                "description: Linked version\n"
                "---\n"
                "# Linked Skill\n",
                encoding="utf-8",
            )

            symlink_path = Path(target_dir) / "learn-from-experience"
            symlink_path.symlink_to(existing_dir, target_is_directory=True)

            remove_skill("learn-from-experience", destination_dir=target_dir)

            self.assertFalse(symlink_path.exists())
            self.assertTrue(existing_dir.exists())


if __name__ == "__main__":
    unittest.main()
