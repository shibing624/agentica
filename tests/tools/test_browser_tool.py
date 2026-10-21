import pytest
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PIL", reason="Browser tests require Pillow")

from agentica.tools.browser_tool import Browser, BrowserTool
from agentica.tools.video_analysis_tool import VideoAnalysisTool


class TestBrowser(unittest.TestCase):
    def test_num_history_turns_initialization(self):
        with patch("agentica.tools.browser_tool.BaseBrowser"), patch.object(
            Browser,
            "_initialize_agent",
            return_value=(MagicMock(), MagicMock()),
        ):
            toolkit = Browser(num_history_turns=7)

        self.assertEqual(toolkit.num_history_turns, 7)

    def test_inherited_models_follow_live_agent_switch(self):
        first = MagicMock(id="first")
        second = MagicMock(id="second")
        tool = BrowserTool()
        tool.set_agent_model(first)
        tool._browser = MagicMock()

        tool.set_agent_model(second)

        self.assertIs(tool.web_agent_model, second)
        self.assertIs(tool.planning_agent_model, second)
        self.assertIs(tool._browser.web_agent.model, second)
        self.assertIs(tool._browser.planning_agent.model, second)
        self.assertIs(tool._browser.browser.web_agent_model, second)

    def test_explicit_browser_models_do_not_follow_agent_switch(self):
        explicit_web = MagicMock(id="web")
        explicit_plan = MagicMock(id="plan")
        tool = BrowserTool(
            web_agent_model=explicit_web,
            planning_agent_model=explicit_plan,
        )

        tool.set_agent_model(MagicMock(id="agent"))

        self.assertIs(tool.web_agent_model, explicit_web)
        self.assertIs(tool.planning_agent_model, explicit_plan)

    def test_video_model_inherited_from_agent_keeps_following_switches(self):
        first = MagicMock(id="first")
        second = MagicMock(id="second")
        with tempfile.TemporaryDirectory() as tmp:
            tool = VideoAnalysisTool(download_directory=tmp)
            tool.set_agent_model(first)
            tool._ensure_vl_model()
            tool.set_agent_model(second)

        self.assertIs(tool.vl_model, second)
        self.assertEqual(tool.model_name, "second")


if __name__ == "__main__":
    unittest.main()
