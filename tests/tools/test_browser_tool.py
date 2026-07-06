import pytest
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PIL", reason="Browser tests require Pillow")

from agentica.tools.browser_tool import Browser


class TestBrowser(unittest.TestCase):
    def test_num_history_turns_initialization(self):
        with patch("agentica.tools.browser_tool.BaseBrowser"), patch.object(
            Browser,
            "_initialize_agent",
            return_value=(MagicMock(), MagicMock()),
        ):
            toolkit = Browser(num_history_turns=7)

        self.assertEqual(toolkit.num_history_turns, 7)


if __name__ == "__main__":
    unittest.main()
