# -*- coding: utf-8 -*-
"""Unit tests for the CHAT log level on the agentica logger."""
import logging
import unittest
from io import StringIO

from agentica.utils.log import logger, CHAT_LEVEL, _PlainLoguruStyleFormatter


class TestChatLogLevel(unittest.TestCase):
    def test_chat_level_value(self):
        # Between INFO (20) and WARNING (30), conventionally 25.
        self.assertEqual(CHAT_LEVEL, 25)

    def test_chat_level_registered(self):
        self.assertEqual(logging.getLevelName(CHAT_LEVEL), "CHAT")

    def test_logger_has_chat_method(self):
        self.assertTrue(hasattr(logger, "chat"))
        self.assertTrue(callable(logger.chat))

    def test_chat_message_emitted(self):
        # Capture chat output via a temporary handler.
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(CHAT_LEVEL)
        handler.setFormatter(_PlainLoguruStyleFormatter())
        logger.addHandler(handler)
        prev_level = logger.level
        logger.setLevel(CHAT_LEVEL)
        try:
            logger.chat("agent_a -> agent_b: hello")
            handler.flush()
            output = buf.getvalue()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
        self.assertIn("CHAT", output)
        self.assertIn("agent_a -> agent_b: hello", output)


if __name__ == "__main__":
    unittest.main()
