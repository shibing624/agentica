# -*- coding: utf-8 -*-
"""Tests for on_user_prompt hook wiring in Runner."""
import asyncio
import unittest
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.hooks import RunHooks


class TestOnUserPromptHookDefinition(unittest.TestCase):
    """RunHooks.on_user_prompt is defined and callable."""

    def test_default_returns_none(self):
        hooks = RunHooks()

        async def _run():
            result = await hooks.on_user_prompt(agent=MagicMock(), message="hello")
            return result

        result = asyncio.run(_run())
        self.assertIsNone(result)

    def test_custom_hook_modifies_message(self):
        class MyHooks(RunHooks):
            async def on_user_prompt(self, agent: Any, message: str, **kwargs) -> Optional[str]:
                return message.upper()

        hooks = MyHooks()

        async def _run():
            result = await hooks.on_user_prompt(agent=MagicMock(), message="hello")
            return result

        result = asyncio.run(_run())
        self.assertEqual(result, "HELLO")


class TestCompositeRunHooksOnUserPrompt(unittest.TestCase):
    """_CompositeRunHooks chains on_user_prompt calls."""

    def test_chaining(self):
        from agentica.hooks import _CompositeRunHooks

        class AddPrefix(RunHooks):
            async def on_user_prompt(self, agent: Any, message: str, **kwargs) -> Optional[str]:
                return "PREFIX: " + message

        class AddSuffix(RunHooks):
            async def on_user_prompt(self, agent: Any, message: str, **kwargs) -> Optional[str]:
                return message + " :SUFFIX"

        composite = _CompositeRunHooks([AddPrefix(), AddSuffix()])

        async def _run():
            result = await composite.on_user_prompt(agent=MagicMock(), message="hello")
            return result

        result = asyncio.run(_run())
        self.assertEqual(result, "PREFIX: hello :SUFFIX")


if __name__ == "__main__":
    unittest.main()
