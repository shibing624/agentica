# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for vision messages with history.

History replay keeps a turn's text and drops its media attachments, so a
base64 image sent once is never re-encoded into later model calls.
"""
import sys
import os
import base64
import unittest
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.memory import WorkingMemory, AgentRun
from agentica.model.message import Message
from agentica.run.response import RunResponse


class TestVisionHistoryCleanup(unittest.TestCase):
    """Test cases for cleaning vision messages from history."""

    def test_clean_text_only_content(self):
        """Test that text-only content remains unchanged."""
        memory = WorkingMemory()
        
        test_msg = Message(
            role='user',
            content='Just a simple text message'
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        self.assertEqual(len(history_msgs), 1)
        self.assertEqual(history_msgs[0].content, 'Just a simple text message')

    def test_history_never_replays_attachment_fields(self):
        """Message.images/audio/videos are one-turn attachments.

        A tool-produced media message (for example analyze_image) can store a
        data URL in ``Message.images`` while its text content is only a short
        description. The next turn must keep the text and drop the attachment,
        otherwise the same payload is re-encoded into every later model call.
        """
        memory = WorkingMemory()
        msg = Message(
            role='user',
            content='Screenshot captured by tool',
            images=[{'url': 'data:image/png;base64,abc'}],
            audio={'data': 'data:audio/wav;base64,abc', 'format': 'wav'},
            videos=[{'url': 'data:video/mp4;base64,abc'}],
        )
        memory.add_run(AgentRun(response=RunResponse(messages=[msg])))

        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)

        self.assertEqual(history_msgs[0].content, 'Screenshot captured by tool')
        self.assertIsNone(history_msgs[0].images)
        self.assertIsNone(history_msgs[0].audio)
        self.assertIsNone(history_msgs[0].videos)
        # The archive keeps the original payload; cleaning happens only at the
        # history replay boundary.
        self.assertEqual(memory.runs[0].response.messages[0].images, [{'url': 'data:image/png;base64,abc'}])

    def test_compacted_runs_do_not_replay_attachment_fields(self):
        """collapse_runs() deep-copies messages, but history replay still strips media."""
        memory = WorkingMemory()
        msg = Message(role='user', content='Uploaded image', images=['/tmp/clip.png'])
        memory.collapse_runs([msg])

        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)

        self.assertEqual(history_msgs[0].content, 'Uploaded image')
        self.assertIsNone(history_msgs[0].images)
        self.assertEqual(memory.runs[0].response.messages[0].images, ['/tmp/clip.png'])


class TestAgentWithVisionHistory(unittest.TestCase):
    """Test Agent with add_history_to_context=True and vision messages.
    
    These tests use mocks to avoid actual LLM API calls.
    """

    @classmethod
    def setUpClass(cls):
        """Load test image."""
        cls.test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'chinese.jpg'
        )
        if os.path.exists(cls.test_image_path):
            with open(cls.test_image_path, 'rb') as f:
                image_data = f.read()
            cls.base64_image = base64.b64encode(image_data).decode('utf-8')
            cls.base64_image_with_prefix = f"data:image/jpeg;base64,{cls.base64_image}"
        else:
            cls.base64_image = None
            cls.base64_image_with_prefix = None

    def _create_mock_model_response(self, content: str):
        """Create a mock model response."""
        mock_response = Mock()
        mock_response.content = content
        mock_response.parsed = None
        mock_response.audio = None
        mock_response.reasoning_content = None
        mock_response.created_at = 1234567890
        mock_response.tool_calls = None
        mock_response.stop_reason = "stop"
        return mock_response

    @patch('agentica.model.openai.chat.OpenAIChat.response')
    def test_agent_vision_with_history(self, mock_response):
        """Test Agent with vision and add_history_to_context=True."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        from agentica import Agent, OpenAIChat
        
        # Setup mock
        mock_response.return_value = self._create_mock_model_response("这是一张中文图片。")
        
        # Create agent with history enabled
        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
            add_history_to_context=True,
        )
        
        # First turn: send image
        response1 = agent.run_sync(
            "这张图片里有什么？",
            images=[self.base64_image_with_prefix]
        )
        
        # Setup mock for second call
        mock_response.return_value = self._create_mock_model_response("图片中显示的是中文文字内容。")
        
        # Second turn: follow-up question (should include history)
        # This should NOT raise "Invalid chat format" error
        response2 = agent.run_sync("能详细描述一下吗？")
        
        # Verify mock was called twice
        self.assertEqual(mock_response.call_count, 2)
        
        # The follow-up request must not re-send the first turn's image payload.
        second_call_messages = mock_response.call_args_list[1][1].get('messages', [])
        self.assertTrue(second_call_messages)
        self.assertNotIn(self.base64_image, str([m.model_dump() for m in second_call_messages]))

    @patch('agentica.model.openai.chat.OpenAIChat.response')
    def test_agent_multi_turn_vision_conversation(self, mock_response):
        """Test multi-turn conversation with multiple images."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        from agentica import Agent, OpenAIChat
        
        # Create agent with history enabled
        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
            add_history_to_context=True,
        )
        
        # Turn 1: First image
        mock_response.return_value = self._create_mock_model_response("第一张图片显示中文内容。")
        agent.run_sync("描述第一张图片", images=[self.base64_image_with_prefix])
        
        # Turn 2: Second image
        mock_response.return_value = self._create_mock_model_response("第二张图片也是中文内容。")
        agent.run_sync("描述第二张图片", images=[self.base64_image_with_prefix])
        
        # Turn 3: Question without image (history should be clean)
        mock_response.return_value = self._create_mock_model_response("两张图片都包含中文文字。")
        agent.run_sync("比较一下两张图片")
        
        # Verify all calls succeeded (no exceptions)
        self.assertEqual(mock_response.call_count, 3)
        
        # Neither earlier image is re-sent with the text-only third turn.
        third_call_messages = mock_response.call_args_list[2][1].get('messages', [])
        self.assertTrue(third_call_messages)
        self.assertNotIn(self.base64_image, str([m.model_dump() for m in third_call_messages]))


if __name__ == "__main__":
    unittest.main()
