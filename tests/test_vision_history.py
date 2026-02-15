# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for vision messages with history.

Tests the fix for:
- "Invalid chat format. Content blocks are expected to be either text, image_url or input_audio type."
- This error occurs when base64 images are stored in database with placeholders,
  and then loaded back into history messages.
"""
import sys
import os
import base64
import unittest
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.memory import WorkingMemory, AgentRun
from agentica.model.message import Message
from agentica.run_response import RunResponse
from agentica.db.base import BASE64_PLACEHOLDER, filter_base64_media, clean_media_placeholders


class TestVisionHistoryCleanup(unittest.TestCase):
    """Test cases for cleaning vision messages from history."""

    def test_clean_image_url_with_placeholder(self):
        """Test cleaning content with image_url containing placeholder."""
        memory = WorkingMemory()
        
        # Simulate a message with image_url content block containing placeholder
        # (This is what happens after base64 image is filtered and stored in DB)
        test_msg = Message(
            role='user',
            content=[
                {'type': 'text', 'text': 'What is in this image?'},
                {'type': 'image_url', 'image_url': {'url': BASE64_PLACEHOLDER}}
            ]
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        # Get messages from history
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        # Content should be simplified to just the text
        self.assertEqual(len(history_msgs), 1)
        self.assertEqual(history_msgs[0].content, 'What is in this image?')

    def test_clean_mixed_content_with_valid_and_placeholder_urls(self):
        """Test cleaning content with both valid URLs and placeholder URLs."""
        memory = WorkingMemory()
        
        test_msg = Message(
            role='user',
            content=[
                {'type': 'text', 'text': 'Compare these images'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image1.jpg'}},
                {'type': 'image_url', 'image_url': {'url': BASE64_PLACEHOLDER}},  # Should be removed
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image2.jpg'}},
            ]
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        # Should have 3 items: text + 2 valid image_urls
        self.assertEqual(len(history_msgs), 1)
        content = history_msgs[0].content
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 3)
        
        # Verify placeholder URL is removed
        for item in content:
            if item.get('type') == 'image_url':
                url = item.get('image_url', {}).get('url', '')
                self.assertNotIn(BASE64_PLACEHOLDER, url)

    def test_clean_input_audio_with_placeholder(self):
        """Test cleaning content with input_audio containing placeholder."""
        memory = WorkingMemory()
        
        test_msg = Message(
            role='user',
            content=[
                {'type': 'text', 'text': 'What does this audio say?'},
                {'type': 'input_audio', 'input_audio': {'data': BASE64_PLACEHOLDER, 'format': 'wav'}}
            ]
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        # Content should be simplified to just the text
        self.assertEqual(len(history_msgs), 1)
        self.assertEqual(history_msgs[0].content, 'What does this audio say?')

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

    def test_clean_images_field_with_placeholder(self):
        """Test cleaning images field containing placeholders."""
        memory = WorkingMemory()
        
        test_msg = Message(
            role='user',
            content='Check this image',
            images=[BASE64_PLACEHOLDER, 'https://example.com/valid.jpg']
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        self.assertEqual(len(history_msgs), 1)
        # Placeholder should be removed from images
        if history_msgs[0].images:
            for img in history_msgs[0].images:
                self.assertNotIn(BASE64_PLACEHOLDER, str(img))


class TestFilterBase64Media(unittest.TestCase):
    """Test cases for filter_base64_media function."""

    def test_filter_base64_image(self):
        """Test filtering base64 image data."""
        data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
        result = filter_base64_media(data)
        self.assertEqual(result, BASE64_PLACEHOLDER)

    def test_filter_base64_audio(self):
        """Test filtering base64 audio data."""
        data = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
        result = filter_base64_media(data)
        self.assertEqual(result, BASE64_PLACEHOLDER)

    def test_filter_base64_video(self):
        """Test filtering base64 video data."""
        data = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
        result = filter_base64_media(data)
        self.assertEqual(result, BASE64_PLACEHOLDER)

    def test_filter_nested_dict(self):
        """Test filtering base64 in nested dict."""
        data = {
            'type': 'image_url',
            'image_url': {
                'url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
            }
        }
        result = filter_base64_media(data)
        self.assertEqual(result['image_url']['url'], BASE64_PLACEHOLDER)

    def test_filter_list(self):
        """Test filtering base64 in list."""
        data = [
            {'type': 'text', 'text': 'Hello'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='}}
        ]
        result = filter_base64_media(data)
        self.assertEqual(result[0]['text'], 'Hello')
        self.assertEqual(result[1]['image_url']['url'], BASE64_PLACEHOLDER)

    def test_preserve_url_images(self):
        """Test that URL images are preserved."""
        data = {'url': 'https://example.com/image.jpg'}
        result = filter_base64_media(data)
        self.assertEqual(result['url'], 'https://example.com/image.jpg')


class TestCleanMediaPlaceholders(unittest.TestCase):
    """Test cases for clean_media_placeholders function."""

    def test_clean_placeholder_string(self):
        """Test cleaning placeholder string."""
        result = clean_media_placeholders(BASE64_PLACEHOLDER)
        self.assertIsNone(result)

    def test_clean_dict_with_placeholder(self):
        """Test cleaning dict containing placeholder.
        
        Note: clean_media_placeholders removes the key with placeholder value,
        but keeps the dict structure. The image_url key is removed.
        """
        data = {
            'type': 'image_url',
            'image_url': {'url': BASE64_PLACEHOLDER}
        }
        result = clean_media_placeholders(data)
        # The image_url key should be removed (or its url value is None)
        # The function removes keys with None values after cleaning
        self.assertIsNotNone(result)
        # The 'type' key should remain
        self.assertEqual(result.get('type'), 'image_url')
        # The 'image_url' key should be removed or empty
        self.assertNotIn('image_url', result)

    def test_clean_list_with_placeholder(self):
        """Test cleaning list containing placeholder items.
        
        Note: clean_media_placeholders doesn't understand image_url structure,
        it just removes items that are exactly the placeholder or contain it.
        The actual cleaning of image_url blocks is done in clean_message_for_history.
        """
        data = [
            {'type': 'text', 'text': 'Hello'},
            {'type': 'image_url', 'image_url': {'url': BASE64_PLACEHOLDER}}
        ]
        result = clean_media_placeholders(data)
        # clean_media_placeholders keeps the dict but removes the url
        # The actual filtering of image_url blocks is done in clean_message_for_history
        self.assertIsNotNone(result)
        self.assertEqual(result[0]['text'], 'Hello')


class TestVisionWithLocalImage(unittest.TestCase):
    """Test cases for vision with local image file."""

    @classmethod
    def setUpClass(cls):
        """Load test image."""
        cls.test_image_path = os.path.join(
            os.path.dirname(__file__), 'data', 'chinese.jpg'
        )
        if os.path.exists(cls.test_image_path):
            with open(cls.test_image_path, 'rb') as f:
                image_data = f.read()
            cls.base64_image = base64.b64encode(image_data).decode('utf-8')
            cls.base64_image_with_prefix = f"data:image/jpeg;base64,{cls.base64_image}"
        else:
            cls.base64_image = None
            cls.base64_image_with_prefix = None

    def test_filter_local_base64_image(self):
        """Test filtering local base64 image."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        result = filter_base64_media(self.base64_image_with_prefix)
        self.assertEqual(result, BASE64_PLACEHOLDER)

    def test_filter_message_with_local_image(self):
        """Test filtering message content with local base64 image."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        content = [
            {'type': 'text', 'text': '这张图片里有什么？'},
            {'type': 'image_url', 'image_url': {'url': self.base64_image_with_prefix}}
        ]
        
        result = filter_base64_media(content)
        
        # Text should be preserved
        self.assertEqual(result[0]['text'], '这张图片里有什么？')
        # Base64 image should be replaced with placeholder
        self.assertEqual(result[1]['image_url']['url'], BASE64_PLACEHOLDER)

    def test_clean_history_with_local_image(self):
        """Test cleaning history messages containing local base64 image."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        memory = WorkingMemory()
        
        # Simulate a message after base64 filtering (as stored in DB)
        filtered_content = filter_base64_media([
            {'type': 'text', 'text': '这张图片里有什么？'},
            {'type': 'image_url', 'image_url': {'url': self.base64_image_with_prefix}}
        ])
        
        test_msg = Message(
            role='user',
            content=filtered_content
        )
        run = AgentRun(
            message=test_msg,
            response=RunResponse(messages=[test_msg])
        )
        memory.add_run(run)
        
        # Get cleaned history messages
        history_msgs = memory.get_messages_from_last_n_runs(last_n=1)
        
        # Content should be simplified to just the text
        self.assertEqual(len(history_msgs), 1)
        self.assertEqual(history_msgs[0].content, '这张图片里有什么？')

    def test_multi_turn_conversation_with_images(self):
        """Test multi-turn conversation with images in history."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        memory = WorkingMemory()
        
        # First turn: user sends image
        filtered_content_1 = filter_base64_media([
            {'type': 'text', 'text': '这是什么？'},
            {'type': 'image_url', 'image_url': {'url': self.base64_image_with_prefix}}
        ])
        msg1 = Message(role='user', content=filtered_content_1)
        resp1 = Message(role='assistant', content='这是一张中文图片。')
        run1 = AgentRun(
            message=msg1,
            response=RunResponse(messages=[msg1, resp1])
        )
        memory.add_run(run1)
        
        # Second turn: user asks follow-up
        msg2 = Message(role='user', content='能详细描述一下吗？')
        resp2 = Message(role='assistant', content='图片中显示的是中文文字。')
        run2 = AgentRun(
            message=msg2,
            response=RunResponse(messages=[msg2, resp2])
        )
        memory.add_run(run2)
        
        # Get all history messages
        history_msgs = memory.get_messages_from_last_n_runs()
        
        # All messages should be valid (no placeholder URLs)
        for msg in history_msgs:
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        self.assertNotIn(BASE64_PLACEHOLDER, url,
                            f"Found placeholder in history: {url}")
            elif isinstance(content, str):
                self.assertNotIn(BASE64_PLACEHOLDER, content,
                    f"Found placeholder in content: {content}")


class TestAgentWithVisionHistory(unittest.TestCase):
    """Test Agent with add_history_to_messages=True and vision messages.
    
    These tests use mocks to avoid actual LLM API calls.
    """

    @classmethod
    def setUpClass(cls):
        """Load test image."""
        cls.test_image_path = os.path.join(
            os.path.dirname(__file__), 'data', 'chinese.jpg'
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
        """Test Agent with vision and add_history_to_messages=True."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        from agentica import Agent, OpenAIChat
        
        # Setup mock
        mock_response.return_value = self._create_mock_model_response("这是一张中文图片。")
        
        # Create agent with history enabled
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            add_history_to_messages=True,
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
        
        # Verify second call's messages don't contain placeholder URLs
        second_call_args = mock_response.call_args_list[1]
        messages = second_call_args[1].get('messages', [])
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        self.assertNotIn(BASE64_PLACEHOLDER, url,
                            f"Found placeholder in API call: {url}")

    @patch('agentica.model.openai.chat.OpenAIChat.response')
    def test_agent_multi_turn_vision_conversation(self, mock_response):
        """Test multi-turn conversation with multiple images."""
        if self.base64_image_with_prefix is None:
            self.skipTest("Test image not found")
        
        from agentica import Agent, OpenAIChat
        
        # Create agent with history enabled
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            add_history_to_messages=True,
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
        
        # Check third call's messages for placeholders
        third_call_args = mock_response.call_args_list[2]
        messages = third_call_args[1].get('messages', [])
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        self.assertNotIn(BASE64_PLACEHOLDER, url,
                            "Found placeholder in multi-turn conversation")

    @patch('agentica.model.openai.chat.OpenAIChat.response')
    def test_agent_url_images_preserved_in_history(self, mock_response):
        """Test that URL-based images are preserved in history.
        
        Note: URL images are converted to image_url format in content,
        but they should NOT be filtered out (only base64 images are filtered).
        """
        from agentica import Agent, OpenAIChat
        
        url_image = "https://example.com/test-image.jpg"
        
        # Create agent with history enabled
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            add_history_to_messages=True,
        )
        
        # Turn 1: URL image
        mock_response.return_value = self._create_mock_model_response("这是一张网络图片。")
        agent.run_sync("描述这张图片", images=[url_image])
        
        # Turn 2: Follow-up
        mock_response.return_value = self._create_mock_model_response("图片内容很丰富。")
        agent.run_sync("还有什么细节？")
        
        # Verify second call succeeded without errors
        self.assertEqual(mock_response.call_count, 2)
        
        # Verify no placeholder URLs in history
        second_call_args = mock_response.call_args_list[1]
        messages = second_call_args[1].get('messages', [])
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        # URL images should NOT have placeholder
                        self.assertNotIn(BASE64_PLACEHOLDER, url,
                            "URL image should not be replaced with placeholder")


if __name__ == "__main__":
    unittest.main()
