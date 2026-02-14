# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for embedding modules.
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.emb.zhipuai_emb import ZhipuAIEmb

pwd_path = Path(__file__).parent


class EmbTest(unittest.TestCase):

    @patch('agentica.emb.openai_emb.OpenAIClient')
    def test_emb(self, mock_openai_client):
        """Test embedding with mocked ZhipuAI API."""
        # Mock the OpenAI client and its embeddings.create method
        mock_client_instance = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 100)]
        mock_client_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai_client.return_value = mock_client_instance

        emb = ZhipuAIEmb(api_key="fake_zhipuai_api_key")
        r = emb.get_embedding("hi")
        print(r)
        self.assertIsNotNone(r)
        self.assertEqual(len(r), 300)


class TestHttpEmb(unittest.TestCase):

    @patch("agentica.emb.http_emb.requests.post")
    def test_get_embedding(self, mock_post):
        """Test HttpEmb get_embedding with mocked HTTP response."""
        from agentica.emb.http_emb import HttpEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = HttpEmb(api_url="http://localhost:8080/v1/embeddings", model="test-model")
        result = emb.get_embedding("hello world")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
        call_json = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json", {}))
        self.assertEqual(call_json["input"], ["hello world"])
        self.assertEqual(call_json["model"], "test-model")

    @patch("agentica.emb.http_emb.requests.post")
    def test_get_embedding_and_usage(self, mock_post):
        """Test HttpEmb get_embedding_and_usage."""
        from agentica.emb.http_emb import HttpEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5, 0.6], "index": 0}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = HttpEmb()
        embedding, usage = emb.get_embedding_and_usage("test")
        self.assertEqual(embedding, [0.5, 0.6])
        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 3)

    @patch("agentica.emb.http_emb.requests.post")
    def test_get_embeddings_batch(self, mock_post):
        """Test HttpEmb get_embeddings with multiple texts."""
        from agentica.emb.http_emb import HttpEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = HttpEmb()
        results = emb.get_embeddings(["hello", "world"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [0.1, 0.2])
        self.assertEqual(results[1], [0.3, 0.4])

    @patch("agentica.emb.http_emb.requests.post")
    def test_get_embedding_with_api_key(self, mock_post):
        """Test HttpEmb includes Authorization header when api_key is set."""
        from agentica.emb.http_emb import HttpEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1], "index": 0}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = HttpEmb(api_key="test-key-123")
        emb.get_embedding("test")

        call_headers = mock_post.call_args.kwargs.get("headers", mock_post.call_args[1].get("headers", {}))
        self.assertIn("Authorization", call_headers)
        self.assertEqual(call_headers["Authorization"], "Bearer test-key-123")


class TestJinaEmb(unittest.TestCase):

    @patch("agentica.emb.jina_emb.requests.post")
    def test_get_embedding(self, mock_post):
        """Test JinaEmb get_embedding with mocked API."""
        from agentica.emb.jina_emb import JinaEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = JinaEmb(api_key="fake_jina_key")
        result = emb.get_embedding("hello")

        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])
        mock_post.assert_called_once()

    @patch("agentica.emb.jina_emb.requests.post")
    def test_get_embeddings_batch(self, mock_post):
        """Test JinaEmb batch embeddings."""
        from agentica.emb.jina_emb import JinaEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = JinaEmb(api_key="fake_jina_key")
        results = emb.get_embeddings(["hello", "world"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [0.1, 0.2])
        self.assertEqual(results[1], [0.3, 0.4])

    @patch("agentica.emb.jina_emb.requests.post")
    def test_get_embedding_error(self, mock_post):
        """Test JinaEmb returns empty list on error."""
        from agentica.emb.jina_emb import JinaEmb

        mock_post.side_effect = Exception("API error")

        emb = JinaEmb(api_key="fake_key")
        result = emb.get_embedding("test")
        self.assertEqual(result, [])


class TestMulanAIEmb(unittest.TestCase):

    @patch("agentica.emb.mulanai_emb.requests.post")
    def test_get_embedding(self, mock_post):
        """Test MulanAIEmb get_embedding with mocked API."""
        from agentica.emb.mulanai_emb import MulanAIEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [[0.1, 0.2, 0.3]],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = MulanAIEmb(api_key="bGlsaTphYmMxMjM=")
        result = emb.get_embedding("hello")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
        # Verify the payload uses "sentences" key
        call_json = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json", {}))
        self.assertIn("sentences", call_json)
        self.assertEqual(call_json["sentences"], ["hello"])

    @patch("agentica.emb.mulanai_emb.requests.post")
    def test_get_embeddings_batch(self, mock_post):
        """Test MulanAIEmb batch embeddings."""
        from agentica.emb.mulanai_emb import MulanAIEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = MulanAIEmb(api_key="bGlsaTphYmMxMjM=")
        results = emb.get_embeddings(["hello", "world"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [0.1, 0.2])
        self.assertEqual(results[1], [0.3, 0.4])

    @patch("agentica.emb.mulanai_emb.requests.post")
    def test_auth_header(self, mock_post):
        """Test MulanAIEmb includes correct auth header."""
        from agentica.emb.mulanai_emb import MulanAIEmb

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [[0.1]]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        emb = MulanAIEmb(api_key="bGlsaTphYmMxMjM=")
        emb.get_embedding("test")

        call_headers = mock_post.call_args.kwargs.get("headers", mock_post.call_args[1].get("headers", {}))
        self.assertEqual(call_headers["Authorization"], "Bearer bGlsaTphYmMxMjM=")


if __name__ == "__main__":
    unittest.main()
