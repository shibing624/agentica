# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for the rerank module.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.document import Document
from agentica.rerank.base import Rerank
import agentica.rerank.jina  # noqa: E402
import agentica.rerank.zhipuai  # noqa: E402


class TestRerankBase(unittest.TestCase):
    def test_rerank_interface(self):
        """Test that base Rerank raises NotImplementedError."""
        reranker = Rerank()
        with self.assertRaises(NotImplementedError):
            reranker.rerank(query="test", documents=[])


class TestJinaRerank(unittest.TestCase):
    @patch("agentica.rerank.jina.requests.post")
    def test_rerank_basic(self, mock_post):
        """Test JinaRerank with mocked API response."""
        from agentica.rerank.jina import JinaRerank

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.60},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        reranker = JinaRerank(api_key="fake_jina_key")
        documents = [
            Document(content="Document A"),
            Document(content="Document B"),
            Document(content="Document C"),
        ]
        result = reranker.rerank(query="test query", documents=documents)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "Document C")
        self.assertAlmostEqual(result[0].reranking_score, 0.95)
        self.assertEqual(result[1].content, "Document A")
        self.assertAlmostEqual(result[1].reranking_score, 0.80)
        self.assertEqual(result[2].content, "Document B")
        self.assertAlmostEqual(result[2].reranking_score, 0.60)

        mock_post.assert_called_once()

    @patch("agentica.rerank.jina.requests.post")
    def test_rerank_empty_documents(self, mock_post):
        """Test JinaRerank with empty document list."""
        from agentica.rerank.jina import JinaRerank

        reranker = JinaRerank(api_key="fake_jina_key")
        result = reranker.rerank(query="test", documents=[])
        self.assertEqual(result, [])
        mock_post.assert_not_called()

    @patch("agentica.rerank.jina.requests.post")
    def test_rerank_top_n(self, mock_post):
        """Test JinaRerank with top_n limit."""
        from agentica.rerank.jina import JinaRerank

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.80},
                {"index": 2, "relevance_score": 0.60},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        reranker = JinaRerank(api_key="fake_jina_key", top_n=2)
        documents = [
            Document(content="Doc A"),
            Document(content="Doc B"),
            Document(content="Doc C"),
        ]
        result = reranker.rerank(query="test", documents=documents)
        self.assertEqual(len(result), 2)


class TestZhipuAIRerank(unittest.TestCase):
    @patch("agentica.rerank.zhipuai.requests.post")
    def test_rerank_basic(self, mock_post):
        """Test ZhipuAIRerank with mocked API response."""
        from agentica.rerank.zhipuai import ZhipuAIRerank

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.90},
                {"index": 0, "relevance_score": 0.70},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        reranker = ZhipuAIRerank(api_key="fake_zhipuai_key")
        documents = [
            Document(content="First document"),
            Document(content="Second document"),
        ]
        result = reranker.rerank(query="test query", documents=documents)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Second document")
        self.assertAlmostEqual(result[0].reranking_score, 0.90)
        self.assertEqual(result[1].content, "First document")
        self.assertAlmostEqual(result[1].reranking_score, 0.70)

        mock_post.assert_called_once()

    @patch("agentica.rerank.zhipuai.requests.post")
    @patch("agentica.rerank.zhipuai.logger")
    def test_rerank_error_returns_original(self, mock_logger, mock_post):
        """Test that ZhipuAIRerank returns original docs on error."""
        from agentica.rerank.zhipuai import ZhipuAIRerank

        mock_post.side_effect = Exception("API error")

        reranker = ZhipuAIRerank(api_key="fake_key")
        documents = [
            Document(content="Doc A"),
            Document(content="Doc B"),
        ]
        result = reranker.rerank(query="test", documents=documents)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Doc A")
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
