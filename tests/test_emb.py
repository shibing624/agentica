# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
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
