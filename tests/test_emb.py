# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.emb.zhipuai_emb import ZhipuAIEmb

pwd_path = Path(__file__).parent


class EmbTest(unittest.TestCase):
    def setUp(self):
        self.emb = ZhipuAIEmb()

    def test_emb(self):
        r = self.emb.get_embedding("hi")
        print(r)
        self.assertIsNotNone(r)
