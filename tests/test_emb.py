# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from pathlib import Path
import sys

sys.path.append('..')
from agentica.emb.zhipuai_emb import ZhipuAIEmb

pwd_path = Path(__file__).parent


class EmbTest(unittest.TestCase):
    def setUp(self):
        self.emb = ZhipuAIEmb()

    def test_emb(self):
        r = self.emb.get_embedding("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
