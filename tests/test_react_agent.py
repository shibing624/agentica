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

from agentica.react_agent import ReactAgent

pwd_path = Path(__file__).parent


class CalcParser(unittest.TestCase):
    def setUp(self):
        self.m = ReactAgent()

    def test_calc(self):
        r = self.m.run("一句话介绍北京")
        print(r)
        self.assertIsNotNone(r)
