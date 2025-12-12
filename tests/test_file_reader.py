# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.utils.file_parser import (
    read_json_file,
    read_csv_file,
    read_txt_file,
    read_pdf_file,
    read_pdf_url,
    read_docx_file,
    read_excel_file,
)

pwd_path = Path(__file__).parent


class TestFileParser(unittest.TestCase):

    def test_read_json_file(self):
        path = Path(os.path.join(pwd_path, "data/mbpp.jsonl"))
        result = read_json_file(path)
        print(result[:100])
        self.assertIsNotNone(result)

    @patch("builtins.open", new_callable=mock_open, read_data="col1,col2\nval1,val2\nval3,val4")
    def test_read_csv_file(self, mock_file):
        path = Path("test.csv")
        result = read_csv_file(path)
        print(result)
        expected = json.dumps([{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}], ensure_ascii=False)
        self.assertEqual(result, expected)

    def test_read_txt_file(self):
        path = os.path.join(pwd_path, "data/news_docs.txt")
        result = read_txt_file(Path(path))
        print(result[:100])
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
