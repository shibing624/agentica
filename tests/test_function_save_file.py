# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a test for the SaveFile class in the actionflow.functions.save_file module. It checks that the file saving process works correctly.
"""

import shutil

from actionflow.output import Output
from actionflow.tools.save_file import SaveFile


def test_execute():
    """
    Tests the execute method of the SaveFile class. It checks that the file saving process works correctly.
    """
    output = Output("test_save_file_execute")
    save_file = SaveFile(output)
    result = save_file.execute("test.txt", "Hello, world!")
    print("result:", result)
    # Check that the returned file path is correct
    assert result.split('/')[-1] == f"test.txt"

    # Clean up the test environment by removing the created file and directory
    shutil.rmtree(output.data_dir)
