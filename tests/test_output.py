# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the Output class.
"""

import json
import os
import shutil

from actionflow.output import Output


def test_init():
    """
    Tests the initialization of the Output object by checking if the output directory is created.
    """
    output = Output("test_output_init")
    assert os.path.exists(output.data_dir), "Output path does not exist"
    os.rmdir(output.data_dir)


def test_save():
    """
    Tests the save method of the Output object by saving a text file and a JSON file,
    and then checking if these files exist in the output directory.
    """
    output = Output("test")

    # Test saving a text file
    txt_file_path = os.path.join(output.data_dir, "test.txt")
    output.save("test.txt", "test_output_save")
    assert os.path.exists(txt_file_path), "Text file was not saved correctly"
    with open(txt_file_path, "r") as f:
        assert f.read() == "test_output_save", "Text file content is incorrect"
    os.remove(txt_file_path)

    # Test saving a JSON file
    json_file_path = os.path.join(output.data_dir, "test.json")
    output.save("test.json", [{"test": "test1"}, {"test": "test2"}])
    assert os.path.exists(json_file_path), "JSON file was not saved correctly"
    with open(json_file_path, "r") as f:
        assert json.load(f) == [
            {"test": "test1"},
            {"test": "test2"},
        ], "JSON file content is incorrect"

    shutil.rmtree(output.data_dir)
