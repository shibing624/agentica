# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the GetUrl function in the actionflow.functions.get_url module. It checks that the URL fetching process works correctly.
"""

import shutil
from unittest.mock import patch

from actionflow.output import Output
from actionflow.tools.get_url import GetUrl


def test_execute_html():
    """
    Tests the execute method of the GetUrl function with format set to 'html'.
    """
    output = Output("test_get_url_execute_html")
    get_url = GetUrl(output)

    with patch("requests.get") as mocked_get:
        # Mock the returned response
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.text = "<html><body>Hello, world!</body></html>"

        # Execute the GetUrl function
        result = get_url.execute("http://test.com", "html")

        # Check that the returned content is correct
        assert result == "<html><body>Hello, world!</body></html>"

    # Clean up the test environment by removing the created directory
    shutil.rmtree(output.data_dir)


def test_execute_text():
    """
    Tests the execute method of the GetUrl function with format set to 'text'.
    """
    output = Output("test_get_url_execute_text")
    get_url = GetUrl(output)

    with patch("requests.get") as mocked_get:
        # Mock the returned response
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.text = "<html><body>Hello, world!</body></html>"

        # Execute the GetUrl function
        result = get_url.execute("http://test.com", "text")

        # Check that the returned content is correct
        assert result == "Hello, world!"

    # Clean up the test environment by removing the created directory
    shutil.rmtree(output.data_dir)


def test_execute_markdown():
    """
    Tests the execute method of the GetUrl function with format set to 'markdown'.
    """
    output = Output("test_get_url_execute_markdown")
    get_url = GetUrl(output)

    with patch("requests.get") as mocked_get:
        # Mock the returned response
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.text = "# Hello, world!"

        # Execute the GetUrl function
        result = get_url.execute("http://test.com", "markdown")

        # Check that the returned content is correct
        assert result == "# Hello, world!"

    # Clean up the test environment by removing the created directory
    shutil.rmtree(output.data_dir)
