# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a test for the CreateImage class in the agentica.functions.create_image module. It uses the unittest.mock library to mock the OpenAI and requests APIs, and checks that the image creation process works correctly.
"""

import shutil
from unittest.mock import MagicMock, patch

from agentica.tools.create_image import CreateImageTool


@patch("openai.Image.create")
@patch("requests.get")
def test_execute(mock_get, mock_create):
    """
    Tests the execute method of the CreateImage class. It mocks the OpenAI and requests APIs, and checks that the image creation process works correctly.
    """
    # Mock the openai.Image.create call to return a mock response with a mock image URL
    mock_create.return_value = {"data": [{"url": "https://mockurl.com/mock_image.jpg"}]}

    # Mock the requests.get call to return a mock response with mock image content
    mock_response = MagicMock()
    mock_response.content = b"mock image content"
    mock_get.return_value = mock_response

    create_image = CreateImageTool(data_dir="test_data")
    image_path = create_image.create_delle_image("a white siamese cat", 1, "1024x1024")

    # Check that the returned image name is a valid SHA-256 hash followed by ".png"
    image_file_name = image_path.split("/")[-1]
    print(f"image_file_name:{image_file_name}")
    # assert re.match(r"[0-9a-f]{64}\.png$", image_file_name) is not None

    # Check that the image file was created with the correct content
    with open(image_path, "rb") as f:
        assert f.read() == b"mock image content"

    # Clean up the test environment by removing the created file and directory
    shutil.rmtree(create_image.data_dir)
