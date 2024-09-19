# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use EasyOCR to extract text from images.
"""

import ssl
import json
try:
    import easyocr
except ImportError:
    raise ImportError("The `easyocr` package is not installed. Please install it via `pip install easyocr`.")

from agentica.tools.base import Toolkit
from agentica.utils.log import logger

# Ignore SSL certificate verification for HTTPS requests (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context


class OcrTool(Toolkit):
    def __init__(self, languages: list = ['ch_sim', 'en'], use_gpu: bool = False):
        """
        Initializes the OCR tool with the specified languages and GPU setting.

        :param languages: List of languages to use for OCR.
        :param use_gpu: Whether to use GPU for OCR processing.
        """
        super().__init__(name="ocr_tool")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        logger.debug(f"Initialized easyocr tool with languages: {languages} and GPU: {use_gpu}")
        self.register(self.read_text)

    def read_text(self, image_path: str, detail: int = 0) -> str:
        """
        Reads text from an image.

        :param image_path: Path to the image file.
        :param detail: Whether to return detailed information (1 for yes, 0 for no).
        :return: str, the recognized text.
        """
        try:
            result = self.reader.readtext(image_path, detail=detail)
            logger.debug(f"Recognized text: {result}, from image: {image_path}")
            if detail == 0:
                result_str = " ".join([text for text in result])
            else:
                result = [dict(text=str(text), box=str(box), prob=str(prob)) for box, text, prob in result]
                result_str = json.dumps(result, ensure_ascii=False)
            return result_str
        except Exception as e:
            logger.error(f"An error occurred while reading the text: {e}")
            return ""


if __name__ == '__main__':
    # Initialize the OCR tool
    ocr_tool = OcrTool()

    # Example usage
    # Recognize text in a Chinese image
    chinese_text = ocr_tool.read_text('../../examples/data/chinese.jpg')
    print('Chinese text:', chinese_text)

    chinese_text = ocr_tool.read_text('../../examples/data/chinese.jpg', detail=1)
    print('Chinese text:', chinese_text)
