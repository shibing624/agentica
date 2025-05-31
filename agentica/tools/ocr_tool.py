# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use EasyOCR to extract text from images.
"""
try:
    from imgocr import ImgOcr
except ImportError:
    raise ImportError("The `imgocr` package is not installed. Please install it via `pip install imgocr`.")

from agentica.tools.base import Tool
from agentica.utils.log import logger


class OcrTool(Tool):
    def __init__(self, use_gpu: bool = False):
        """
        Initializes the OCR tool with the specified languages and GPU setting.

        :param languages: List of languages to use for OCR.
        :param use_gpu: Whether to use GPU for OCR processing.
        """
        super().__init__(name="ocr_tool")
        self.ocr_model = ImgOcr(use_gpu=use_gpu)
        logger.debug(f"Initialized imgocr tool, use GPU: {use_gpu}")
        self.register(self.read_text)

    def read_text(self, image_path: str) -> str:
        """Reads text from an image.

        Args:
            image_path (str): Path to the image file.

        Example:
            from agentica.tools.ocr_tool import OcrTool
            ocr_tool = OcrTool()
            text = ocr_tool.read_text('../../examples/data/chinese.jpg')
            print('Text:', text)

        Returns:
            str: The recognized text.
        """
        try:
            result = self.ocr_model.ocr(image_path)
            result_str = " ".join([i['text'] for i in result])
            logger.info(f"Ocr image: {image_path}, result: {result_str}")
            return result_str
        except Exception as e:
            logger.error(f"An error occurred while reading the text: {e}")
            return ""


if __name__ == '__main__':
    # Initialize the OCR tool
    ocr_tool = OcrTool()
    chinese_text = ocr_tool.read_text('../../examples/data/chinese.jpg')
    print('Chinese text:', chinese_text)
