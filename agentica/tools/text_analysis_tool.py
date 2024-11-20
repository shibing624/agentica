# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import Optional

from agentica.message import Message
from agentica.llm import LLM, OpenAILLM
from agentica.tools.base import Toolkit


class TextAnalysisTool(Toolkit):
    def __init__(self, llm: Optional[LLM] = None):
        super().__init__(name="text_analysis_tool")
        self.llm = llm
        self.register(self.text_analysis_use_llm)

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAILLM()

    def text_analysis_use_llm(self, prompt: str) -> str:
        """Use LLM to analyze text, including sentiment analysis, entity recognition, text summarization,
        and information extraction, etc.

        Args:
            prompt (str): The prompt to analyze, send to LLM.

        Example:
            ```python
            from agentica.tools.text_analysis_tool import TextAnalysisTool
            tool = TextAnalysisTool()
            result = tool.text_analysis_use_llm("分析下面内容的情感：这面条好难吃啊。")
            print(result)
            ```

        Returns:
            str: The result of the analysis.
        """
        self.update_llm()
        llm_messages = [Message.model_validate({"role": "user", "content": prompt})]
        return self.llm.response(llm_messages)


if __name__ == '__main__':
    # from agentica.tools.text_analysis_tool import TextAnalysisTool
    tool = TextAnalysisTool()
    result = tool.text_analysis_use_llm("分析下面内容的情感：这面条好难吃啊。")
    print(result)
