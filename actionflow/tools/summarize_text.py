# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a class for summarizing text.
"""
import os
from typing import Dict, List

from actionflow.llm import LLM, Settings
from actionflow.output import Output
from actionflow.tool import BaseTool


class SummarizeText(BaseTool):
    """
    This class inherits from the BaseFunction class. It defines a function for summarizing text.
    """

    def __init__(self, output: Output, max_input_chars: int = 4000):
        """
        Initializes the SummarizeText object.
        """
        super().__init__(output)
        self.default_instructions = "Return a summary that succinctly captures its main points."
        # See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        self.max_input_chars = max_input_chars  # To allow room for the instructions and summary

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",
            "function": {
                "name": "summarize_text",
                "description": "Summarizes text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_to_summarize": {
                            "type": "string",
                            "description": "The text to summarize.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Instructions for summarizing the text.",
                            "default": self.default_instructions,
                        }
                    },
                    "required": ["text_to_summarize"],
                },
            }
        }

    def execute(self, text_to_summarize: str, instructions: str = None) -> str:
        """
        Summarizes text.

        :param text_to_summarize: The text to summarize.
        :type text_to_summarize: str
        :param instructions: Optional instructions for summarizing the text. Defaults to default instructions.
        :type instructions: str
        :return: The summary of the text.
        :rtype: str
        """
        truncated_text = self._truncate_text(text_to_summarize)
        messages = self._prepare_messages(truncated_text, instructions)
        return self._summarize(messages)

    def _truncate_text(self, text: str) -> str:
        """
        Truncates text.

        :param text: The text to truncate.
        :type text: str
        :return: The truncated text.
        :rtype: str
        """
        return text[: self.max_input_chars]

    def _prepare_messages(self, truncated_text: str, instructions: str) -> List[Dict[str, str]]:
        """
        Prepares messages for the language model.

        :param truncated_text: The text to summarize.
        :type truncated_text: str
        :param instructions: Instructions for summarizing the text.
        :type instructions: str
        :return: The messages for the language model.
        :rtype: list[dict[str, str  ]]
        """
        system_content = f"You are an AI summarizer. {instructions}"
        user_content = f"Text to summarize: {truncated_text}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _summarize(
            self,
            messages: List[Dict[str, str]],
            max_return_tokens: int = 1000,
    ) -> str:
        """
        Summarizes text.
        :param messages: The messages for the language model.
        :type messages: list[dict[str, str]]
        :param max_return_tokens: The maximum number of tokens to return.
        :type max_return_tokens: int
        """
        settings = Settings(
            max_tokens=max_return_tokens,
            temperature=0,
        )
        summary_message = LLM().respond(settings, messages)
        return summary_message.content


if __name__ == '__main__':
    output = Output('o')
    summarizer = SummarizeText(output)
    text = "a" * 110
    r = summarizer.execute(text)
    print(text, '\n\n', r)
    os.removedirs('o')
