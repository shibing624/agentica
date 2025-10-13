# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Jina 有2个功能：
1）Jina Reader, 从URL获取内容，输出markdown格式的网页内容
2）Jina Search, 输出查询query，结合Reader的检索结果，输出markdown格式的检索网页内容，用于LLM做RAG。
    如：苹果的最新产品是啥？ -> ## 苹果的最新产品是Apple Intelligence，iOS 18。
"""
import hashlib
import os
import requests
import json
from os import getenv
from urllib.parse import urlparse
from typing import Optional, cast

from agentica.model.base import Model
from agentica.model.openai.chat import OpenAIChat
from agentica.tools.base import Tool
from agentica.utils.log import logger

EXTRACT_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**

json的value值的输出语言需要跟【Webpage Content】一致。
"""


class JinaTool(Tool):
    def __init__(
            self,
            api_key: Optional[str] = None,
            jina_reader: bool = True,
            jina_search: bool = False,
            jina_reader_by_goal: bool = True,
            work_dir: str = None,
            llm: Optional[Model] = None,
            extract_prompt: str = EXTRACT_PROMPT,
            model_name: str = "gpt-4o-mini"
    ):
        super().__init__(name="jina_tool")
        self.api_key = api_key or getenv("JINA_API_KEY")
        self.work_dir = work_dir or os.path.curdir
        self.llm = llm
        self.model_name = model_name
        self.extract_prompt = extract_prompt
        if self.api_key:
            logger.debug(f"Use JINA_API_KEY: {'*' * 10 + self.api_key[-4:]}")

        if jina_reader:
            self.register(self.jina_url_reader)
        if jina_reader_by_goal:
            self.register(self.jina_url_reader_by_goal)
        if jina_search:
            self.register(self.jina_search)

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAIChat(id=self.model_name)

    def _get_headers(self) -> dict:
        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _trim_content(self, content: str, limit_len: int = 8000) -> str:
        """Trim to the maximum allowed length."""
        if len(content) > limit_len:
            truncated = content[: limit_len]
            return truncated + "... (content truncated)"
        return content

    @staticmethod
    def _generate_file_name_from_url(url: str, max_length=32) -> str:
        """Get file name with hash suffix"""
        url_bytes = url.encode("utf-8")
        hash = hashlib.blake2b(url_bytes).hexdigest()[:16]
        parsed_url = urlparse(url)
        prefix = parsed_url.netloc
        file_name = f"{prefix}_{hash}"
        if len(file_name) > max_length:
            file_name = file_name[:max_length]
        return file_name

    def jina_url_reader(self, url: str, limit_len: int = 8000) -> str:
        """Reads a URL and returns the html text content using Jina Reader API.

        Args:
            url: str, The URL to read.
            limit_len: int, url page content limit char length, default: 8000

        Example:
            from agentica.tools.jina_tool import JinaTool
            m = JinaTool(jina_reader=True, jina_search=True)
            url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
            r = m.jina_url_reader(url)
            print(url, '\n\n', r)

        Returns:
            str, The result of the crawling html text content, Markdown format.
        """
        logger.debug(f"Reading URL: {url}")

        try:
            data = {'url': url}
            response = requests.post('https://r.jina.ai/', headers=self._get_headers(), json=data)
            response.raise_for_status()
            content = response.text
            result = self._trim_content(content, limit_len)
        except Exception as e:
            msg = f"Error reading URL: {str(e)}"
            logger.error(msg)
            result = msg
            content = ''
        if content:
            filename = self._generate_file_name_from_url(url)
            save_path = os.path.realpath(os.path.join(str(self.work_dir), filename))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Url: {url}, saved content to: {save_path}")
        return result

    def jina_search(self, query: str) -> str:
        """Performs a web search using Jina Search API and returns the search content.

        Args:
            query (str): The query to search for.

        Example:
            from agentica.tools.jina_tool import JinaTool
            m = JinaTool(jina_reader=True, jina_search=True)
            query = "苹果的最新产品是啥？"
            r = m.jina_search(query)
            print(query, '\n\n', r)

        Returns:
            str: The results of the search.
        """
        query = query.strip()
        logger.debug(f"Search query: {query}")
        url = f'https://s.jina.ai/{query}'
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            content = response.text
            result = self._trim_content(content)
        except Exception as e:
            content = ''
            msg = f"Error performing search: {str(e)}"
            logger.error(msg)
            result = msg
        if content:
            filename = self._generate_file_name_from_url(url)
            save_path = os.path.realpath(os.path.join(str(self.work_dir), filename))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Query: {query}, saved content to: {save_path}")
        return result

    def jina_url_reader_by_goal(self, url: str, goal: str) -> str:
        """
        Read webpage content and extract information based on a user goal using LLM.

        Args:
            url: The URL of the webpage to read.
            goal: The user's goal or question for extracting information from the page.

        Returns:
            str: Extracted information or an error message.
        """
        self.update_llm()
        content = self.jina_url_reader(url, 95000)

        if content and not content.startswith("Error reading URL"):
            messages = [{"role": "user", "content": self.extract_prompt.format(webpage_content=content, goal=goal)}]
            self.llm = cast(Model, self.llm)
            response = self.llm.get_client().chat.completions.create(
                model=self.model_name, messages=messages
            )
            raw = response.choices[0].message.content
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] "
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                logger.debug(status_msg)
                content = content[:truncate_length]
                extraction_prompt = self.extract_prompt.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                response = self.llm.get_client().chat.completions.create(
                    model=self.model_name, messages=messages
                )
                raw = response.choices[0].message.content
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    response = self.llm.get_client().chat.completions.create(
                        model=self.model_name, messages=messages
                    )
                    raw = response.choices[0].message.content
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                logger.warning("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            return useful_information
        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url,
                                                                                                                goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information


if __name__ == '__main__':
    os.environ["JINA_API_KEY"] = ''
    m = JinaTool()
    url = "https://raw.githubusercontent.com/shibing624/agentica/refs/heads/main/agentica/tools/base.py"
    r = m.jina_url_reader(url)
    print(r)

    url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    r = m.jina_url_reader(url)
    print(url, '\n\n', r)

    query = "苹果的最新产品是啥？"
    r = m.jina_search(query)
    print(query, '\n\n', r)

    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    goal = "Explain the history of artificial intelligence."
    print(m.jina_url_reader_by_goal(url, goal))

    url = 'https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0'
    goal = "中国政府将如何应对经济增长"
    print(m.jina_url_reader_by_goal(url, goal))
