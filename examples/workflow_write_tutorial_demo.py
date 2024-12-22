# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
from textwrap import dedent
from typing import Optional, Dict, Iterator
from loguru import logger
from pathlib import Path

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica.workflow import Workflow
from agentica import RunResponse, RunEvent, SqlWorkflowStorage, pprint_run_response
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.search_exa_tool import SearchExaTool
from agentica.tools.wikipedia_tool import WikipediaTool

tutorial_dir = Path(__file__).parent.joinpath("tutorial")
tutorial_dir.mkdir(parents=True, exist_ok=True)
tutorial_file = str(tutorial_dir.joinpath("tutorial_v1.md"))


class WriteTutorialWorkflow(Workflow):
    description: str = "Generate a comprehensive technical tutorial on a given topic."

    topic_generator: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            请生成一个关于xxx的技术教程目录。
            目录结构应包括主目录和必要的二级目录，要求具体并具有实用意义。
            输出必须严格按照以下字典格式：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}, {"dir 2": ["sub dir 3", "sub dir 4"]})。
            输出必须为中文，并严格遵守格式要求，不得有多余空格或换行。
            """),
        ],
    )

    reflector: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        tools=[SearchSerperTool(), WikipediaTool(), SearchExaTool()],
        instructions=[
            dedent("""
            请根据Google搜索结果, Exa搜索结果和Wiki搜索结果，对初始生成的教程目录进行反思和优化，生成一个更为具体和实用的目录结构。
            目录结构应包括主目录和必要的二级目录，要求具体并具有实用意义。
            输出必须严格按照以下字典格式：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}, {"dir 2": ["sub dir 3", "sub dir 4"]})。
            输出必须为中文，并严格遵守格式要求，不得有多余空格或换行。
            """),
        ],
    )

    writer: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            请根据提供的教程目录编写详细的教程内容。
            使用Markdown语法对内容进行排版，提供标准代码示例和注释。
            输出必须为中文，并严格遵守格式要求，不得有多余或冗余内容。
            """),
        ],
        save_response_to_file=tutorial_file,
    )

    def run(self, topic: str) -> Iterator[RunResponse]:
        logger.info(f"Generating a tutorial on: {topic}")

        # Generate the initial tutorial directory
        topic_response = self.topic_generator.run(topic)
        topic_content = topic_response.content

        # Reflect and improve the tutorial directory
        logger.info("Reflecting on the initial tutorial directory...")
        reflection_response = self.reflector.run(topic_content)
        improved_topic_content = reflection_response.content

        # Write the detailed tutorial content
        logger.info("Writing the tutorial content...")
        writer_response = self.writer.run(improved_topic_content, stream=True)
        yield from writer_response


# Run the workflow if the script is executed directly
if __name__ == "__main__":
    from rich.prompt import Prompt

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a tutorial topic[/bold]\n✨",
        default="MySQL教程",
    )

    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the tutorial workflow
    write_tutorial_workflow = WriteTutorialWorkflow(
        session_id=f"write-tutorial-on-{url_safe_topic}",
        storage=SqlWorkflowStorage(
            table_name="write_tutorial_workflows",
            db_file="tmp/tutorial_workflows.db",
        ),
    )

    # Execute the workflow
    tutorial_stream: Iterator[RunResponse] = write_tutorial_workflow.run(topic=topic)

    # Print the response
    pprint_run_response(tutorial_stream, markdown=True)
