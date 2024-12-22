# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
from textwrap import dedent
from loguru import logger
from typing import Iterator
from pydantic import BaseModel, Field

sys.path.append('..')
from agentica import Agent, OpenAIChat, AzureOpenAIChat
from agentica import Workflow, RunResponse, RunEvent, SqlWorkflowStorage, pprint_run_response
from agentica.tools.search_serper_tool import SearchSerperTool


class WriteNovelWorkflow(Workflow):
    description: str = "Generate a comprehensive novel on a given topic."

    outlines: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            ## context
            You need to generate a novel named 'xxx'.
            Be creative and thorough while generating the outline. The output should be formatted as specified below.

            ## format example
            [CONTENT]
            {
                "name": "Novel Name",
                "user_group": "Target User Group",
                "outlines": [
                    "Chapter 1: ...",
                    "Chapter 2: ...",
                    "Chapter 3: ..."
                ],
                "background": "...",
                "character_names": [
                    "Character1",
                    "Character2",
                    "Character3"
                ],
                "conflict": "...",
                "plot": "...",
                "ending": "..."
            }
            [/CONTENT]

            ## nodes: "<node>: <type>  # <instruction>"
            - name: <class 'str'>  # The name of the novel.
            - user_group: <class 'str'>  # The target audience for the novel.
            - outlines: typing.List[str]  # The outlines of the novel. No more than 10 chapters.
            - background: <class 'str'>  # The background setting of the novel.
            - character_names: typing.List[str]  # List of main characters in the novel.
            - conflict: <class 'str'>  # The primary conflict in the novel.
            - plot: <class 'str'>  # Summary of the novel's plot.
            - ending: <class 'str'>  # The novel's ending.

            ## constraint
            Language: The output should be in the language of this prompt.
            Format: Ensure the output is wrapped inside [CONTENT][/CONTENT] tags like the format example above.

            ## action
            Generate the outlines for the novel and ensure it follows the formatting guidelines.
            """),
        ],
    )

    reflection: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        tools=[SearchSerperTool()],
        instructions=[
            dedent("""
            ## context
            Review the previously generated novel outlines and use Google search results to reflect and improve upon them.

            ## format example
            [CONTENT]
            {
                "name": "Novel Name",
                "user_group": "Target User Group",
                "outlines": [
                    "Chapter 1: ...",
                    "Chapter 2: ...",
                    "Chapter 3: ..."
                ],
                "background": "...",
                "character_names": [
                    "Character1",
                    "Character2",
                    "Character3"
                ],
                "conflict": "...",
                "plot": "...",
                "ending": "..."
            }
            [/CONTENT]

            ## nodes: "<node>: <type>  # <instruction>"
            - name: <class 'str'>  # The name of the novel.
            - user_group: <class 'str'>  # The target audience for the novel.
            - outlines: typing.List[str]  # The outlines of the novel. No more than 10 chapters.
            - background: <class 'str'>  # The background setting of the novel.
            - character_names: typing.List[str]  # List of main characters in the novel.
            - conflict: <class 'str'>  # The primary conflict in the novel.
            - plot: <class 'str'>  # Summary of the novel's plot.
            - ending: <class 'str'>  # The novel's ending.

            ## constraint
            Language: The output should be in the language of this prompt.
            Format: Ensure the output is wrapped inside [CONTENT][/CONTENT] tags like the format example above.

            ## action
            搜索时，可以适当调整搜索关键词，以获取更多相关信息，写小说内容可以搜索一些小说写作技巧。
            Conduct searches using the provided tools to gather information and insights. Reflect on the gathered results and improve the original outlines accordingly.
            """),
        ],
    )

    writer: Agent = Agent(
        model=AzureOpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            ## context
            You need to write detailed content for the novel, based on the generated outlines.

            ## output format example
            输出markdown格式的小说，章节内容结构清晰。不用给```markdown```标记，直接输出md格式的内容。用中文写。

            ## action
            Write the detailed novel chapters and ensure they follow the formatting guidelines.
            """),
        ],
    )

    def run(self, topic: str):
        logger.info(f"Generating a novel on: {topic}")

        # Generate the initial outlines
        outlines_response = self.outlines.run(topic)
        if outlines_response is None or not outlines_response.content:
            yield RunResponse(run_id=self.run_id, content="Sorry, could not generate the novel outlines.")
            return

        logger.info(f"Reflecting and improving the novel outlines on: {topic}")
        reflection_response = self.reflection.run(outlines_response.content)
        if reflection_response is None or not reflection_response.content:
            yield RunResponse(run_id=self.run_id, content="Sorry, could not reflect and improve the novel outlines.")
            return

        logger.info(f"Writing the novel content on: {topic}")
        yield from self.writer.run(reflection_response.content, stream=True)


if __name__ == "__main__":
    from rich.prompt import Prompt

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a novel topic[/bold]\n✨",
        default="月球求生",
    )

    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the novel writing workflow
    write_novel_workflow = WriteNovelWorkflow(
        session_id=f"write-novel-on-{url_safe_topic}",
        storage=SqlWorkflowStorage(
            table_name="write_novel_workflows",
            db_file="tmp/novel_workflows.db",
        ),
    )

    # Execute the workflow
    novel_stream: Iterator[RunResponse] = write_novel_workflow.run(topic=topic)

    # Print the response
    pprint_run_response(novel_stream, markdown=True)
