# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Novel writing workflow demo - Multi-step novel generation with reflection

This example demonstrates a creative writing workflow that:
1. Generates novel outlines
2. Reflects and improves the outlines using web search
3. Writes detailed chapter content
"""
import sys
import os
from textwrap import dedent
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica import Workflow, RunResponse, pprint_run_response
from agentica.tools.baidu_search_tool import BaiduSearchTool


class WriteNovelWorkflow(Workflow):
    """A workflow for generating novels with outline, reflection, and writing stages."""

    description: str = "Generate a comprehensive novel on a given topic."

    outlines: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            ## context
            You need to generate a novel named 'xxx'.
            Be creative and thorough while generating the outline.

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
                "character_names": ["Character1", "Character2"],
                "conflict": "...",
                "plot": "...",
                "ending": "..."
            }
            [/CONTENT]

            ## constraint
            - No more than 10 chapters
            - Output in the language of the prompt
            - Wrap output in [CONTENT][/CONTENT] tags
            """),
        ],
    )

    reflection: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[BaiduSearchTool()],
        instructions=[
            dedent("""
            ## context
            Review the novel outlines and use web search to improve them.
            Search for relevant information to make the story more realistic.

            ## action
            1. Conduct searches to gather information
            2. Reflect on the results
            3. Improve the original outlines accordingly
            4. Output in the same format as input
            """),
        ],
    )

    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            dedent("""
            ## context
            Write detailed content for the novel based on the outlines.

            ## output format
            Output markdown formatted novel with clear chapter structure.
            Write in Chinese. Do not use ```markdown``` tags.
            """),
        ],
    )

    def run(self, topic: str):
        """Execute the novel writing workflow."""
        logger.info(f"Generating a novel on: {topic}")

        # Step 1: Generate outlines
        outlines_response = self.outlines.run(topic)
        if outlines_response is None or not outlines_response.content:
            yield RunResponse(
                run_id=self.run_id,
                content="Sorry, could not generate the novel outlines."
            )
            return

        # Step 2: Reflect and improve
        logger.info(f"Reflecting and improving the novel outlines on: {topic}")
        reflection_response = self.reflection.run(outlines_response.content)
        if reflection_response is None or not reflection_response.content:
            yield RunResponse(
                run_id=self.run_id,
                content="Sorry, could not reflect and improve the novel outlines."
            )
            return

        # Step 3: Write the novel
        logger.info(f"Writing the novel content on: {topic}")
        yield from self.writer.run(reflection_response.content, stream=True)


if __name__ == "__main__":
    topic = "月球求生"

    workflow = WriteNovelWorkflow(
        session_id=f"write-novel-on-{topic}",
    )

    novel_stream = workflow.run(topic=topic)
    pprint_run_response(novel_stream)
