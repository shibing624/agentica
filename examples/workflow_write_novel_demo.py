# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
from textwrap import dedent

sys.path.append('..')
from agentica import Assistant, OpenAILLM
from agentica.workflow import Workflow, Task
from agentica.tools.search_serper_tool import SearchSerperTool

output_dir = "outputs/novel_v1/"
# 生成初始大纲的 Assistant
outlines = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Initial Outlines Generator",
    description="You are a creative assistant specialized in generating novel outlines.",
    output_dir=output_dir,
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

# 反思和重新梳理大纲的 Assistant
reflection = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Outline Reflector",
    output_dir=output_dir,
    description="You are a thoughtful assistant who evaluates and refines novel outlines based on additional research.",
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
# 写详细章节内容的 Assistant
writer = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Detailed Novel Writer",
    output_dir=output_dir,
    output_file_name="novel_content.md",
    description="You are an experienced technical writer skilled in writing detailed novel content in Chinese, following Markdown syntax.",
    instructions=[
        dedent("""
        ## context
        You need to write detailed content for the novel named 'xxx', based on the generated outlines.

        ## output format example
        输出markdown格式的小说，章节内容结构清晰。不用给```markdown```标记，直接输出md格式的内容。

        ## action
        Write the detailed novel chapters and ensure they follow the formatting guidelines.
        """),
    ],
)

# 创建工作流
flow = Workflow(
    name="Write Novel Workflow",
    tasks=[
        Task(
            description="Generate the initial outlines.",
            assistant=outlines,
        ),
        Task(
            description="Reflect and improve the outlines based on search results.",
            assistant=reflection,
        ),
        Task(
            description="Write the detailed novel content in Chinese.",
            assistant=writer,
        ),
    ],
    debug_mode=True,
)

# 执行工作流
r = flow.run("Write a novel: 月球求生.")
print("".join(r))
