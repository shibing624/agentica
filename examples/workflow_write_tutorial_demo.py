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
from agentica.tools.search_exa_tool import SearchExaTool
from agentica.tools.wikipedia_tool import WikipediaTool

output_dir = "outputs/tutorial_v1/"
# 首次生成目录的任务助手
topic = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Initial Topic Generator",
    description="You are a seasoned technical professional, your goal is to generate a detailed and well-structured technical tutorial outline in Chinese.",
    output_dir=output_dir,
    instructions=[
        dedent("""
        请生成一个关于xxx的技术教程目录。
        目录结构应包括主目录和必要的二级目录，要求具体并具有实用意义。
        输出必须严格按照以下字典格式：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}, {"dir 2": ["sub dir 3", "sub dir 4"]})。
        输出必须为中文，并严格遵守格式要求，不得有多余空格或换行。
        """),
    ],
)

# 反思任务助手
reflector = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Reflective Topic Generator",
    tools=[SearchSerperTool(), WikipediaTool(), SearchExaTool()],
    description="You are an internet field expert, reflect on the initial tutorial directory based on Google search results and generate an optimized directory.",
    output_dir=output_dir,
    instructions=[
        dedent("""
        请根据Google搜索结果, Exa搜索结果和Wiki搜索结果，对初始生成的MySQL教程目录进行反思和优化，生成一个更为具体和实用的目录结构。
        目录结构应包括主目录和必要的二级目录，要求具体并具有实用意义。
        输出必须严格按照以下字典格式：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}, {"dir 2": ["sub dir 3", "sub dir 4"]})。
        输出必须为中文，并严格遵守格式要求，不得有多余空格或换行。
        """),
    ],
)

# 编写教程任务助手
writer = Assistant(
    llm=OpenAILLM(model="gpt-4o-mini"),
    name="Tutorial Writer",
    output_dir=output_dir,
    output_file_name="tutorial.md",
    description="You are an experienced technical writer, your goal is to write detailed tutorial documents in Chinese, adhering strictly to Markdown syntax.",
    instructions=[
        dedent("""
        请根据提供的教程目录编写详细的教程内容。
        使用Markdown语法对内容进行排版，提供标准代码示例和注释。
        输出必须为中文，并严格遵守格式要求，不得有多余或冗余内容。
        """),
    ],
)

# 创建工作流
flow = Workflow(
    name="Write Tutorial Workflow",
    tasks=[
        # 第一步：生成初始目录
        Task(
            description="Generate the initial tutorial directory.",
            assistant=topic,
        ),
        # 第二步：基于谷歌搜索结果反思和优化目录
        Task(
            description="Reflect on and optimize the tutorial directory based on Google search results.",
            assistant=reflector,
        ),
        # 第三步：编写教程内容
        Task(
            description="Write the detailed tutorial content in Chinese.",
            assistant=writer,
        ),
    ],
    debug_mode=True,
)

# 执行工作流
r = flow.run("Write a tutorial about redis")
print("".join(r))
