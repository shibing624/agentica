"""
Please install dependencies using:
pip install actionflow
"""

import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.actionflow import Actionflow, Task
from actionflow.tools.file import FileTool

idea_assistant = Assistant(
    name="企业家",
    llm=AzureOpenAILLM(model="gpt-4o"),
    description="你是一位杰出的企业家。你擅长于产生新的商业创意并将其营销推广。",
    tools=[FileTool(data_dir="outputs")],
    output_dir="outputs",
    output_file_name="save.md",
)

workflow = Actionflow(
    name="商业想法生成",
    tasks=[
        Task(
            description="""
            1.提出五个教育产品创意。
            2.根据可能的购买意图，从列表中选择最佳想法。
            3.编写所选产品的简短描述。
            4.创建一个简单的HTML页面，包括所选产品、描述，并在表格中列出该产品的一些优点。
            5.将HTML页面保存为 'product.html'.
            """,
            assistant=idea_assistant,
        ),
    ],
    debug_mode=True,
)

workflow.run()
