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
    name="brilliant entrepreneur",
    llm=AzureOpenAILLM(model="gpt-4o"),
    description="You are a brilliant entrepreneur. You are exceptional at generating new business ideas and marketing them.",
    tools=[FileTool(data_dir="outputs")],
    output_dir="outputs",
    output_file_name="save.md",
)

workflow = Actionflow(
    name="Brainstorm Workflow",
    tasks=[
        Task(
            description="""1.Brainstorm five ideas for a education product.
            2.Choose the best idea from the list based on likely purchase intent
            3.Write a short description of the chosen product.
            4.Create a simple HTML page with the chosen product, description, and list some benefits of the product in the table. 
            5.Save the HTML page as 'product.html'.
            """,
            assistant=idea_assistant,
        ),
    ],
    debug_mode=True,
)

workflow.print_response(markdown=True)
