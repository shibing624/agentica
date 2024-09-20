"""
Please install dependencies using:
pip install yfinance agentica
"""

import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.workflow import Workflow, Task
from agentica.tools.yfinance import YFinanceTool

stock_analyst_report = "stock_analyst_report.md"
research_analyst_report = "research_analyst_report.md"
investment_report = "investment_report.md"

stock_analyst = Assistant(
    name="Stock Analyst",
    llm=AzureOpenAILLM(model="gpt-4o"),
    tools=[YFinanceTool(company_info=True, analyst_recommendations=True, company_news=True)],
    description="You are a Senior Investment Analyst for Goldman Sachs tasked with producing a research report for a very important client.",
    instructions=[
        "You will be provided with a list of companies to write a report on.",
        "Get the company information, analyst recommendations and news for each company",
        "Generate an in-depth report for each company in markdown format with all the facts and details."
        "Note: This is only for educational purposes.",
    ],
    expected_output="Report in markdown format",
    output_file_name=stock_analyst_report,
)
research_analyst = Assistant(
    name="Research Analyst",
    llm=AzureOpenAILLM(model="gpt-4o"),
    description="You are a Senior Investment Analyst for Goldman Sachs tasked with producing a ranked list of companies based on their investment potential.",
    instructions=[
        "You will write a research report based on the information provided by the Stock Analyst.",
        "Think deeply about the value of each stock.",
        "Be discerning, you are a skeptical investor focused on maximising growth.",
        "Then rank the companies in order of investment potential, with as much detail about your decision as possible.",
        "Prepare a markdown report with your findings with as much detail as possible.",
    ],
    expected_output="Report in markdown format",
    output_file_name=research_analyst_report,
)

investment_lead = Assistant(
    name="Investment Lead",
    llm=AzureOpenAILLM(model="gpt-4o"),
    description="You are a Senior Investment Analyst for Goldman Sachs tasked with producing a research report for a very important client.",
    instructions=[
        "Review the report provided and produce a final client-worth report",
    ],
    output_file_name=investment_report,
)

workflow = Workflow(
    name="Investment Research Workflow",
    tasks=[
        Task(
            description="Collect information about NVDA & TSLA.",
            assistant=stock_analyst,
            show_output=False,
        ),
        Task(
            description="Produce a ranked list based on the information provided by the stock analyst.",
            assistant=research_analyst,
            show_output=False,
        ),
        Task(
            description="Review the research report and produce a final report for the client.",
            assistant=investment_lead,
        ),
    ],
    debug_mode=True,
)

r = workflow.run(stream=False)
print(r)
