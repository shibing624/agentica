# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Investment workflow demo - Multi-agent investment report generation

This example shows a complex workflow for investment analysis:
1. Stock analyst - Gathers company information
2. Research analyst - Ranks companies by investment potential
3. Investment lead - Creates investment proposal

pip install yfinance agentica
"""
import sys
import os
from pathlib import Path
from shutil import rmtree

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.workflow import Workflow
from agentica import logger
from agentica import RunResponse, pprint_run_response
from agentica.tools.yfinance_tool import YFinanceTool
from agentica.db.sqlite import SqliteDb

# Setup output directory
reports_dir = Path(__file__).parent.joinpath("outputs", "report_investment")
if reports_dir.is_dir():
    rmtree(path=reports_dir, ignore_errors=True)
reports_dir.mkdir(parents=True, exist_ok=True)

stock_analyst_report = str(reports_dir.joinpath("stock_analyst_report.md"))
research_analyst_report = str(reports_dir.joinpath("research_analyst_report.md"))
investment_report = str(reports_dir.joinpath("investment_report.md"))


class InvestmentReportGenerator(Workflow):
    """Generate investment reports for a list of companies."""
    
    description: str = (
        "Produce a research report on a list of companies and rank them based on investment potential."
    )

    stock_analyst: Agent = Agent(
        tools=[YFinanceTool(company_info=True, analyst_recommendations=True, company_news=True)],
        description="You are a Senior Investment Analyst for Goldman Sachs.",
        instructions=[
            "You will be provided with a list of companies to write a report on.",
            "Get the company information, analyst recommendations and news for each company",
            "Generate an in-depth report for each company in markdown format.",
            "Note: This is only for educational purposes.",
        ],
        expected_output="Report in markdown format",
        save_response_to_file=stock_analyst_report,
    )

    research_analyst: Agent = Agent(
        name="Research Analyst",
        description="You are a Senior Investment Analyst tasked with ranking companies.",
        instructions=[
            "You will write a research report based on the Stock Analyst's information.",
            "Think deeply about the value of each stock.",
            "Be discerning, you are a skeptical investor focused on maximising growth.",
            "Rank the companies in order of investment potential.",
            "Prepare a markdown report with your findings.",
        ],
        expected_output="Report in markdown format",
        save_response_to_file=research_analyst_report,
    )

    investment_lead: Agent = Agent(
        name="Investment Lead",
        description="You are a Senior Investment Lead tasked with investing $100,000.",
        instructions=[
            "Review the report provided by the research analyst.",
            "Produce an investment proposal for the client.",
            "Provide the amount to invest in each company and explain why.",
        ],
        save_response_to_file=investment_report,
    )

    def run(self, companies: str):
        logger.info(f"Getting investment reports for companies: {companies}")
        
        # Step 1: Stock analysis
        initial_report = self.stock_analyst.run(companies)
        if initial_report is None or not initial_report.content:
            yield RunResponse(run_id=self.run_id, content="Sorry, could not get the stock analyst report.")
            return

        # Step 2: Research and ranking
        logger.info("Ranking companies based on investment potential.")
        ranked_companies = self.research_analyst.run(initial_report.content)
        if ranked_companies is None or not ranked_companies.content:
            yield RunResponse(run_id=self.run_id, content="Sorry, could not get the ranked companies.")
            return

        # Step 3: Investment proposal
        logger.info("Reviewing the research report and producing an investment proposal.")
        yield from self.investment_lead.run(ranked_companies.content, stream=True)


if __name__ == "__main__":
    companies = 'TSLA'

    # Convert to URL-safe string
    url_safe_companies = companies.lower().replace(" ", "-").replace(",", "")

    # Initialize workflow
    investment_report_generator = InvestmentReportGenerator(
        session_id=f"investment-report-{url_safe_companies}",
        db=SqliteDb(
            db_file="outputs/investment_workflows.db",
        ),
    )

    # Execute workflow
    report = investment_report_generator.run(companies=companies)
    pprint_run_response(report)
