# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Investment analysis workflow with session persistence

WHY Workflow (not a single Agent or Skill):
1. Multi-agent isolation: each analyst has different tools and prompts
2. Session state persistence: resume interrupted analysis from DB
3. Deterministic pipeline: always Stock -> Research -> Investment order
4. Per-step file output: each agent writes to its own report file

pip install yfinance agentica
"""
import sys
import os
from pathlib import Path
from shutil import rmtree

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Workflow, RunResponse, pprint_run_response, logger
from agentica.tools.yfinance_tool import YFinanceTool
from agentica.db.sqlite import SqliteDb

# Setup output directory
reports_dir = Path(__file__).parent.joinpath("outputs", "investment")
if reports_dir.is_dir():
    rmtree(path=reports_dir, ignore_errors=True)
reports_dir.mkdir(parents=True, exist_ok=True)


class InvestmentPipeline(Workflow):
    """Three-stage investment analysis: Data Collection -> Research -> Proposal.

    Each stage uses a different agent with different tools and expertise.
    Session state is persisted to SQLite, allowing resumption on failure.
    """

    description: str = "Multi-agent investment analysis with session persistence."

    # Stage 1: Data collection agent (has financial data tools)
    data_collector: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="DataCollector",
        tools=[YFinanceTool(company_info=True, analyst_recommendations=True, company_news=True)],
        instructions=[
            "Collect financial data for the given companies.",
            "Get company info, analyst recommendations, and recent news.",
            "Output a structured markdown report with all collected data.",
        ],
        save_response_to_file=str(reports_dir / "01_data_collection.md"),
    )

    # Stage 2: Research analyst (no tools, pure analysis)
    researcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="ResearchAnalyst",
        instructions=[
            "Analyze the financial data and rank companies by investment potential.",
            "Consider: growth trajectory, market position, risk factors, analyst consensus.",
            "Be skeptical - focus on maximizing risk-adjusted returns.",
        ],
        save_response_to_file=str(reports_dir / "02_research_analysis.md"),
    )

    # Stage 3: Investment lead (final decision maker)
    investment_lead: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="InvestmentLead",
        instructions=[
            "Create an investment proposal based on the research analysis.",
            "Allocate a $100,000 budget across the recommended companies.",
            "Specify exact dollar amounts and provide clear rationale for each allocation.",
        ],
        save_response_to_file=str(reports_dir / "03_investment_proposal.md"),
    )

    def run(self, companies: str):
        """Execute the investment analysis pipeline."""
        # Stage 1: Collect financial data
        logger.info(f"Stage 1: Collecting data for {companies}")
        data_response = self.data_collector.run_sync(companies)
        if not data_response or not data_response.content:
            yield RunResponse(content="Data collection failed.")
            return

        # Cache in session state for potential resume
        self.session_state["data_collected"] = True

        # Stage 2: Research and ranking
        logger.info("Stage 2: Analyzing and ranking companies")
        research_response = self.researcher.run_sync(data_response.content)
        if not research_response or not research_response.content:
            yield RunResponse(content="Research analysis failed.")
            return

        self.session_state["research_completed"] = True

        # Stage 3: Investment proposal (streamed)
        logger.info("Stage 3: Creating investment proposal")
        yield from self.investment_lead.run_stream_sync(research_response.content)


if __name__ == "__main__":
    companies = "TSLA, NVDA"

    # Session ID enables persistence and resume
    session_key = companies.lower().replace(" ", "").replace(",", "-")
    pipeline = InvestmentPipeline(
        session_id=f"invest-{session_key}",
        db=SqliteDb(db_file=str(reports_dir / "sessions.db")),
    )

    print("=" * 60)
    print(f"Investment Pipeline: {companies}")
    print("=" * 60)

    result = pipeline.run_sync(companies=companies)
    pprint_run_response(result)
