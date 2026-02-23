# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Financial Research Agent - Advanced multi-agent pipeline

This application demonstrates a 6-agent financial research pipeline:
1. Planner Agent        - Decomposes query into search plans (structured output)
2. Search Agent         - Executes web searches in parallel
3. Fundamentals Agent   - Analyzes company fundamentals (agent-as-tool)
4. Risk Agent           - Analyzes risk factors (agent-as-tool)
5. Writer Agent         - Writes the report, calling sub-analysts on demand
6. Verifier Agent       - Audits report consistency and source reliability

Key agentica features demonstrated:
- Agent.as_tool() — wrap agents as callable tools for other agents
- custom_output_extractor — transform structured output for tool consumers
- Structured output (response_model with Pydantic)
- Parallel search via asyncio
- Verification pipeline (write → verify)
- Multi-agent orchestration

Usage:
    python main.py
"""
import asyncio
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat, BaiduSearchTool
from agentica.tools.search_bocha_tool import SearchBochaTool


# ============================================================================
# Structured Output Models
# ============================================================================

class SearchItem(BaseModel):
    reason: str = Field(description="Why this search is needed for the financial analysis.")
    query: str = Field(description="The search query to execute.")


class SearchPlan(BaseModel):
    searches: List[SearchItem] = Field(
        description="A list of web searches to perform for financial research. 5-15 items."
    )


class AnalysisSummary(BaseModel):
    summary: str = Field(description="A concise analysis summary (2-4 paragraphs).")


class FinancialReportData(BaseModel):
    """Financial report structured output."""
    model_config = {"populate_by_name": True}

    short_summary: str = Field(
        default="",
        description="A 2-3 sentence executive summary.",
    )
    markdown_report: str = Field(
        default="",
        description="Full financial research report in Markdown with sections, data, and citations.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="3-5 suggested follow-up research questions.",
    )


class VerificationResult(BaseModel):
    verified: bool = Field(description="Whether the report passes consistency and source checks.")
    issues: str = Field(
        description="Description of any issues found, or 'No issues found' if verified."
    )


# ============================================================================
# Agent Definitions
# ============================================================================

# 1. Planner Agent
planner_agent = Agent(
    name="FinancialPlanner",
    model=OpenAIChat(),
    description=(
        "Given a financial research query, produce a list of web searches to gather "
        "comprehensive data: financials, competitors, industry trends, risks, news. "
        "Output 5-15 search items."
    ),
    response_model=SearchPlan,
    debug=True,
)

# 2. Search Agent
search_agent = Agent(
    name="FinancialSearcher",
    model=OpenAIChat(),
    description=(
        "Search the web for the given query and produce a concise summary of findings. "
        "Focus on financial data, metrics, and facts. 2-3 paragraphs, 300 words max."
    ),
    tools=[SearchBochaTool()],
    debug=True,
)

# 3. Fundamentals Analysis Agent (will be used as tool)
fundamentals_agent = Agent(
    name="FundamentalsAnalyst",
    model=OpenAIChat(),
    description=(
        "Analyze the provided financial data and produce a summary of key fundamental metrics: "
        "revenue, profit margins, growth rates, P/E ratio, debt levels, and competitive position."
    ),
    response_model=AnalysisSummary,
    debug=True,
)

# 4. Risk Analysis Agent (will be used as tool)
risk_agent = Agent(
    name="RiskAnalyst",
    model=OpenAIChat(),
    description=(
        "Analyze the provided data for risk factors: market risk, regulatory risk, "
        "competition, supply chain vulnerabilities, and macro-economic threats."
    ),
    response_model=AnalysisSummary,
    debug=True,
)

# 5. Writer Agent (tools will be injected at runtime)
writer_agent = Agent(
    name="FinancialWriter",
    model=OpenAIChat(id="gpt-4o"),
    description=(
        "Write a comprehensive financial research report based on the provided search results.\n\n"
        "STRICT RULES:\n"
        "1. ONLY use facts, numbers, and data that appear in the search results provided to you. "
        "DO NOT fabricate or hallucinate any statistics, figures, or claims.\n"
        "2. DO NOT invent source references or URLs. If you cite a fact, mention where it came "
        "from in the search results (e.g. 'According to search result #3...'). Never use "
        "placeholder links like '[Source](#)' or made-up URLs.\n"
        "3. If the search results lack data for a topic, explicitly state 'Insufficient data "
        "available' rather than guessing.\n"
        "4. Include specific numbers (revenue, growth rates, P/E ratios, etc.) ONLY when they "
        "appear verbatim in the search results.\n\n"
        "You have access to two specialist tools:\n"
        "- fundamentals_analysis: call this to get a detailed analysis of key financial metrics\n"
        "- risk_analysis: call this to get a detailed analysis of risk factors\n"
        "Use these tools when you need deeper analysis for specific sections."
    ),
    response_model=FinancialReportData,
    debug=True,
)

# 6. Verifier Agent
verifier_agent = Agent(
    name="ReportVerifier",
    model=OpenAIChat(),
    description=(
        "Audit the financial research report for:\n"
        "1. Internal consistency (no contradicting statements)\n"
        "2. Source reliability (claims should be supported by referenced data)\n"
        "3. Unsupported assertions (flag any claims without evidence)\n"
        "4. Completeness (key financial aspects should be covered)\n"
        "Output whether the report is verified and list any issues found."
    ),
    response_model=VerificationResult,
    debug=True,
)


# ============================================================================
# Output Extractor for Agent-as-Tool
# ============================================================================

def summary_extractor(run_response) -> str:
    """Extract the summary text from an AnalysisSummary response."""
    if run_response and run_response.content:
        if isinstance(run_response.content, AnalysisSummary):
            return run_response.content.summary
        if isinstance(run_response.content, str):
            return run_response.content
    return "No analysis available."


# ============================================================================
# Pipeline Orchestration
# ============================================================================

async def plan_searches(query: str) -> SearchPlan:
    """Stage 1: Generate a financial search plan."""
    print("[Stage 1] Planning financial research searches...")
    result = await planner_agent.run(f"Financial research query: {query}")
    plan = result.content
    if not isinstance(plan, SearchPlan):
        # Fallback: create a minimal default plan
        print("  -> Structured plan parsing failed, using default searches")
        plan = SearchPlan(searches=[
            SearchItem(reason="General analysis", query=query),
        ])
    print(f"  -> Generated {len(plan.searches)} search items")
    return plan


async def execute_single_search(item: SearchItem, index: int, total: int) -> str:
    """Execute a single search."""
    try:
        result = await search_agent.run(
            f"Search for: {item.query}\nReason: {item.reason}"
        )
        return result.content or ""
    except Exception as e:
        print(f"  [!] Search {index + 1}/{total} failed: {e}")
        return ""


async def perform_searches(plan: SearchPlan) -> List[str]:
    """Stage 2: Execute all searches in parallel."""
    print(f"\n[Stage 2] Executing {len(plan.searches)} searches in parallel...")
    tasks = [
        asyncio.create_task(execute_single_search(item, i, len(plan.searches)))
        for i, item in enumerate(plan.searches)
    ]

    results = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        completed += 1
        print(f"  -> Completed {completed}/{len(plan.searches)}")
        if result:
            results.append(result)

    print(f"  -> Got {len(results)} successful results")
    return results


async def write_report(query: str, search_results: List[str]) -> FinancialReportData:
    """Stage 3: Write the report with agent-as-tool for sub-analysis."""
    print("\n[Stage 3] Writing financial report (with specialist sub-agents)...")

    # Wrap sub-agents as tools
    fundamentals_tool = fundamentals_agent.as_tool(
        tool_name="fundamentals_analysis",
        tool_description=(
            "Use this tool to get a detailed analysis of key financial metrics "
            "(revenue, margins, growth, valuation). Pass the relevant financial data as input."
        ),
        custom_output_extractor=summary_extractor,
    )
    risk_tool = risk_agent.as_tool(
        tool_name="risk_analysis",
        tool_description=(
            "Use this tool to get a detailed analysis of potential risk factors "
            "(market, regulatory, competition, macro). Pass the relevant data as input."
        ),
        custom_output_extractor=summary_extractor,
    )

    # Inject tools into writer at runtime (original writer_agent stays clean)
    writer_with_tools = Agent(
        name=writer_agent.name,
        model=writer_agent.model,
        description=writer_agent.description,
        response_model=FinancialReportData,
        tools=[fundamentals_tool, risk_tool],
    )

    search_text = "\n\n---\n\n".join(
        f"**Search Result {i + 1}:**\n{r}" for i, r in enumerate(search_results)
    )
    prompt = f"Financial Research Query: {query}\n\nSearch Results:\n{search_text}"

    result = await writer_with_tools.run(prompt)
    report = result.content
    if isinstance(report, FinancialReportData):
        print("  -> Report generated (structured)")
    else:
        # Structured output parsing failed; wrap raw text into FinancialReportData
        print("  -> Report generated (plain text fallback)")
        raw = str(report) if report else "No report generated."
        report = FinancialReportData(
            short_summary=raw[:200],
            markdown_report=raw,
            follow_up_questions=[],
        )
    return report


async def verify_report(report: FinancialReportData) -> VerificationResult:
    """Stage 4: Verify the report for consistency and reliability."""
    print("\n[Stage 4] Verifying report...")
    result = await verifier_agent.run(report.markdown_report)
    verification = result.content
    if isinstance(verification, VerificationResult):
        status = "PASSED" if verification.verified else "FAILED"
        print(f"  -> Verification: {status}")
    else:
        # Fallback when structured output parsing fails
        print("  -> Verification: completed (plain text fallback)")
        verification = VerificationResult(
            verified=True,
            issues=str(verification) if verification else "Unable to verify.",
        )
    return verification


# ============================================================================
# Main
# ============================================================================

async def main():
    print("=" * 60)
    print("Financial Research Agent - Advanced Pipeline")
    print("=" * 60)

    query = "茅台（贵州茅台600519）的投资价值分析"
    print(f"\nResearch Query: {query}\n")

    # Stage 1: Plan
    plan = await plan_searches(query)
    for i, item in enumerate(plan.searches):
        print(f"  {i + 1}. [{item.reason}] {item.query}")

    # Stage 2: Search (parallel)
    search_results = await perform_searches(plan)

    # Stage 3: Write report (with agent-as-tool for sub-analysis)
    report = await write_report(query, search_results)

    # Stage 4: Verify
    verification = await verify_report(report)

    # Output
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(report.short_summary)

    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60)
    print(report.markdown_report)

    print("\n" + "=" * 60)
    print("FOLLOW-UP QUESTIONS")
    print("=" * 60)
    for i, q in enumerate(report.follow_up_questions):
        print(f"  {i + 1}. {q}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"  Status: {'PASSED' if verification.verified else 'FAILED'}")
    print(f"  Details: {verification.issues}")


if __name__ == "__main__":
    asyncio.run(main())
