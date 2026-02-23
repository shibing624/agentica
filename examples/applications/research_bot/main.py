# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Research Bot - Multi-Agent research pipeline

This application demonstrates a 3-agent research pipeline:
1. Planner Agent - Decomposes a query into structured search plans (Pydantic output)
2. Search Agent  - Executes web searches in parallel (asyncio)
3. Writer Agent  - Synthesizes findings into a streaming Markdown report

Key agentica features demonstrated:
- Structured output (response_model with Pydantic)
- Parallel search via asyncio.create_task
- Streaming output via run_stream
- Multi-agent orchestration without team (explicit pipeline)

Usage:
    python main.py
"""
import asyncio
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat, RunEvent, BaiduSearchTool
from agentica.tools.search_bocha_tool import SearchBochaTool


# ============================================================================
# Structured Output Models
# ============================================================================

class WebSearchItem(BaseModel):
    reason: str = Field(description="Why this search is needed for the research.")
    query: str = Field(description="The search query to execute.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(
        description="A list of web searches to perform. 5-15 items recommended."
    )


class ReportData(BaseModel):
    """Research report structured output."""
    short_summary: str = Field(
        default="",
        description="A 2-3 sentence summary of the research findings.",
    )
    markdown_report: str = Field(
        default="",
        description="The full research report in Markdown format, 1000+ words with sections and citations.",
    )
    follow_up_questions: str = Field(
        default="",
        description="3 suggested follow-up research questions. split by \\n",
    )


# ============================================================================
# Agent Definitions
# ============================================================================

planner_agent = Agent(
    name="PlannerAgent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "Given a research query, come up with a list of web searches to perform "
        "to best answer the query. Output between 2 and 5 searches."
    ),
    response_model=WebSearchPlan,
    debug=True,
)

search_agent = Agent(
    name="SearchAgent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "Search the web for the given query and produce a concise summary "
        "of the findings. The summary should be 2-3 paragraphs and at most 300 words. "
        "Capture the key points and include source URLs when available."
    ),
    # tools=[BaiduSearchTool()],
    tools=[SearchBochaTool()],
    debug=True,
)

writer_agent = Agent(
    name="WriterAgent",
    model=OpenAIChat(id="gpt-4o"),
    description=(
        "Given a research query and a collection of search results, write a comprehensive "
        "research report. The report should be well-structured with sections, detailed analysis, "
        "and citations. Aim for 1000+ words. Include a short summary at the beginning "
        "and suggest follow-up research questions at the end."
    ),
    response_model=ReportData,
    debug=True,
)


# ============================================================================
# Pipeline Orchestration
# ============================================================================

async def plan_searches(query: str) -> WebSearchPlan:
    """Stage 1: Generate a search plan from the query."""
    print("[Stage 1] Planning searches...")
    result = await planner_agent.run(f"Query: {query}")
    plan = result.content
    if not isinstance(plan, WebSearchPlan):
        print("  -> Structured plan parsing failed, using default search")
        plan = WebSearchPlan(searches=[
            WebSearchItem(reason="General research", query=query),
        ])
    print(f"  -> Generated {len(plan.searches)} search items")
    return plan


async def execute_single_search(item: WebSearchItem, index: int, total: int) -> str:
    """Execute a single search and return the summary."""
    try:
        result = await search_agent.run(
            f"Search for: {item.query}\nReason: {item.reason}"
        )
        return result.content or ""
    except Exception as e:
        print(f"  [!] Search {index + 1}/{total} failed: {e}")
        return ""


async def perform_searches(plan: WebSearchPlan) -> List[str]:
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

    print(f"  -> Got {len(results)} successful search results")
    return results


async def write_report(query: str, search_results: List[str]) -> ReportData:
    """Stage 3: Synthesize search results into a report."""
    print(f"\n[Stage 3] Writing report...")

    search_text = "\n\n---\n\n".join(
        f"**Search Result {i + 1}:**\n{r}" for i, r in enumerate(search_results)
    )
    prompt = f"Research Query: {query}\n\nSearch Results:\n{search_text}"

    result = await writer_agent.run(prompt)
    report = result.content
    if isinstance(report, ReportData):
        print("  -> Report generated (structured)")
    else:
        print("  -> Report generated (plain text fallback)")
        raw = str(report) if report else "No report generated."
        report = ReportData(
            short_summary=raw[:200],
            markdown_report=raw,
            follow_up_questions='',
        )
    return report


# ============================================================================
# Main
# ============================================================================

async def main():
    print("=" * 60)
    print("Research Bot - Multi-Agent Pipeline")
    print("=" * 60)

    query = "RAG技术的原理和最佳实践"
    print(f"\nResearch Query: {query}\n")

    # Stage 1: Plan
    plan = await plan_searches(query)
    for i, item in enumerate(plan.searches):
        print(f"  {i + 1}. [{item.reason}] {item.query}")

    # Stage 2: Search (parallel)
    search_results = await perform_searches(plan)

    # Stage 3: Write report
    report = await write_report(query, search_results)

    # Output
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(report.short_summary)

    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60)
    print(report.markdown_report)

    print("\n" + "=" * 60)
    print("FOLLOW-UP QUESTIONS")
    print("=" * 60)
    print(report.follow_up_questions)


if __name__ == "__main__":
    asyncio.run(main())
