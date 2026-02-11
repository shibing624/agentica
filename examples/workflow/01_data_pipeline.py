# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Data processing pipeline - demonstrates Workflow's unique value

WHY Workflow (not a single Agent or Skill):
1. Deterministic step ordering: Extract -> Validate -> Transform -> Summarize
2. Non-LLM steps in between: Python data validation and deduplication
3. Different models per step: cheap model for extraction, powerful model for analysis
4. Structured data flow with type safety between steps

pip install agentica
"""
import sys
import os
from typing import List, Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Workflow, RunResponse


class ExtractedItem(BaseModel):
    """A single data item extracted from raw text."""
    name: str = Field(..., description="Item name")
    category: str = Field(..., description="Category")
    value: Optional[float] = Field(None, description="Numeric value if present")
    source: str = Field(..., description="Where this was mentioned")


class ExtractedData(BaseModel):
    """Collection of extracted items."""
    items: List[ExtractedItem] = []


class AnalysisReport(BaseModel):
    """Final analysis output."""
    summary: str
    key_findings: List[str]
    recommendations: List[str]


class DataPipeline(Workflow):
    """ETL-style pipeline: Extract -> Validate -> Transform -> Analyze.

    Steps 1 and 3 use LLM (different models for cost optimization).
    Step 2 is pure Python (no LLM needed, deterministic).
    """

    description: str = "Extract, validate, and analyze structured data from raw text."

    # Step 1: Cheap model for extraction (high-volume, low-complexity)
    extractor: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="DataExtractor",
        instructions=[
            "Extract all structured data items from the given text.",
            "For each item, identify: name, category, numeric value (if any), and source context.",
        ],
        response_model=ExtractedData,
    )

    # Step 3: Powerful model for analysis (low-volume, high-complexity)
    analyst: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="DataAnalyst",
        instructions=[
            "Analyze the validated data and produce insights.",
            "Focus on trends, anomalies, and actionable recommendations.",
            "Be specific and data-driven in your analysis.",
        ],
        response_model=AnalysisReport,
    )

    def _validate_and_deduplicate(self, data: ExtractedData) -> ExtractedData:
        """Step 2: Pure Python validation - no LLM needed, fully deterministic."""
        seen = set()
        valid_items = []
        for item in data.items:
            # Deduplicate by name
            key = item.name.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            # Validate: skip items with empty names or categories
            if not item.name.strip() or not item.category.strip():
                continue
            valid_items.append(item)
        return ExtractedData(items=valid_items)

    def run(self, raw_text: str) -> RunResponse:
        """Execute the data pipeline."""
        # Step 1: LLM extraction (cheap model)
        extract_response = self.extractor.run_sync(
            f"Extract structured data from this text:\n\n{raw_text}"
        )
        if not extract_response or not isinstance(extract_response.content, ExtractedData):
            return RunResponse(content="Extraction failed.")

        extracted = extract_response.content
        print(f"[Step 1] Extracted {len(extracted.items)} items")

        # Step 2: Pure Python validation (no LLM cost)
        validated = self._validate_and_deduplicate(extracted)
        print(f"[Step 2] Validated: {len(validated.items)} items (removed {len(extracted.items) - len(validated.items)} duplicates/invalid)")

        if not validated.items:
            return RunResponse(content="No valid data items found after validation.")

        # Step 3: LLM analysis (powerful model)
        items_text = "\n".join(
            f"- {item.name} ({item.category}): value={item.value}, source='{item.source}'"
            for item in validated.items
        )
        analysis_response = self.analyst.run_sync(
            f"Analyze these validated data items:\n{items_text}"
        )
        if not analysis_response or not isinstance(analysis_response.content, AnalysisReport):
            return RunResponse(content="Analysis failed.")

        report = analysis_response.content
        # Format final output
        output = f"# Data Pipeline Report\n\n"
        output += f"## Summary\n{report.summary}\n\n"
        output += f"## Key Findings\n"
        for finding in report.key_findings:
            output += f"- {finding}\n"
        output += f"\n## Recommendations\n"
        for rec in report.recommendations:
            output += f"- {rec}\n"
        output += f"\n---\nPipeline: {len(extracted.items)} extracted -> {len(validated.items)} validated -> 1 report"

        return RunResponse(content=output)


if __name__ == "__main__":
    pipeline = DataPipeline()

    sample_text = """
    2024年Q3科技行业报告摘要：
    
    苹果公司(Apple)在智能手机市场份额达到23.5%，较上季度增长2个百分点。
    三星(Samsung)市场份额为19.8%，保持稳定。华为(Huawei)凭借Mate系列回归，
    市场份额从8.1%增至12.3%。小米(Xiaomi)份额为13.2%，同比略降。
    
    在AI芯片领域，英伟达(NVIDIA)占据数据中心GPU市场约82%的份额，
    AMD份额约12%，Intel份额约5%。英伟达H100芯片单价约25000美元。
    
    云服务市场中，AWS份额32%（收入约262亿美元），Azure份额23%，
    Google Cloud份额11%。AWS份额同比下降1个百分点，Azure增长2个百分点。
    
    注：苹果公司手机份额23.5%（重复数据），NVIDIA GPU份额82%。
    """

    print("=" * 60)
    print("Data Pipeline Workflow Demo")
    print("=" * 60)

    result = pipeline.run_sync(sample_text)
    print("\n" + result.content)
