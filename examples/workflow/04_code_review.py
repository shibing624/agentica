# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Code review pipeline - demonstrates deterministic quality assurance

WHY Workflow (not a single Agent or Skill):
1. Forced multi-pass review: code MUST go through all reviewers in order
2. Aggregation step is pure Python: merge reviews, compute pass/fail
3. Different reviewer perspectives: security, performance, maintainability
4. Deterministic quality gate: reject if any reviewer finds critical issues

Usage:
    python 04_code_review.py
"""
import sys
import os
from typing import List, Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Workflow, RunResponse


class ReviewFinding(BaseModel):
    severity: str = Field(..., description="critical / warning / info")
    description: str = Field(..., description="What was found")
    suggestion: str = Field(..., description="How to fix it")


class CodeReview(BaseModel):
    reviewer: str = Field(..., description="Reviewer name")
    passed: bool = Field(..., description="Whether the code passed this review")
    findings: List[ReviewFinding] = []
    summary: str = Field(..., description="Overall assessment")


class CodeReviewPipeline(Workflow):
    """Deterministic code review: Security -> Performance -> Maintainability -> Verdict.

    Each reviewer agent focuses on one aspect. The final verdict is computed
    by pure Python logic (not LLM), ensuring deterministic quality gating.
    """

    description: str = "Multi-perspective code review with deterministic quality gate."

    # Reviewer 1: Security focus
    security_reviewer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="SecurityReviewer",
        instructions=[
            "Review the code for security vulnerabilities.",
            "Check for: injection risks, authentication issues, data exposure, unsafe operations.",
            "Set passed=false if any critical security issue is found.",
        ],
        response_model=CodeReview,
    )

    # Reviewer 2: Performance focus
    performance_reviewer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="PerformanceReviewer",
        instructions=[
            "Review the code for performance issues.",
            "Check for: O(n^2) algorithms, memory leaks, unnecessary I/O, missing caching.",
            "Set passed=false if any critical performance issue is found.",
        ],
        response_model=CodeReview,
    )

    # Reviewer 3: Maintainability focus
    maintainability_reviewer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="MaintainabilityReviewer",
        instructions=[
            "Review the code for maintainability and code quality.",
            "Check for: code duplication, missing error handling, unclear naming, missing docs.",
            "Set passed=false if any critical maintainability issue is found.",
        ],
        response_model=CodeReview,
    )

    def _aggregate_reviews(self, reviews: List[CodeReview]) -> str:
        """Pure Python aggregation: deterministic pass/fail logic."""
        all_passed = all(r.passed for r in reviews)
        critical_count = sum(
            1 for r in reviews for f in r.findings if f.severity == "critical"
        )
        warning_count = sum(
            1 for r in reviews for f in r.findings if f.severity == "warning"
        )

        verdict = "APPROVED" if all_passed and critical_count == 0 else "REJECTED"

        lines = [
            f"# Code Review Report",
            f"",
            f"**Verdict: {verdict}**",
            f"- Critical issues: {critical_count}",
            f"- Warnings: {warning_count}",
            f"- Reviewers passed: {sum(1 for r in reviews if r.passed)}/{len(reviews)}",
        ]

        for review in reviews:
            status = "PASS" if review.passed else "FAIL"
            lines.append(f"\n## {review.reviewer} [{status}]")
            lines.append(review.summary)
            if review.findings:
                lines.append("\n### Findings:")
                for f in review.findings:
                    lines.append(f"- **[{f.severity.upper()}]** {f.description}")
                    lines.append(f"  Fix: {f.suggestion}")

        return "\n".join(lines)

    def run(self, code: str, language: str = "python") -> RunResponse:
        """Run the code review pipeline."""
        prompt = f"Review this {language} code:\n```{language}\n{code}\n```"

        # Step 1-3: Run all reviewers (deterministic order)
        reviews: List[CodeReview] = []
        reviewers = [
            ("Security", self.security_reviewer),
            ("Performance", self.performance_reviewer),
            ("Maintainability", self.maintainability_reviewer),
        ]

        for name, reviewer in reviewers:
            print(f"[Review] {name} reviewing...")
            response = reviewer.run_sync(prompt)
            if response and isinstance(response.content, CodeReview):
                response.content.reviewer = name
                reviews.append(response.content)
                status = "PASS" if response.content.passed else "FAIL"
                print(f"[Review] {name}: {status} ({len(response.content.findings)} findings)")
            else:
                print(f"[Review] {name}: SKIPPED (no response)")

        # Step 4: Pure Python aggregation (no LLM, deterministic)
        report = self._aggregate_reviews(reviews)
        return RunResponse(content=report)


if __name__ == "__main__":
    pipeline = CodeReviewPipeline()

    sample_code = '''
import sqlite3
import os

def get_user(user_id):
    """Get user from database."""
    conn = sqlite3.connect("users.db")
    # SQL injection vulnerability: user_id is directly interpolated
    cursor = conn.execute(f"SELECT * FROM users WHERE id = '{user_id}'")
    result = cursor.fetchone()
    conn.close()
    return result

def process_data(items):
    """Process a list of items."""
    result = []
    for i in items:
        for j in items:
            if i != j:
                result.append((i, j))
    return result

def read_config():
    """Read config file."""
    password = "admin123"  # Hardcoded password
    api_key = os.environ.get("API_KEY", "sk-default-key-12345")
    return {"password": password, "api_key": api_key}
'''

    print("=" * 60)
    print("Code Review Pipeline")
    print("=" * 60)

    result = pipeline.run_sync(sample_code, language="python")
    print("\n" + result.content)
