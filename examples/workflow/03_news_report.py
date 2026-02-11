# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: News report workflow with caching and incremental processing

WHY Workflow (not a single Agent or Skill):
1. Session state caching: search results and scraped articles persist across runs
2. Incremental processing: re-run skips already-scraped articles
3. Multi-agent with different response_model: searcher returns structured data,
   scraper returns structured data, writer produces free-form text
4. Pure Python orchestration between steps (cache lookup, data merging)

pip install newspaper4k agentica
"""
import sys
import os
import json
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Workflow, RunResponse, RunEvent, pprint_run_response, logger
from agentica.tools.newspaper_tool import NewspaperTool
from agentica.tools.baidu_search_tool import BaiduSearchTool


class NewsArticle(BaseModel):
    title: str = Field(..., description="Article title")
    url: str = Field(..., description="Article URL")
    summary: Optional[str] = Field(None, description="Brief summary")


class SearchResults(BaseModel):
    articles: List[NewsArticle] = []


class ScrapedArticle(BaseModel):
    title: str
    url: str
    summary: Optional[str] = None
    content: Optional[str] = Field(None, description="Markdown content")


class NewsReportWorkflow(Workflow):
    """News report generation with caching for incremental re-runs.

    First run: searches, scrapes, writes. Caches intermediate results in session_state.
    Second run: skips search/scrape (cache hit), only rewrites report with updated prompt.
    """

    description: str = "Generate news reports with persistent caching."

    searcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="NewsSearcher",
        tools=[BaiduSearchTool()],
        instructions=["Search for recent articles on the topic. Return the 5 most relevant."],
        response_model=SearchResults,
    )

    scraper: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="ArticleScraper",
        tools=[NewspaperTool()],
        instructions=["Scrape the article at the given URL. Return title, URL, and markdown content."],
        response_model=ScrapedArticle,
    )

    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="ReportWriter",
        instructions=[
            "Write a comprehensive, well-structured news report based on the provided articles.",
            "Include sections, key takeaways, and source citations.",
            "Write in Chinese.",
        ],
    )

    def _get_cached_search(self, topic: str) -> Optional[SearchResults]:
        """Check session_state cache for search results."""
        cache = self.session_state.get("search_cache", {})
        if topic in cache:
            try:
                results = SearchResults.model_validate(cache[topic])
                logger.info(f"Cache hit: {len(results.articles)} search results for '{topic}'")
                return results
            except Exception:
                pass
        return None

    def _get_cached_articles(self) -> Dict[str, ScrapedArticle]:
        """Load scraped articles from session_state cache."""
        cache = self.session_state.get("article_cache", {})
        articles = {}
        for url, data in cache.items():
            try:
                articles[url] = ScrapedArticle.model_validate(data)
            except Exception:
                continue
        if articles:
            logger.info(f"Cache hit: {len(articles)} scraped articles")
        return articles

    def run(self, topic: str):
        """Generate news report with caching."""
        # Step 1: Search (with cache)
        search_results = self._get_cached_search(topic)
        if search_results is None:
            response = self.searcher.run_sync(f"Search latest news about: {topic}")
            if response and isinstance(response.content, SearchResults):
                search_results = response.content
                # Cache search results
                self.session_state.setdefault("search_cache", {})[topic] = search_results.model_dump()

        if not search_results or not search_results.articles:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"No articles found for: {topic}",
            )
            return

        # Step 2: Scrape (with cache, incremental)
        scraped = self._get_cached_articles()
        for article in search_results.articles:
            if article.url in scraped:
                continue  # Already scraped
            response = self.scraper.run_sync(article.url)
            if response and isinstance(response.content, ScrapedArticle):
                scraped[response.content.url] = response.content

        # Update cache
        self.session_state["article_cache"] = {url: a.model_dump() for url, a in scraped.items()}

        # Step 3: Write report (always re-run, uses latest prompt)
        articles_json = json.dumps([a.model_dump() for a in scraped.values()], ensure_ascii=False, indent=2)
        yield from self.writer.run_sync(
            f"Write a report on '{topic}' based on these articles:\n{articles_json}",
            stream=True,
        )


if __name__ == "__main__":
    topic = "AI Agent最新进展"

    workflow = NewsReportWorkflow(
        session_id=f"news-{topic}",
    )

    print("=" * 60)
    print(f"News Report: {topic}")
    print("=" * 60)

    report = workflow.run(topic=topic)
    pprint_run_response(report)
