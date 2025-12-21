# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: News article workflow demo - Multi-step news report generation

This example demonstrates a complex workflow that:
1. Searches for news articles on a topic
2. Scrapes and processes article content
3. Generates a comprehensive news report

pip install newspaper4k
"""
import sys
import os
from textwrap import dedent
from typing import Optional, Dict, Iterator
from pydantic import BaseModel, Field
from loguru import logger
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.workflow import Workflow
from agentica import RunResponse, RunEvent, pprint_run_response
from agentica.db.sqlite import SqliteDb
from agentica.tools.newspaper_tool import NewspaperTool
from agentica.tools.baidu_search_tool import BaiduSearchTool


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


class SearchResults(BaseModel):
    articles: list[NewsArticle]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")
    content: Optional[str] = Field(
        ...,
        description="Content of the in markdown format if available.",
    )


class NewsReportGenerator(Workflow):
    """Generate a comprehensive news report on a given topic."""

    description: str = "Generate a comprehensive news report on a given topic."

    web_searcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[BaiduSearchTool()],
        instructions=[
            "Given a topic, search for 10 articles and return the 5 most relevant articles.",
        ],
        response_model=SearchResults,
    )

    article_scraper: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[NewspaperTool()],
        instructions=[
            "Given a url, scrape the article and return the title, url, and markdown formatted content.",
            "If the content is not available or does not make sense, return None as the content.",
        ],
        response_model=ScrapedArticle,
    )

    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description="You are a Senior NYT Editor writing a cover story.",
        instructions=[
            "You will be provided with news articles and their contents.",
            "Carefully read each article and think about the contents",
            "Then generate a final New York Times worthy article.",
            "Break the article into sections and provide key takeaways at the end.",
            "Make sure the title is catchy and engaging.",
            "Always provide sources for the article.",
        ],
        expected_output=dedent("""\
        An engaging, informative, and well-structured article:
        
        ## Engaging Article Title

        ### Overview
        {brief introduction and hook for the reader}

        ### Section 1
        {details and facts}

        ... more sections as necessary...

        ### Key Takeaways
        {key takeaways from the article}

        ### Sources
        - [Title](url)
        """),
    )

    def get_search_results(self, topic: str, use_cache: bool) -> Optional[SearchResults]:
        """Search for articles on the topic."""
        search_results: Optional[SearchResults] = None

        if use_cache and "search_results" in self.session_state:
            if topic in self.session_state["search_results"]:
                try:
                    search_results = SearchResults.model_validate(
                        self.session_state["search_results"][topic]
                    )
                    logger.info(f"Found {len(search_results.articles)} articles in cache.")
                except Exception as e:
                    logger.warning(f"Could not read search results from cache: {e}")

        if search_results is None:
            response = self.web_searcher.run(topic)
            if response and response.content and isinstance(response.content, SearchResults):
                logger.info(f"Found {len(response.content.articles)} articles.")
                search_results = response.content

        if search_results is not None:
            if "search_results" not in self.session_state:
                self.session_state["search_results"] = {}
            self.session_state["search_results"][topic] = search_results.model_dump()

        return search_results

    def scrape_articles(
        self, search_results: SearchResults, use_cache: bool
    ) -> Dict[str, ScrapedArticle]:
        """Scrape content from found articles."""
        scraped_articles: Dict[str, ScrapedArticle] = {}

        if use_cache and "scraped_articles" in self.session_state:
            for url, article in self.session_state["scraped_articles"].items():
                try:
                    validated = ScrapedArticle.model_validate(article)
                    scraped_articles[validated.url] = validated
                except Exception as e:
                    logger.warning(f"Could not read scraped article from cache: {e}")
            logger.info(f"Found {len(scraped_articles)} scraped articles in cache.")

        for article in search_results.articles:
            if article.url in scraped_articles:
                continue

            response = self.article_scraper.run(article.url)
            if response and response.content and isinstance(response.content, ScrapedArticle):
                scraped_articles[response.content.url] = response.content
                logger.info(f"Scraped article: {response.content.url}")

        if "scraped_articles" not in self.session_state:
            self.session_state["scraped_articles"] = {}
        for url, article in scraped_articles.items():
            self.session_state["scraped_articles"][url] = article.model_dump()

        return scraped_articles

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
    ) -> Iterator[RunResponse]:
        """Generate a comprehensive news report on a given topic."""
        logger.info(f"Generating a report on: {topic}")

        # Step 1: Search for articles
        search_results = self.get_search_results(topic, use_search_cache)
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        # Step 2: Scrape articles
        scraped_articles = self.scrape_articles(search_results, use_scrape_cache)

        # Step 3: Write the report
        logger.info("Writing news report")
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()]
        }
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)


if __name__ == "__main__":
    topic = "AI最新进展"

    url_safe_topic = topic.lower().replace(" ", "-")

    workflow = NewsReportGenerator(
        session_id=f"news-report-{url_safe_topic}",
    )

    report_stream = workflow.run(topic=topic)
    pprint_run_response(report_stream)
