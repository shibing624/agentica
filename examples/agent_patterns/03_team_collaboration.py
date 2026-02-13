# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Team collaboration demo - Demonstrates multi-agent collaboration

This example shows how to create a team of agents that work together:
1. Researcher agent - Searches and analyzes articles
2. Writer agent - Writes articles based on research
3. Team coordinator - Orchestrates the collaboration
"""
import sys
import os
from textwrap import dedent
from typing import Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio

from agentica import Agent, OpenAIChat, BaiduSearchTool
from agentica.agent.config import PromptConfig


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


# Create researcher agent
researcher = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Article Researcher",
    tools=[BaiduSearchTool()],
    description="Given a topic, search for 10 articles and return the 5 most relevant articles.",
    response_model=NewsArticle,
)

# Create writer agent
writer = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Article Writer",
    description="You are a Senior NYT Editor and your task is to write a NYT cover story worthy article.",
    instructions=[
        "You will be provided with news articles and their links.",
        "Carefully read each article and think about the contents",
        "Then generate a final New York Times worthy article in the <article_format> provided below.",
        "Break the article into sections and provide key takeaways at the end.",
        "Make sure the title is catchy and engaging.",
        "Give the section relevant titles and provide details/facts/processes in each section.",
        "Ignore articles that you cannot read or understand.",
        "REMEMBER: you are writing for the New York Times, so the quality of the article is important.",
    ],
    prompt_config=PromptConfig(
        expected_output=dedent(
            """\
    An engaging, informative, and well-structured article in the following format:
    <article_format>
    ## Engaging Article Title

    ### Overview
    {give a brief introduction of the article and why the user should read this report}
    {make this section engaging and create a hook for the reader}

    ### Section 1
    {break the article into sections}
    {provide details/facts/processes in this section}

    ... more sections as necessary...

    ### Takeaways
    {provide key takeaways from the article}

    ### References
    - [Title](url)
    - [Title](url)
    </article_format>
    """
        ),
    ),
)

# Create team coordinator
team = Agent(
    name="News Article Team",
    team=[researcher, writer],
    debug_mode=True,
)


async def main():
    print("=" * 60)
    print("Team Collaboration Demo: News Article Writing")
    print("=" * 60)
    
    await team.print_response_stream(
        """
        Find the 5 most relevant articles on a topic: 人工智能最新发展,
        Read each article and write a NYT worthy news article. 用中文写。
        """,
        save_response_to_file="outputs/team_article.md",
    )


if __name__ == "__main__":
    asyncio.run(main())
