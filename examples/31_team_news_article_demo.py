# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import sys
from textwrap import dedent
from typing import Optional

from pydantic import BaseModel, Field

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica.tools.search_exa_tool import SearchExaTool


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


researcher = Agent(
    model=AzureOpenAIChat(model="gpt-4o"),
    name="Article Researcher",
    tools=[SearchExaTool()],
    description="Given a topic, search for 15 articles and return the 7 most relevant articles.",
    output_model=NewsArticle,
)

writer = Agent(
    model=AzureOpenAIChat(model="gpt-4o"),
    name="Article Writer",
    save_response_to_file="outputs/article.md",
    description="You are a Senior NYT Editor and your task is to write a NYT cover story worthy article due tomorrow.",
    instructions=[
        "You will be provided with news articles and their links.",
        "Carefully read each article and think about the contents",
        "Then generate a final New York Times worthy article in the <article_format> provided below.",
        "Break the article into sections and provide key takeaways at the end.",
        "Make sure the title is catchy and engaging.",
        "Give the section relevant titles and provide details/facts/processes in each section."
        "Ignore articles that you cannot read or understand.",
        "REMEMBER: you are writing for the New York Times, so the quality of the article is important.",
    ],
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
    - [Title](url)
    </article_format>
    """
    ),
)

team = Agent(
    name="write Article",
    team=[researcher, writer],
    show_tool_calls=True,
    debug_mode=True,
)
team.print_response(
    """
    Find the 7 most relevant articles on a topic: 美国继续制裁俄罗斯,
    Read each article and and write a NYT worthy news article. 用中文写。
    """,
    stream=True,
    markdown=True,
)
