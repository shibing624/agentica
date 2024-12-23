"""
The research Assistant searches for EXA for a topic
and writes an article in Markdown format.
"""

import sys
from textwrap import dedent

sys.path.append('..')
from agentica import Agent
from agentica import OpenAIChat
from agentica.tools.search_serper_tool import SearchSerperTool

m = Agent(
    model=OpenAIChat(model='gpt-4o-mini'),
    tools=[SearchSerperTool()],
    description="You are a senior NYT researcher writing an article on a topic.中文撰写报告",
    instructions=[
        "For the provided topic, run 3 different searches.",
        "Read the results carefully and prepare a NYT worthy article.",
        "Focus on facts and make sure to provide references. 中文撰写报告",
    ],
    add_datetime_to_instructions=True,
    expected_output=dedent(
        """\
    An engaging, informative, and well-structured article in markdown format:

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
    - [Reference 1](link)
    - [Reference 2](link)
    - [Reference 3](link)

    ### About the Author
    {write a made up for yourself, give yourself a cyberpunk name and a title}

    - published on {date} in dd/mm/yyyy
    """
    ),
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)
r = m.run("苹果 WWDC 发布会")
print(r)
