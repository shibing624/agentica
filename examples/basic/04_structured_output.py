# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Structured output demo - Demonstrates how to use Pydantic models for structured output

This example shows how to use response_model to get structured output from the agent.
"""
import sys
import os
from typing import List
from pydantic import BaseModel, Field
from rich.pretty import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent


# Define a Pydantic model for the response
class MovieScript(BaseModel):
    setting: str = Field(..., description="Provide a nice setting for a blockbuster movie.")
    ending: str = Field(..., description="Ending of the movie. If not available, provide a happy ending.")
    genre: str = Field(
        ..., description="Genre of the movie. If not available, select action, thriller or romantic comedy."
    )
    name: str = Field(..., description="Give a name to this movie")
    characters: List[str] = Field(..., description="Name of characters for this movie.")
    storyline: str = Field(..., description="3 sentence storyline for the movie. Make it exciting!")


# Example 1: Movie script generation
print("=" * 60)
print("Example 1: Movie Script Generation")
print("=" * 60)

agent = Agent(
    description="You help write movie scripts.",
    response_model=MovieScript,
)

response = agent.run("Write a movie script about a time traveler.")
pprint(response)
print(f"\nMovie name: {response.content.name}")
print(f"Genre: {response.content.genre}")


# Example 2: Book recommendation
print("\n" + "=" * 60)
print("Example 2: Book Recommendation")
print("=" * 60)


class Book(BaseModel):
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    year: int = Field(..., description="Publication year")
    summary: str = Field(..., description="Brief summary of the book")


class BookRecommendation(BaseModel):
    topic: str = Field(..., description="The topic or genre of books")
    books: List[Book] = Field(..., description="List of recommended books")
    reason: str = Field(..., description="Why these books are recommended")


agent2 = Agent(
    description="You are a book recommendation expert.",
    response_model=BookRecommendation,
)

response2 = agent2.run("推荐3本关于人工智能的书籍")
pprint(response2)

print("\nRecommended books:")
for book in response2.content.books:
    print(f"  - {book.title} by {book.author} ({book.year})")
