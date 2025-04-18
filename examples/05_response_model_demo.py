from typing import List
from pydantic import BaseModel, Field
from rich.pretty import pprint
import sys

sys.path.append('..')
from agentica import Agent


class MovieScript(BaseModel):
    setting: str = Field(..., description="Provide a nice setting for a blockbuster movie.")
    ending: str = Field(..., description="Ending of the movie. If not available, provide a happy ending.")
    genre: str = Field(
        ..., description="Genre of the movie. If not available, select action, thriller or romantic comedy."
    )
    name: str = Field(..., description="Give a name to this movie")
    characters: List[str] = Field(..., description="Name of characters for this movie.")
    storyline: str = Field(..., description="3 sentence storyline for the movie. Make it exciting!")


m = Agent(
    description="You help write movie scripts.",
    response_model=MovieScript,
)
r = m.run("Write a movie script.")
pprint(r)
print(r.content.name)
