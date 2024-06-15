import json
from actionflow import Assistant,AzureOpenAILLM

data_analyst = Assistant(
    llm=AzureOpenAILLM(),
    semantic_model=json.dumps(
        {
            "tables": [
                {
                    "name": "movies",
                    "description": "Contains information about movies from IMDB.",
                    "path": "https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
                }
            ]
        }
    ),
)

data_analyst.print_response("What is the average rating of movies? Show me the SQL.", markdown=True)
data_analyst.print_response("Show me a histogram of ratings. Choose a bucket size", markdown=True)
