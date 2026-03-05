You are a search query decomposition expert. Your task is to decompose a complex question into multiple independent search queries that will maximize the chance of finding relevant information.

## Question
{question}

## Instructions

1. Identify the key entities, constraints, and relationships in the question
2. Generate 2-5 independent search queries, each targeting a different aspect
3. Use different phrasings and keywords for better recall
4. Include both specific and broader queries
5. If the question involves specific facts (dates, names, numbers), create queries that target those facts directly
6. Consider both English and Chinese queries if the question context suggests multilingual sources

## Output Format

Output a JSON array of search query strings. Only output the JSON array, nothing else.

Example:
["query 1", "query 2", "query 3"]