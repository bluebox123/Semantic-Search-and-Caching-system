Trademarkia - AI&ML Engineer Task

Submission Requirement
Please provide the GitHub repository link and the project link for your solution. Ensure access is granted to recruitments@trademarkia.com.
Assignment Submission Link - https://forms.gle/4RpHZpAi8rbG9QCE8

Problem Statement
You are given the 20 Newsgroups dataset: a collection of ~20,000 news posts spanning 20 overlapping topic categories. Your task is to build a lightweight semantic search system with three core components:
Fuzzy clustering of the corpus using vector embeddings and a vector database
A semantic cache layer that avoids redundant computation on similar queries built from first principles, without Redis or any dedicated caching middleware
A FastAPI service that exposes the cache as a live API endpoint with proper state management
This task is intentionally open-ended in places. Your design decisions and how you justify them matter as much as the code.
Dataset link: https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
Part 1 - Embedding & Vector Database Setup
Prepare the corpus for semantic analysis and persist it in a way that supports efficient filtered retrieval.
The dataset is noisy. Make deliberate choices about what to keep and what to discard — your code comments are the only place you need to justify these decisions, so make them count. The same applies to your choice of embedding model and vector store: there is no single right answer, but there are better and worse fits for this problem. Your choices should reflect that you understand what this system needs to do downstream.
Part 2 - Fuzzy Clustering
The 20 Newsgroups dataset has 20 labelled categories — but the real semantic structure of the corpus is messier than that. Your task is to uncover it.
Hard cluster assignments are not acceptable here. A document about gun legislation does not belong to either politics or firearms — it belongs to both, to varying degrees. Your clustering output should reflect this. For each document, the result should be a distribution, not a label.
The number of clusters is your decision. Justify it with evidence, not convenience.
Once clustered, your analysis should convince a sceptical reader that the clusters are semantically meaningful. Show what lives in them, show what sits at their boundaries, and show where the model is genuinely uncertain, those boundary cases are often the most interesting.
Part 3 - Semantic Cache
A traditional cache breaks the moment two users ask the same question in different words. Build something better.
Your cache should be able to recognise that a query it has seen before even if phrased differently does not need to be recomputed. The mechanism for deciding whether two queries are "close enough" is yours to design. So is the data structure that holds the cache.
The cluster structure you built in Part 2 should be doing real work here. Think about what it means for cache lookup efficiency when your cache grows large, and how cluster membership changes that problem.
There is one tunable decision at the heart of this component. Explore it. The interesting question is not which value performs best, it is what each value reveals about the system's behaviour. That explicit value is what determines how good your system is, and is where heuristics can’t help you.
No Redis, Memcached, or any caching library. If you didn't write it, it shouldn't be in your cache.
Part 4 - FastAPI Service
Implement the following endpoints:
POST /query
Accepts a JSON body: { "query": "<natural language query>" }
Embeds the query, checks the semantic cache, and returns:

{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
On a miss, compute and store the result before returning

GET /cache/stats
Returns current cache state:
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
DELETE /cache
Flushes the cache entirely and resets all stats

Note: Provide a venv environment for running your code, and ensure your service should start cleanly with a single uvicorn command


Bonus - Dockerisation
This is optional. It will not penalise you if skipped, but a clean submission here signals production readiness.
Write a Dockerfile that containerises your FastAPI service
Your container should start the uvicorn server on port 8000
A docker-compose.yml is welcome but not required