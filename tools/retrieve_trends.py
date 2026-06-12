from retrieval import retrieve

SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_trends",
        "description": (
            "Semantic search over approved trend briefs in the internal knowledge base. "
            "Use when the user asks about trends, memes, or cultural moments — "
            "especially when a time reference like 'this week', 'last month', or a specific date is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query rewritten for semantic similarity",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of trend briefs to retrieve (default 25)",
                },
                "date_from": {
                    "type": "string",
                    "description": "ISO date string YYYY-MM-DD — filter briefs from this date",
                },
                "date_to": {
                    "type": "string",
                    "description": "ISO date string YYYY-MM-DD — filter briefs up to this date",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(query: str, top_k: int = 25, date_from: str | None = None, date_to: str | None = None) -> list[dict]:
    return retrieve(query, top_k=top_k, date_from=date_from, date_to=date_to)
