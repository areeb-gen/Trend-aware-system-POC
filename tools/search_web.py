import os
from tavily import TavilyClient

SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Live web search for current trends, memes, news, and cultural moments using Tavily. "
            "Use this to find recent information not in the internal knowledge base — "
            "breaking news, viral moments, or anything happening right now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query optimized for web search",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Limit results to this recency window",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(query: str, time_range: str | None = None, max_results: int = 5) -> dict:
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    kwargs: dict = dict(
        query=query,
        search_depth="advanced",
        include_answer=True,
        include_images=True,
        include_image_descriptions=True,
        max_results=max_results,
    )
    if time_range:
        kwargs["time_range"] = time_range

    result = tavily.search(**kwargs)

    raw_images = result.get("images", [])
    images = []
    for img in raw_images:
        if isinstance(img, dict) and img.get("url"):
            images.append({"url": img["url"], "description": img.get("description", "")})
        elif isinstance(img, str):
            images.append({"url": img, "description": ""})

    sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "published_date": r.get("published_date"),
            "score": float(r.get("score") or 0.0),
        }
        for r in result.get("results", [])
    ]

    return {"sources": sources, "images": images, "answer": result.get("answer")}
