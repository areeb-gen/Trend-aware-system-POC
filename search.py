import os
import json
import requests
from openai import OpenAI
from tavily import TavilyClient


def _is_valid_image(url: str) -> bool:
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        content_type = r.headers.get("content-type", "")
        content_length = int(r.headers.get("content-length", 1))
        return r.status_code == 200 and content_type.startswith("image/") and content_length > 1000
    except Exception:
        return False


def _run_tavily(query: str, days: int | None, include_domains: list[str]) -> dict:
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    kwargs = dict(
        query=query,
        search_depth="advanced",
        include_images=True,
        include_answer=True,
        max_results=13,
    )
    if days:
        kwargs["days"] = days
    if include_domains:
        kwargs["include_domains"] = include_domains

    result = tavily.search(**kwargs)

    raw_images = result.get("images", [])
    images = [url for url in raw_images if _is_valid_image(url)]
    sources = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in result.get("results", [])
    ]
    return {"images": images, "sources": sources}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for memes, trends, or cultural references. "
                "Use 'days' to restrict to recent results when the user wants trending/new content. "
                "Use 'include_domains' to target specific platforms like Reddit, Twitter/X, TikTok, or KnowYourMeme."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query optimized for finding meme/trend context and images",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Limit results to last N days. Use 3-7 for trending, omit for classic/evergreen memes.",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these domains e.g. ['reddit.com', 'instagram.com', 'twitter.com', 'tiktok.com', 'knowyourmeme.com']",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def search_meme(query: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {
            "role": "system",
            "content": (
                "You are Stampy, a culturally aware assistant who specializes in memes and internet trends. "
                "When a user asks about a meme or trend, use the search_web tool to find relevant information. "
                "If they want something trending or new, set 'days' to 3-7 and target Reddit, Twitter, or TikTok. "
                "If they want a classic meme, search broadly without a day limit. "
                "After getting results, explain the meme/trend in 2-3 casual sentences: what it is, where it came from, "
                "and why it resonates. Surface any trending signals you find (platforms, post counts, engagement)."
            ),
        },
        {"role": "user", "content": query},
    ]

    # First call — LLM decides search parameters
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Execute Tavily with LLM-chosen params
    search_results = _run_tavily(
        query=args["query"],
        days=args.get("days"),
        include_domains=args.get("include_domains", []),
    )

    # Second call — LLM synthesizes results
    context = "\n\n".join(
        f"[{s['title']}] {s['url']}\n{s['content']}" for s in search_results["sources"] if s["content"]
    )
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": context or "No results found.",
    })

    final = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=messages,
    )

    return {
        "explanation": final.choices[0].message.content,
        "images": search_results["images"],
        "sources": search_results["sources"],
        "search_params": args,
    }
