import os
from openai import OpenAI
from tavily import TavilyClient


def search_meme(query: str) -> dict:
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    result = tavily.search(
        query=f"{query} meme explained origin",
        search_depth="advanced",
        include_images=True,
        include_answer=True,
        max_results=5,
    )

    images = result.get("images", [])
    sources = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in result.get("results", [])
    ]

    context_snippets = "\n\n".join(
        f"[{s['title']}]\n{s['content']}" for s in sources if s["content"]
    )

    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Stampy, a culturally aware assistant who knows memes and internet trends. "
                    "Explain memes casually and accurately, like you're talking to a friend."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A user asked about: \"{query}\"\n\n"
                    f"Web search results:\n{context_snippets}\n\n"
                    "Explain this meme/trend in 2-3 sentences: what it is, where it came from, "
                    "and why it's funny or culturally relevant. If it's very recent or niche, say so."
                ),
            },
        ],
    )
    explanation = response.choices[0].message.content

    return {"explanation": explanation, "images": images, "sources": sources}
