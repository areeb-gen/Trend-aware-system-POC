import os
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tavily import TavilyClient

TODAY = date.today()
TODAY_STR = TODAY.strftime("%B %d, %Y")

IMAGE_DOMAINS = [
    "imgur.com",
    "reddit.com",
    "knowyourmeme.com",
    "api.meme.com",
    "tenor.com",
    "giphy.com",
]

PROVIDERS = {
    "openai": {
        "label": "OpenAI — gpt-4o-mini",
        "model": "gpt-4o-mini",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "xai": {
        "label": "xAI — grok-4.20-non-reasoning",
        "model": "grok-4.20-non-reasoning",
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
    },
}


@dataclass
class Config:
    provider: str = "openai"
    use_classifier: bool = True
    use_pytrends: bool = False
    search_depth: str = "advanced"
    max_results: int = 13
    topic_override: str | None = None
    time_range_override: str | None = None
    exact_match_override: bool | None = None
    freshness_rerank: bool = True
    freshness_half_life_days: int = 14
    synthesize: bool = True
    image_domains: list[str] = field(default_factory=lambda: list(IMAGE_DOMAINS))
    include_image_descriptions: bool = True  # Tavily vision-model captions (context call only)


def _make_client(provider: str) -> tuple[OpenAI, str]:
    cfg = PROVIDERS[provider]
    kwargs: dict = {"api_key": os.environ[cfg["env_key"]]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg["model"]


def _classify_query(client: OpenAI, model: str, query: str) -> dict:
    """One small LLM call: topic, time_range, query rewrite for year disambiguation."""
    system = (
        f"Today's date is {TODAY_STR}. You classify user queries about memes, trends, and news "
        "for a web search pipeline. Output JSON only."
    )
    user = f"""Query: "{query}"

Return a JSON object with these keys:
- intent: "breaking_news" | "recent_trend" | "classic_meme" | "general"
- topic: "news" | "general"   (news = current events, celebrity news, same-day happenings)
- time_range: "day" | "week" | "month" | "year" | null   (null for evergreen/classic content)
- rewritten_query: query optimized for web search. If the query implies a time-bound event (Coachella, Olympics, an election) but no year is given, append the current year ({TODAY.year}). Keep it natural.
- should_exact_match: true iff the rewritten query contains a specific phrase that must match literally (e.g. a specific year).

Return only the JSON object."""

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        pass

    # Fallback: free text, loose JSON extraction
    try:
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=250)
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return {
        "intent": "general",
        "topic": "general",
        "time_range": None,
        "rewritten_query": query,
        "should_exact_match": False,
    }


def _pytrends_preflight(query: str) -> dict | None:
    """Google Trends velocity + rising related queries. Fails quietly on 429/import errors."""
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError as e:
        return {"status": "not_installed", "error": str(e)}

    try:
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload([query], timeframe="now 7-d")
        iot = pytrends.interest_over_time()
        related = pytrends.related_queries() or {}
        rising_df = related.get(query, {}).get("rising")

        if iot is None or iot.empty:
            return {"status": "no_data"}

        peak = int(iot[query].max())
        current = int(iot[query].iloc[-1])
        rising = rising_df.head(5).to_dict("records") if rising_df is not None else []
        return {
            "status": "ok",
            "peak_7d": peak,
            "current": current,
            "ratio": (current / peak) if peak else None,
            "rising_related": rising,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def _score_freshness(published_date: str | None, base_score: float, half_life_days: int) -> float:
    if not published_date:
        return base_score * 0.5
    try:
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        age_days = max((datetime.now(timezone.utc) - pub).days, 0)
        return base_score * (0.5 ** (age_days / half_life_days))
    except Exception:
        return base_score * 0.5


def _run_tavily(
    query: str,
    topic: str | None,
    time_range: str | None,
    include_domains: list[str] | None,
    search_depth: str,
    max_results: int,
    exact_match: bool,
    auto_parameters: bool,
    include_image_descriptions: bool = True,
) -> dict:
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    kwargs: dict = dict(
        query=query,
        search_depth=search_depth,
        include_images=True,
        include_image_descriptions=include_image_descriptions,
        include_answer=True,
        max_results=max_results,
    )
    if topic:
        kwargs["topic"] = topic
    if time_range:
        kwargs["time_range"] = time_range
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exact_match:
        kwargs["exact_match"] = True
    if auto_parameters:
        kwargs["auto_parameters"] = True

    result = tavily.search(**kwargs)

    raw_images = result.get("images", [])
    images: list[dict] = []
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
    return {"images": images, "sources": sources, "answer": result.get("answer")}


def _synthesize(
    client: OpenAI,
    model: str,
    query: str,
    sources: list[dict],
    trend_signal: dict | None,
) -> str:
    top = sources[:6]
    context = "\n\n".join(
        f"[{s['title']}] {s['url']} ({s.get('published_date') or 'undated'})\n{s['content']}"
        for s in top
        if s.get("content")
    )
    trend_line = ""
    if trend_signal and trend_signal.get("status") == "ok":
        cur = trend_signal.get("current")
        peak = trend_signal.get("peak_7d")
        trend_line = f"\nGoogle Trends (7d): current interest {cur} vs peak {peak}."

    prompt = (
        f"Today's date is {TODAY_STR}.\n"
        f'User query: "{query}"{trend_line}\n\n'
        f"Sources:\n{context or '(no content)'}\n\n"
        "Write 2–3 casual sentences explaining what this trend/meme is, where it started, and why it resonates. "
        "Call out any recency or engagement signal evident in the sources."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


def search_meme(query: str, config: Config | None = None) -> dict:
    cfg = config or Config()
    client, model = _make_client(cfg.provider)

    # 1. Optional pytrends pre-flight
    trend_signal = _pytrends_preflight(query) if cfg.use_pytrends else None

    # 2. Classify query (light LLM) OR let Tavily auto_parameters decide
    if cfg.use_classifier:
        classification = _classify_query(client, model, query)
        rewritten = classification.get("rewritten_query") or query
        topic = cfg.topic_override or classification.get("topic")
        time_range = cfg.time_range_override or classification.get("time_range")
        exact_match = (
            cfg.exact_match_override
            if cfg.exact_match_override is not None
            else bool(classification.get("should_exact_match", False))
        )
        auto_parameters = False
    else:
        classification = None
        rewritten = query
        topic = cfg.topic_override
        time_range = cfg.time_range_override
        exact_match = bool(cfg.exact_match_override)
        auto_parameters = True

    # 3. Parallel Tavily: context search + dedicated image search (same query, different domain scope)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_ctx = ex.submit(
            _run_tavily,
            query=rewritten,
            topic=topic,
            time_range=time_range,
            include_domains=None,
            search_depth=cfg.search_depth,
            max_results=cfg.max_results,
            exact_match=exact_match,
            auto_parameters=auto_parameters,
            include_image_descriptions=cfg.include_image_descriptions,
        )
        fut_img = ex.submit(
            _run_tavily,
            query=rewritten,
            topic=None,
            time_range=time_range,
            include_domains=cfg.image_domains,
            search_depth=cfg.search_depth,
            max_results=cfg.max_results,
            exact_match=False,
            auto_parameters=False,
            include_image_descriptions=False,  # image call returns empty descriptions anyway
        )
        ctx = fut_ctx.result()
        img = fut_img.result()

    # Tag each result with the tool call that produced it
    for s in ctx["sources"]:
        s["source_call"] = "tavily_context"
    for s in img["sources"]:
        s["source_call"] = "tavily_image"
    for i in ctx["images"]:
        i["source_call"] = "tavily_context"
    for i in img["images"]:
        i["source_call"] = "tavily_image"

    # 4. Merge + dedupe (first occurrence wins, provenance tag preserved)
    seen_src: set[str] = set()
    all_sources: list[dict] = []
    for s in ctx["sources"] + img["sources"]:
        if s["url"] and s["url"] not in seen_src:
            seen_src.add(s["url"])
            all_sources.append(s)

    seen_img: set[str] = set()
    all_images: list[dict] = []
    for i in ctx["images"] + img["images"]:
        if i["url"] and i["url"] not in seen_img:
            seen_img.add(i["url"])
            all_images.append(i)

    # 5. Freshness re-rank (decay-only)
    if cfg.freshness_rerank:
        for s in all_sources:
            s["fresh_score"] = _score_freshness(
                s.get("published_date"), s["score"], cfg.freshness_half_life_days
            )
        all_sources.sort(key=lambda s: s.get("fresh_score", 0.0), reverse=True)
    else:
        all_sources.sort(key=lambda s: s.get("score", 0.0), reverse=True)

    # 6. Explanation: LLM synthesis OR Tavily's built-in answer
    explanation = (
        _synthesize(client, model, query, all_sources, trend_signal)
        if cfg.synthesize
        else (ctx.get("answer") or "")
    )

    tool_calls = {
        "pytrends": {
            "enabled": cfg.use_pytrends,
            "result": trend_signal,
        },
        "classifier": {
            "enabled": cfg.use_classifier,
            "model": model if cfg.use_classifier else None,
            "result": classification,
        },
        "tavily_context": {
            "query": rewritten,
            "params": {
                "topic": topic,
                "time_range": time_range,
                "exact_match": exact_match,
                "auto_parameters": auto_parameters,
                "search_depth": cfg.search_depth,
                "max_results": cfg.max_results,
            },
            "source_count": len(ctx["sources"]),
            "image_count": len(ctx["images"]),
            "answer": ctx.get("answer"),
        },
        "tavily_image": {
            "query": rewritten,
            "params": {
                "include_domains": cfg.image_domains,
                "time_range": time_range,
                "search_depth": cfg.search_depth,
                "max_results": cfg.max_results,
            },
            "source_count": len(img["sources"]),
            "image_count": len(img["images"]),
        },
        "synthesis": {
            "enabled": cfg.synthesize,
            "model": model if cfg.synthesize else None,
        },
    }

    return {
        "explanation": explanation,
        "images": [i["url"] for i in all_images],
        "image_meta": all_images,
        "sources": all_sources,
        "trend_signal": trend_signal,
        "classification": classification,
        "tool_calls": tool_calls,
        "search_meta": {
            "provider": cfg.provider,
            "model": model,
            "rewritten_query": rewritten,
            "topic": topic,
            "time_range": time_range,
            "exact_match": exact_match,
            "auto_parameters": auto_parameters,
            "synthesize": cfg.synthesize,
        },
    }
