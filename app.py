import streamlit as st
from dotenv import load_dotenv
from search import Config, IMAGE_DOMAINS, PROVIDERS, search_meme

load_dotenv()

st.set_page_config(page_title="Stampy Trend Scout", page_icon="🕵️", layout="wide")

st.title("🕵️ Stampy Trend Scout")
st.caption("Ask about any meme, trend, or cultural reference — Stampy will find it.")

with st.sidebar:
    st.header("Controls")
    provider_key = st.selectbox(
        "Model",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: PROVIDERS[k]["label"],
        index=0,
    )
    use_classifier = st.toggle(
        "Use light LLM classifier",
        value=False,
        help="On: one small LLM call picks topic/time_range and rewrites the query (e.g. appends the current year). Off: Tavily auto_parameters does the classification.",
    )
    use_pytrends = st.toggle(
        "Attach Google Trends signal (pytrends)",
        value=False,
        disabled=True,
        help="Disabled for now — pytrends integration in progress.",
    )
    synthesize = st.toggle(
        "LLM synthesis for explanation",
        value=False,
        help="Disabled for now — using Tavily's built-in answer.",
    )
    freshness_rerank = st.toggle(
        "Freshness re-rank sources",
        value=False,
        help="Multiplicative time decay on Tavily's relevance score.",
    )

    with st.expander("Advanced search"):
        search_depth = st.selectbox(
            "Tavily search_depth", ["advanced", "basic", "fast", "ultra-fast"], index=0
        )
        max_results = st.slider("max_results per call", 1, 20, 4)
        topic_override = st.selectbox(
            "topic override", ["(auto)", "news", "general"], index=0
        )
        time_range_override = st.selectbox(
            "time_range override", ["(auto)", "day", "week", "month", "year"], index=0
        )
        exact_match_override = st.selectbox(
            "exact_match override", ["(auto)", "on", "off"], index=0
        )
        half_life_days = st.slider(
            "Freshness half-life (days)", 1, 60, 14, disabled=not freshness_rerank
        )
        include_image_descriptions = st.toggle(
            "Tavily image descriptions (vision captions)",
            value=True,
            help="Tavily runs a vision model on context-call images and returns captions. Off = save Tavily credits. (Image-call descriptions are already disabled — Tavily returns empty for narrow-domain searches.)",
        )
        domains_text = st.text_area(
            "Image search include_domains (comma-separated)",
            value=", ".join(IMAGE_DOMAINS),
        )

query = st.text_input(
    "What meme or trend are you looking for?",
    placeholder="e.g. Justin Bieber at Coachella",
)

if st.button("Find your thing", disabled=not query, type="primary"):
    cfg = Config(
        provider=provider_key,
        use_classifier=use_classifier,
        use_pytrends=use_pytrends,
        synthesize=synthesize,
        freshness_rerank=freshness_rerank,
        search_depth=search_depth,
        max_results=max_results,
        topic_override=None if topic_override == "(auto)" else topic_override,
        time_range_override=None if time_range_override == "(auto)" else time_range_override,
        exact_match_override=(
            None if exact_match_override == "(auto)" else exact_match_override == "on"
        ),
        freshness_half_life_days=half_life_days,
        image_domains=[d.strip() for d in domains_text.split(",") if d.strip()],
        include_image_descriptions=include_image_descriptions,
    )

    with st.spinner("Scouring the internet for your dankest thing..."):
        try:
            results = search_meme(query, cfg)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # Trend signal badge
    ts = results.get("trend_signal")
    if ts and ts.get("status") == "ok":
        cur, peak = ts.get("current"), ts.get("peak_7d")
        ratio = ts.get("ratio")
        trend_bits = [f"Google Trends 7d: **{cur}** now vs peak **{peak}**"]
        if ratio is not None:
            trend_bits.append(f"({ratio*100:.0f}% of peak)")
        st.info(" ".join(trend_bits))
        rising = ts.get("rising_related") or []
        if rising:
            st.caption("Rising related: " + ", ".join(r.get("query", "") for r in rising[:5]))
    elif ts and ts.get("status") != "ok":
        st.caption(f"pytrends: {ts.get('status')}")

    st.subheader("What Stampy knows")
    st.write(results.get("explanation") or "No explanation generated.")

    images = results.get("images") or []
    image_meta = {i["url"]: i for i in results.get("image_meta", [])}
    if images:
        st.subheader(f"Related images ({len(images)})")
        st.caption(
            "If a tile is blank, the host is hotlink-protected (Reddit, X, some CDNs "
            "block cross-origin embeds). Use the Open ↗ link to view directly."
        )
        cols = st.columns(3)
        for idx, url in enumerate(images[:13]):
            with cols[idx % 3]:
                meta_entry = image_meta.get(url, {})
                caption_bits = []
                if meta_entry.get("description"):
                    caption_bits.append(meta_entry["description"])
                if meta_entry.get("source_call"):
                    caption_bits.append(f"via {meta_entry['source_call']}")
                caption = " · ".join(caption_bits) if caption_bits else None
                st.image(url, caption=caption, width="stretch")
                host = url.split("/")[2] if "://" in url else url
                st.markdown(f"[Open ↗]({url}) · `{host}`")
    else:
        st.warning("No images found. Try different wording or toggle `topic` to 'general'.")

    # Per-tool breakdown — what each toggle/tool actually returned
    tool_calls = results.get("tool_calls", {})
    if tool_calls:
        st.subheader("Per-tool breakdown")
        cols = st.columns(2)

        with cols[0]:
            pt = tool_calls.get("pytrends", {})
            st.markdown(f"**pytrends** — {'on' if pt.get('enabled') else 'off'}")
            if pt.get("enabled"):
                st.json(pt.get("result") or {"status": "no_result"})
            else:
                st.caption("Toggle off — no call made.")

            clf = tool_calls.get("classifier", {})
            st.markdown(f"**Classifier** — {'on' if clf.get('enabled') else 'off'} ({clf.get('model') or '—'})")
            if clf.get("enabled"):
                st.json(clf.get("result") or {})
            else:
                st.caption("Toggle off — Tavily auto_parameters used instead.")

            syn = tool_calls.get("synthesis", {})
            st.markdown(f"**LLM synthesis** — {'on' if syn.get('enabled') else 'off'} ({syn.get('model') or '—'})")
            if not syn.get("enabled"):
                st.caption("Toggle off — used Tavily's built-in `answer` instead.")

        with cols[1]:
            tc = tool_calls.get("tavily_context", {})
            st.markdown(f"**Tavily — context search** ({tc.get('source_count', 0)} sources · {tc.get('image_count', 0)} images)")
            st.caption(f"query: `{tc.get('query', '')}`")
            st.json(tc.get("params", {}))
            if tc.get("answer"):
                st.caption(f"Tavily answer: {tc['answer'][:200]}{'…' if len(tc['answer']) > 200 else ''}")

            ti = tool_calls.get("tavily_image", {})
            st.markdown(f"**Tavily — image search** ({ti.get('source_count', 0)} sources · {ti.get('image_count', 0)} images)")
            st.caption(f"query: `{ti.get('query', '')}`")
            st.json(ti.get("params", {}))

    meta = results.get("search_meta", {})
    with st.expander("Overall search_meta"):
        st.json(meta)

    sources = results.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                pub = s.get("published_date") or "undated"
                fresh = s.get("fresh_score")
                score = s.get("score", 0.0)
                call = s.get("source_call", "—")
                rank_bits = f"score {score:.2f}"
                if fresh is not None:
                    rank_bits += f" · fresh {fresh:.2f}"
                rank_bits += f" · via `{call}`"
                st.markdown(f"**[{s['title']}]({s['url']})** — _{pub}_ · {rank_bits}")
                content = s.get("content") or ""
                st.caption(content[:200] + "…" if len(content) > 200 else content)
