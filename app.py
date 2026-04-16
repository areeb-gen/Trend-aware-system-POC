import streamlit as st
from dotenv import load_dotenv
from search import search_meme

load_dotenv()

st.set_page_config(page_title="Stampy Trend Scout", page_icon="🕵️", layout="wide")

st.title("🕵️ Stampy Trend Scout")
st.caption("Ask about any meme, trend, or cultural reference — Stampy will find it.")

query = st.text_input("What meme or trend are you looking for?", placeholder="e.g. Trump as Jesus meme")

if st.button("Search", disabled=not query):
    with st.spinner("Searching the internet..."):
        try:
            results = search_meme(query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    st.subheader("What Stampy knows")
    if results["explanation"]:
        st.write(results["explanation"])
    else:
        st.write("No explanation found — see sources below.")

    if results["images"]:
        st.subheader("Related images")
        cols = st.columns(min(len(results["images"]), 3))
        for i, img_url in enumerate(results["images"][:6]):
            with cols[i % 3]:
                st.image(img_url, use_container_width=True)

    if results["sources"]:
        with st.expander("Sources"):
            for s in results["sources"]:
                st.markdown(f"**[{s['title']}]({s['url']})**")
                st.caption(s["content"][:200] + "..." if len(s["content"]) > 200 else s["content"])
