from ._embed import embed_query
from ._search import semantic_search


def retrieve(
    query: str,
    top_k: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    embedding = embed_query(query)
    return semantic_search(embedding, top_k, date_from=date_from, date_to=date_to)
