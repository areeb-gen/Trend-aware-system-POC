import os
from supabase import create_client, Client

_supabase: Client | None = None


def _client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _supabase


def semantic_search(
    embedding: list[float],
    top_k: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    params: dict = {"query_embedding": embedding, "match_count": top_k}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    result = (
        _client()
        .postgrest
        .schema("public")
        .rpc("match_trend_items", params)
        .execute()
    )
    return result.data or []
