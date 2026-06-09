import os
from openai import OpenAI


def embed_query(text: str) -> list[float]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return resp.data[0].embedding
