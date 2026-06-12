import json
import logging
import os
from datetime import date, timedelta
from openai import OpenAI
from tools import TOOLS, dispatch

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

def _system_prompt() -> str:
    today = date.today()
    return f"""You are Stampy, a trend intelligence assistant. You help users discover trends, memes, cultural moments, and emerging trends.
    Your answers are user facing, so keep them as you are talking to a user, be a natural assistant and try to be concise.

Today's date is {today.isoformat()}.

You have two tools:
1. retrieve_trends — searches the internal Supabase knowledge base of approved trend briefs
2. search_web — searches the live web via Tavily for current results

Always start with retrieve_trends.
- If it returns several relevant results, answer from those alone.
- Only call search_web if retrieve_trends returns nothing, or returns results that don't actually address the user's question. In that case, try search_web with the same (or a rephrased) query before telling the user nothing was found.

When the user mentions a time reference, translate it to ISO dates (YYYY-MM-DD):
- "today" → date_from and date_to = {today.isoformat()}
- "this week" → date_from = {(today - timedelta(days=7)).isoformat()}, date_to = {today.isoformat()}
- "last month" → date_from = {(today - timedelta(days=30)).isoformat()}, date_to = {today.isoformat()}
- "this year" → date_from = {today.replace(month=1, day=1).isoformat()}, date_to = {today.isoformat()}

Be concise, casual, and specific. Always ground your answer in what the tools return — never answer from memory."""


def run(query: str, model: str = "gpt-5-mini") -> tuple[str, list[dict]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": query},
    ]
    collected_images: list[dict] = []

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "", collected_images

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            logger.info("TOOL CALL: %s | args: %s", call.function.name, args)
            result = dispatch(call.function.name, args)
            if isinstance(result, dict):
                chunk_count = len(result.get("sources", result.get("chunks", [])))
                image_count = len(result.get("images", []))
                logger.info("TOOL RESULT: %s | sources/chunks: %d | images: %d", call.function.name, chunk_count, image_count)
                if "images" in result:
                    collected_images.extend(result["images"])
            elif isinstance(result, list):
                logger.info("TOOL RESULT: %s | %d items returned", call.function.name, len(result))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
