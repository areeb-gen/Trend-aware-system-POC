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
    return f"""You are Stampy, a trend intelligence assistant. You help users discover memes, cultural moments, and emerging trends.

Today's date is {today.isoformat()}.

You have access to an internal knowledge base of approved trend briefs. Always use the retrieve_trends tool to answer questions about trends, memes, or cultural moments — never answer from memory.

When the user mentions a time reference, translate it to ISO dates (YYYY-MM-DD):
- "today" → date_from and date_to = {today.isoformat()}
- "this week" → date_from = {(today - timedelta(days=7)).isoformat()}, date_to = {today.isoformat()}
- "last month" → date_from = {(today - timedelta(days=30)).isoformat()}, date_to = {today.isoformat()}
- "this year" → date_from = {today.replace(month=1, day=1).isoformat()}, date_to = {today.isoformat()}

Be concise, casual, and specific. Always ground your answer in what the tool returns."""


def run(query: str, model: str = "gpt-4o-mini") -> tuple[str, list[dict]]:
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
