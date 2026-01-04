import asyncio
import os

import requests

from eve.tool import ToolContext


async def handler(context: ToolContext):
    NEWSAPI_API_KEY = os.environ["NEWSAPI_API_KEY"]

    category = context.args["subject"]
    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={NEWSAPI_API_KEY}"

    response = await asyncio.to_thread(requests.get, url)
    news = response.json()
    articles = [a for a in news["articles"] if a["title"] != "[Removed]"]

    news_summary = "# News Summary:\n\n"
    for article in articles:
        news_summary += (
            f"Title: {article['title']}\nDescription: {article['description']}\n\n"
        )

    return {"output": news_summary}
