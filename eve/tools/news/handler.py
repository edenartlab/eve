import os

import httpx

from eve.tool import ToolContext


async def handler(context: ToolContext):
    NEWSAPI_API_KEY = os.environ["NEWSAPI_API_KEY"]

    category = context.args["subject"]
    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={NEWSAPI_API_KEY}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        news = response.json()

    articles = [a for a in news["articles"] if a["title"] != "[Removed]"]

    news_summary = "# News Summary:\n\n"
    for article in articles:
        news_summary += (
            f"Title: {article['title']}\nDescription: {article['description']}\n\n"
        )

    return {"output": news_summary}
