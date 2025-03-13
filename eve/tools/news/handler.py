import os
import json
import requests
from openai import OpenAI


async def handler(args: dict, user: str = None, requester: str = None):
    NEWSAPI_API_KEY = os.environ['NEWSAPI_API_KEY']

    category = args['subject']
    url = f'https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={NEWSAPI_API_KEY}'

    response = requests.get(url)
    news = response.json()
    articles = [a for a in news['articles'] if a['title'] != "[Removed]"]

    headline = articles[0]
    print(json.dumps(articles, indent=2))
    
    print(json.dumps(headline, indent=2))

    print(headline['content'])

    news_summary = "# News Summary:\n\n"
    for article in articles:
        news_summary += f"Title: {article['title']}\nDescription: {article['description']}\n\n"

    print(news_summary)

    return {
        "output": news_summary
    }

