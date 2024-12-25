import os
import argparse
import re
import time
import logging
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session

from .... import eden_utils
from ....agent import Agent
from .. import X


async def handler(args: dict, db: str):
    agent = Agent.load(args["agent"], db=db)
    
    x = X(agent)

    tweet = x.post(
        tweet_text=args["content"],
        media_urls=args["images"]
    )
    print(tweet)

    return {
        "output": tweet['data']
    }
    