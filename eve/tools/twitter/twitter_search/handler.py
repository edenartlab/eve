import re
import json
import math
from operator import itemgetter
from typing import List, Dict, Tuple

from pydantic import BaseModel, Field, validator

from eve.agent.thread import UserMessage
from eve.agent.llm import async_prompt
from eve.agent import Agent
from eve.user import User
from eve.deploy import Deployment
from .. import X


# ───────────────────────────────────────────────
# 1. LLM output schema
# ───────────────────────────────────────────────
class TwitterSearchQuery(BaseModel):
    query: str = Field(..., description="Twitter advanced search")

    @validator("query")
    def needs_or_or_from(cls, v):
        if "OR" not in v and "from:" not in v:
            raise ValueError("query must include OR clauses or from: filters")
        return v

# ───────────────────────────────────────────────
# 2. Prompt
# ───────────────────────────────────────────────
BASE_FILTERS = "-is:retweet -is:reply lang:en is:verified (has:links OR has:media)"
SYSTEM_MESSAGE = f"""You are a Twitter advanced-search parser.

Always add these filters unless the user explicitly forbids them:
{BASE_FILTERS}

Do **not** use the since:/until: operators (they are invalid in v2).  
Instead, the caller will pass start_time/end_time separately.

Return JSON that matches the schema you were given.
"""

# ───────────────────────────────────────────────
# 3. Engagement thresholds
# ───────────────────────────────────────────────
MIN_LIKES_HIGH       = 50
MIN_RETWEETS_HIGH    = 10
MIN_FOLLOWERS_HIGH   = 5_000

MIN_LIKES_MEDIUM     = 5
MIN_RETWEETS_MEDIUM  = 1
MIN_FOLLOWERS_MEDIUM = 250

MIN_LIKES_LOW        = 0
MIN_RETWEETS_LOW     = 0
MIN_FOLLOWERS_LOW    = 0

MIN_KEEP             = 3
MAX_KEEP             = 100

# ───────────────────────────────────────────────
# 4. Helper functions
# ───────────────────────────────────────────────
def compute_engagement_score(pm: Dict) -> Tuple[int, str]:
    # Real-world stats pulled from Twitter used to calibrate each metric to a common scale:
    IMPRESSION_VALUE = 0.005
    LIKE_VALUE       = 1
    RETWEET_VALUE    = 7

    engagement_score = pm["retweet_count"] * RETWEET_VALUE + pm["like_count"] * LIKE_VALUE

    if "impression_count" in pm:
        engagement_score = (engagement_score + pm["impression_count"] * IMPRESSION_VALUE) / 3
    else:
        engagement_score = engagement_score / 2

    # Finally we can apply a simple log scaling to the score:
    engagement_score = int(math.log(engagement_score + 1, math.e))
    engagement_score = max(engagement_score, 0)

    return engagement_score, "tweet_engagement_score"


def orig_photo_url(url: str) -> str:
    """
    Convert https://pbs.twimg.com/media/XXXX?format=jpg&name=large
    →       https://pbs.twimg.com/media/XXXX?format=jpg&name=orig
    Works even if the API already returned '&name=small' or no name param.
    """
    if "pbs.twimg.com/media/" not in url:
        return url
    if "name=" in url:
        return re.sub(r"name=[a-z]+", "name=orig", url)
    # rare: no name param included
    if "?" in url:
        return f"{url}&name=orig"
    return f"{url}?name=orig"


# ───────────────────────────────────────────────
# 5. Twitter helper (v2 only)
# ───────────────────────────────────────────────
def twitter_search(x: X, query: str, start=None, end=None) -> Dict:
    params = {
        "query"        : query,
        "sort_order"   : "relevancy",
        "max_results"  : MAX_KEEP,
        "tweet.fields" : "created_at,public_metrics,attachments",
        "user.fields"  : "username,name,public_metrics,verified",
        "media.fields" : "type,url,preview_image_url,width,height,alt_text",
        "expansions"   : "author_id,attachments.media_keys",
    }
    if start: params["start_time"] = start
    if end  : params["end_time"]   = end
    return x._make_request(
        "get",
        "https://api.twitter.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {x.bearer_token}"},
        params=params,
    ).json()

# ───────────────────────────────────────────────
# 6. Payload processing
# ───────────────────────────────────────────────
def process_payload(
    raw: Dict, 
    min_likes: int = MIN_LIKES_HIGH,
    min_retweets: int = MIN_RETWEETS_HIGH,
    min_followers: int = MIN_FOLLOWERS_HIGH,
) -> List[Dict]:
    users = {u["id"]: u for u in raw.get("includes", {}).get("users", [])}
    media = {m["media_key"]: m for m in raw.get("includes", {}).get("media", [])}

    keep: List[Dict] = []
    for t in raw.get("data", []):
        pm   = t["public_metrics"]
        uid  = t["author_id"]
        user = users.get(uid, {})
        fcnt = user.get("public_metrics", {}).get("followers_count", 0)

        if (pm["like_count"] >= min_likes and
            pm["retweet_count"] >= min_retweets and
            fcnt >= min_followers):

            score, metric = compute_engagement_score(pm)
            media_objs = []
            for mk in t.get("attachments", {}).get("media_keys", []):
                m = media.get(mk)
                if not m:
                    continue
                if m["type"] == "photo" and "url" in m:
                    m = dict(m)                             # shallow copy
                    m["url_orig"] = orig_photo_url(m["url"])
                media_objs.append(m)
            
            tweet_url = f"https://x.com/{user.get('username')}/status/{t['id']}" if user.get('username') else None

            keep.append({
                "id"               : t["id"],
                "created_at"       : t["created_at"],
                "text"             : t["text"],       # full up to 280 chars
                "tweet_url"        : tweet_url,
                "public_metrics"   : pm,
                "score"            : score,
                "score_metric"     : metric,
                "author_username"  : user.get("username"),
                "author_name"      : user.get("name"),
                "author_followers" : fcnt,
                "media"            : media_objs,
            })

    keep.sort(key=itemgetter("score"), reverse=True)
    return keep[:MAX_KEEP+1]

# ───────────────────────────────────────────────
# 7. Main Eve handler
# ───────────────────────────────────────────────
async def handler(args: dict, user: str, agent: str):
    agent      = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise RuntimeError("No valid Twitter deployments found")

    user_query = args["query"]
    parsed: TwitterSearchQuery = await async_prompt(
        messages       = [UserMessage(role="user",
                                      content=f"Build a Twitter search for: {user_query}")],
        system_message = SYSTEM_MESSAGE,
        response_model = TwitterSearchQuery,
        model          = "gpt-4o",
    )

    print("--------------------------------")
    print(f"LLM query → {parsed.query}")
    print("--------------------------------")

    x = X(deployment)

    # strict pass
    raw_strict = twitter_search(x, parsed.query,
                                start=args.get("start_time"),
                                end=args.get("end_time"))
    tweets = process_payload(raw_strict, min_likes=MIN_LIKES_HIGH, min_retweets=MIN_RETWEETS_HIGH, min_followers=MIN_FOLLOWERS_HIGH)

    # relaxed pass
    if len(tweets) < MIN_KEEP:
        # Include non-verified accounts and tweets with links or media
        relaxed_q = parsed.query.replace("is:verified", " ") \
                                .replace("(has:links OR has:media)", " ")
        print("relaxed_q", relaxed_q)
        raw_relaxed = twitter_search(x, relaxed_q,
                                     start=args.get("start_time"),
                                     end=args.get("end_time"))
        tweets = process_payload(raw_relaxed, min_likes=MIN_LIKES_MEDIUM, min_retweets=MIN_RETWEETS_MEDIUM, min_followers=MIN_FOLLOWERS_MEDIUM)

    # very relaxed pass
    if len(tweets) < MIN_KEEP:
        # Include non-verified accounts, tweets with links or media and retweets
        relaxed_q = parsed.query.replace("is:verified", " ") \
                                .replace("-is:retweet", " ") \
                                .replace("(has:links OR has:media)", " ")
        print("relaxed_q 2", relaxed_q)
        raw_very_relaxed = twitter_search(x, relaxed_q,
                                     start=args.get("start_time"),
                                     end=args.get("end_time"))
        tweets = process_payload(raw_very_relaxed, min_likes=MIN_LIKES_LOW, min_retweets=MIN_RETWEETS_LOW, min_followers=MIN_FOLLOWERS_LOW)

    # pretty-print without breaking on newlines
    def flat(txt: str) -> str:
        return txt.replace("\n", "\\n")

    print(f"Returned {len(tweets)} tweets.\n")
    for t in tweets[:10]:
        print(f"- @{t['author_username']} :: {flat(t['text'])[:280]}…  ({t['public_metrics']['retweet_count']} RT, {t['public_metrics']['like_count']} likes, {t['score']} engagement score)")
        print("--------------------------------")

    print("--------------------------------")
    print(json.dumps(tweets, indent=4))
    print("--------------------------------")

    return {"output": tweets}
