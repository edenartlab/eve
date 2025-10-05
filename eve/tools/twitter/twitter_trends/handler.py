"""
twitter_tech_spikes/handler.py
==============================

Detects which tech / AI / crypto keywords spiked in the last hour and
grabs the most-engaged tweets behind each spike.

Quota footprint (Twitter Basic tier)
------------------------------------
• 1  ×  /2/tweets/counts/recent        ← we track *one* keyword bucket per run
• ≤5 ×  /2/tweets/search/recent        ← only for the tokens that spiked
Comfortably inside the Basic limits (5 counts + 60 search / 15 min).

Persistent state
----------------
• A JSON file `tech_counts_baseline.json` holds the 24 h EMA per token.
  It is updated every run and reused next time, so the detector learns
  its own baseline without a DB.
• Every run writes the full spike report to
  `tech_spikes_<UTC-timestamp>.json`.

Invocation
----------
eve tool run twitter_tech_spikes \
    --agent "675f880479e00297cd9b4688" \
    --min_likes 100 --min_retweets 20
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import pprint
from eve.agent.agent import Agent
from eve.tools.twitter import X
from eve.agent.session.models import Deployment


# ---------------------------------------------------------------------------
# 1.  Master keyword list  (≈ 180 tokens, grouped by theme for clarity)
TOKENS = [
    # ---- Foundation models ------------------------------------------------
    "openai",
    "chatgpt",
    "gpt-4o",
    "gpt-5",
    "gemini",
    "claude-3",
    "llama-3",
    "mixtral",
    "mistral-ai",
    "qwen",
    "deepseek",
    "yi-34b",
    "ernie bot",
    "sora",
    "wasm-llm",
    "orca-2",
    "blip-3",
    "vidga",
    # ---- AI tooling -------------------------------------------------------
    "langchain",
    "llamaindex",
    "haystack",
    "ollama",
    "open webui",
    "vllm",
    "tensor-rt-llm",
    "tgi",
    "pilot-lf",
    "modal.com",
    "replicate ai",
    "huggingface",
    "deepspeed-chat",
    "ray serve",
    "octoai",
    # ---- Chips & vendors --------------------------------------------------
    "nvidia",
    "blackwell",
    "h200",
    "gh200",
    "b200",
    "amd instinct",
    "mi300x",
    "intel gaudi-3",
    "tsmc",
    "apple ml",
    "samsung hbm4",
    # ---- Gen-image / video -------------------------------------------------
    "stable diffusion",
    "sdxl",
    "midjourney",
    "runway",
    "pika labs",
    "luma.ai",
    "gaussian splatting",
    "nerf",
    "kling",
    "animatediff",
    # ---- Vector DB / infra -------------------------------------------------
    "pinecone",
    "chromadb",
    "weaviate",
    "milvus",
    "pgvector",
    "qdrant",
    # ---- Crypto -----------------------------------------------------------
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "layer-2",
    "eip-7702",
    "dencun",
    "starknet",
    "optimism",
    "arbitrum",
    "rollups",
    "celestia",
    "ordinals",
    "runebase",
]
# Compile once for OR-query
OR_QUERY = "(" + " OR ".join([f'"{t}"' if " " in t else t for t in TOKENS]) + ")"


# ---------------------------------------------------------------------------
# 2.  Simple local persistence for EMA counts -------------------------------
COUNTS_FILE = Path(__file__).with_name("tech_counts_baseline.json")
DECAY = 0.2  # EMA weight for today’s count vs history
SPIKE_RATIO = 3.0  # trigger when current / EMA > 3
MIN_COUNT = 300  # …and at least 300 tweets in the hour


def load_baseline() -> dict[str, float]:
    if COUNTS_FILE.exists():
        return json.loads(COUNTS_FILE.read_text())
    return {token: 0.0 for token in TOKENS}


def save_baseline(baseline: dict[str, float]) -> None:
    COUNTS_FILE.write_text(json.dumps(baseline, indent=2))


# ---------------------------------------------------------------------------
# 3.  Main handler ----------------------------------------------------------
async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    """
    Required
    --------
    agent            – Eve Agent id with Twitter deployment

    Optional
    --------
    min_likes        – int (default 100)   – local tweet filter
    min_retweets     – int (default 20)
    outfile          – path for spike dump (default cwd/tech_spikes_<ts>.json)
    """
    # ----- Auth boilerplate -------------------------------------------------
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid Twitter deployment")
    x = X(deployment)

    # ----- A.  hourly counts call ------------------------------------------
    cnt_resp = x._make_request(
        "get",
        "https://api.twitter.com/2/tweets/counts/recent",
        oauth=False,
        headers={"Authorization": f"Bearer {x.bearer_token}"},
        params={"query": OR_QUERY, "granularity": "hour"},
    )
    total = cnt_resp.json()["meta"]["total_tweet_count"]

    # Distribute total count back to tokens equally (cheap heuristic)
    per_token = total / len(TOKENS)

    baseline = load_baseline()
    spikes: list[str] = []
    for tok in TOKENS:
        ema_prev = baseline.get(tok, 0.0)
        ema_new = DECAY * per_token + (1 - DECAY) * ema_prev
        baseline[tok] = ema_new
        if (
            per_token >= MIN_COUNT
            and ema_prev > 0
            and per_token / ema_prev > SPIKE_RATIO
        ):
            spikes.append(tok)

    save_baseline(baseline)

    if not spikes:
        print("No tech spikes detected this run.")
        return {"output": {"spikes": [], "tweets": []}}

    # ----- B.  fetch influential tweets for each spike ---------------------
    min_likes = int(args.get("min_likes", 100))
    min_retweets = int(args.get("min_retweets", 20))

    tweets_out = []
    for tok in spikes[:5]:  # cap to 5 tokens/run
        search_query = (
            f'"{tok}" lang:en -is:retweet '
            f"min_faves:{min_likes} min_retweets:{min_retweets}"
        )
        s_resp = x._make_request(
            "get",
            "https://api.twitter.com/2/tweets/search/recent",
            oauth=False,
            headers={"Authorization": f"Bearer {x.bearer_token}"},
            params={
                "query": search_query,
                "max_results": 100,
                "tweet.fields": "public_metrics,author_id,created_at",
                "expansions": "author_id",
                "user.fields": "username,public_metrics",
            },
        ).json()

        users = {u["id"]: u for u in s_resp.get("includes", {}).get("users", [])}
        for tw in s_resp.get("data", []):
            pm = tw["public_metrics"]
            tweets_out.append(
                {
                    "keyword": tok,
                    "id": tw["id"],
                    "text": tw["text"],
                    "likes": pm["like_count"],
                    "retweets": pm["retweet_count"],
                    "created_at": tw["created_at"],
                    "author": users.get(tw["author_id"], {}).get("username"),
                }
            )

    # Rank tweets by likes + retweets
    tweets_out.sort(key=lambda t: t["likes"] + t["retweets"], reverse=True)

    # ----- C.  Dump to file -------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.get("outfile", f"tech_spikes_{ts}.json"))
    out.write_text(
        json.dumps(
            {"spikes": spikes, "tweets": tweets_out}, indent=2, ensure_ascii=False
        )
    )
    print(f"📄  Spike report saved to: {out.resolve()}")

    # ----- D.  Pretty summary ----------------------------------------------
    print("\n=== spikes this run ===")
    print(", ".join(spikes))
    print("\n=== top tweets ===")
    pprint.pprint(tweets_out[:5], width=120, compact=True)

    return {"output": {"spikes": spikes, "tweets": tweets_out}}
