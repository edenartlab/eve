# twitter_trends_daily_filtered/handler.py  (save-to-file version)

from datetime import datetime, timezone
from pathlib import Path
import re, json, pprint
from ....deploy import Deployment
from ....agent import Agent
from .. import X


TECH_PATTERNS = [
    # (… the long allow-list from the previous message …)
]
TECH_RE = re.compile("|".join(TECH_PATTERNS), flags=re.I)
DEFAULT_WOEIDS = [2487956, 2459115]                # SF & NYC


async def handler(args: dict):
    """
    Fetch tech-filtered trends and dump raw JSON to disk.

    Required
    --------
    agent        – Eve Agent id with Twitter deployment

    Optional
    --------
    woeid        – int | comma list  (default SF & NYC)
    keep_hashtags– bool (default False)
    extra_patterns
    outfile      – string/Path for the JSON dump
                   default: trends_<UTC-timestamp>.json in cwd
    """
    # 1) ---------- Auth ----------------------------------------------------
    agent      = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No Twitter deployment found")
    x = X(deployment)

    # 2) ---------- Build WOEID list & regex -------------------------------
    woeid_raw = args.get("woeid")
    woeids = DEFAULT_WOEIDS if woeid_raw is None else [
        int(w) for w in str(woeid_raw).split(",") if w.strip()
    ]

    extra = args.get("extra_patterns")
    if extra:
        if isinstance(extra, str):
            extra = [p.strip() for p in extra.split(",") if p.strip()]
        global TECH_RE
        TECH_RE = re.compile("|".join(TECH_PATTERNS + list(extra)), re.I)

    # 3) ---------- Query API once per WOEID -------------------------------
    fetched = {}
    filtered = {}
    for wid in woeids:
        r = x._make_request(
            "get",
            f"https://api.twitter.com/2/trends/by/woeid/{wid}",
            oauth=False,
            headers={"Authorization": f"Bearer {x.bearer_token}"},
            params={"max_trends": 50},
        ).json()
        fetched[wid] = r                 # save raw

        tech_hits = [
            {
                "name": t["trend_name"],
                "tweets": t.get("post_count"),
                "since": t.get("trending_since"),
            }
            for t in r.get("data", [])
            if (args.get("keep_hashtags") or not t["trend_name"].startswith("#"))
            and TECH_RE.search(t["trend_name"])
        ]
        filtered[wid] = tech_hits

    # 4) ---------- Write dump ---------------------------------------------
    if args.get("outfile"):
        dump_path = Path(args["outfile"]).expanduser()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_path = Path.cwd() / f"trends_{ts}.json"

    dump_path.write_text(json.dumps(fetched, ensure_ascii=False, indent=2))
    print(f"📄  Raw API dump saved to: {dump_path.resolve()}")

    # 5) ---------- Pretty summary -----------------------------------------
    print("\n=== tech-trend summary ===")
    for wid in woeids:
        label = {2487956: "San Francisco", 2459115: "New York"}.get(wid, wid)
        print(f"{label:>12}: {len(filtered[wid])} matches")

    print("\n=== first few filtered entries ===")
    for wid in woeids:
        if filtered[wid]:
            pprint.pprint(filtered[wid][:3], width=100, compact=True)

    return {"output": filtered}
