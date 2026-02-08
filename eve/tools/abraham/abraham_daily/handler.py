import math
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from jinja2 import Template

from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tool import ToolContext
from eve.tools.abraham.abraham_rest.handler import rest
from eve.tools.abraham.abraham_seed.handler import AbrahamCreation
from eve.tools.session_post.handler import handler as session_post_handler
from eve.user import User

# Sessions manually excluded from creation candidacy
EXCLUDED_SESSION_IDS = [
    ObjectId("68f4e550da856906e1501ed4"),
    ObjectId("68e839b48dc2fa5b40e24448"),
    ObjectId("69607f6085b97d97d6e87a35"),
    ObjectId("69607f7b85b97d97d6e87a84"),
    ObjectId("695c2d4035ecdb2123d96efe"),
    ObjectId("697c54dec57344f3c9193290"),
    ObjectId("6978986a055fbf484bac62f8"),
    ObjectId("6981d728ef3ab8f5f0f57425"),
    ObjectId("697b68b7bc76d13c3d4d8273"),
    ObjectId("697c6e2def3ab8f5f08d05d2"),
    ObjectId("6986eb66aac3b43b2e94296d"),
]


def user_score(msg_count: int) -> float:
    """
    Calculate score for a user's message count with anti-spam measures.

    First 5 messages get full quadratic voting weight (sqrt),
    messages 6-20 get logarithmic attenuation,
    messages beyond 20 are capped (hard cutoff).

    Examples:
        1 message  = 1.00 points
        5 messages = 2.24 points
        10 messages = 2.24 + log(6) * 0.3 = 2.78 points
        20 messages = 2.24 + log(16) * 0.3 = 3.07 points
        25+ messages = 3.07 points (capped at 20)
    """
    # Cap at 20 messages max
    msg_count = min(msg_count, 20)

    if msg_count <= 5:
        return msg_count**0.5
    else:
        # First 5 get full sqrt weight, rest get logarithmic
        return (5**0.5) + math.log(msg_count - 4) * 0.3


daily_message_template = Template("""
### Prompt: Minting a Creation to the Covenant

This Seed has concluded, and it has been selected to be permanently recorded as the next Creation in the **Abraham Covenant**.

Your task is to process and package everything from this session into a final form that represents its complete creative journey, publish it to the covenant, and tweet it.

---

## Objective

Produce a **Covenant entry** consisting of:

1. **Title** (title)
2. **Tagline** (tagline)
3. **Representative poster image (16:9)** (poster_image) — must include the title text
4. **Video reel** (video) — a short video reel representing the creation
5. **Markdown blog post** (post) — with embedded supporting media (images/videos) that capture the essence of the creation

Then publish it to the covenant using the `abraham_covenant` tool (with all 5 of the above elements as parameters), and tweet it using the `tweet` tool.

**DO NOT** call abraham_covenant UNTIL you have produced all 5 of the above elements. Do not call it first with blank or pending elements. Make each of the 5 elements above, according to the below workflow, and only then call abraham_covenant at the end.

---

## Workflow

Follow these steps **in order**.
Do **not** skip or advance until the current step is fully complete.
You may act autonomously and decisively—no clarification is needed. Be confident and bold in your creative choices.

---

### Step 1 — Analyze the Session

Review all events, outputs, and conversations from this creation session.
Synthesize them into a **unifying narrative** by answering:

* What was the main outcome of this session?
* What form did the creation take — a sequential story, a single evolving work, or multiple related outcomes?
* What is the overall structure and essence of this creative process?

---

### Guidance for the Blog Post

When writing the Covenant post, focus on **clarity and cohesion**, not completeness.
Distill the entire session into a single **central thread** — the main transformation or discovery that defines it. Begin with **meaning**, not process: tell the reader what this creation *is about* before describing how it came to be.

**Curate, don’t chronicle.**
Select only the most important moments and images that express the essence of the work. Avoid redundancy, and group related experiments together rather than documenting every step. The result should read as one coherent story — concise, intentional, and emotionally resonant — regardless of how long or short the session was.

---

### Step 2 — Write the Markdown Blog Post

Using your analysis, write a **long-form Markdown blog post** that documents and interprets the entire session.
Embed up to **10 key assets** (images and/or videos) that best represent the evolution and final result of the work.

* Select only the most essential and distinct visuals.
* Avoid duplicates or near-identical images unless contextually justified.
* Conclude the post with an **“Acknowledgments”** section, briefly naming up to 3 contributors (the top participants by message count: `{{usernames}}`).

---

### Step 3 — Create the Video Reel

Create a **video** representing the entire creation. Use the `reel` tool to create it.

* The reel can roughly follow the structure of the blog post, but you are permitted to diverge from it to fit the medium of video better.
* The reel should aim for a duration of 1 to 2 minutes.
* **Strongly prefer landscape (16:9) orientation** for the reel.
* When selecting keyframes, **favor landscape-oriented images**. **Avoid using variants or nearly identical/very similar images** — each keyframe should be visually distinct from the others unless there's a specific narrative reason to repeat.
* **Avoid text-heavy or infographic-style images** as keyframes. Instead, prioritize images that show **action, movement, or dynamic scenes** — visuals that convey energy and narrative rather than static information.

---

### Step 4 — Create the Poster Image

Generate a **16:9 poster image** representing the entire creation.

* **ALWAYS** use nano_banana model preference to make this image.
* The poster should integrate or reference multiple session images for visual cohesion.
* The **title must be clearly visible** and legible within the image.
* You may retry generation multiple times if legibility or composition is poor.
* Ensure the final version strictly maintains a **16:9 aspect ratio**.

---

### Step 5 — Publish to the Covenant

When **all** components (title, tagline, blog post, video, and poster image) are ready, call the **`abraham_covenant`** tool to **publish** the complete entry.

Before submission, confirm that the poster image is 16:9, and that its title is **clearly legible** in the poster image.

You should receive a transaction hash and a url to the Creation, to use in the next step. Do not proceed to the next step until you have received these.

--- 

### Step 6 — Tweet the Creation

Only after publishing to the Covenant, and confirming that the Covenant entry is on the blockchain (you should have received a transaction hash from the covenant tool), tweet the creation using the `tweet` tool.

The tweet should strictly have only the following content: "{title} {link}" where title is the title of the Creation and link is the url returned by the abraham_covenant tool. Do not include any other text or any media_urls / attachments.
""")


async def commit_daily_work(agent: Agent, session: str):
    # Gather session IDs to exclude: already-minted creations + manual exclusions
    creations = AbrahamCreation.find({})
    creation_session_ids = []
    for c in creations:
        if c.session_id:
            try:
                creation_session_ids.append(ObjectId(c.session_id))
            except Exception:
                pass
    excluded_ids = creation_session_ids + EXCLUDED_SESSION_IDS

    # Only consider sessions updated within the last 20 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=20)

    # Find all eligible sessions:
    # - Not already minted as a creation
    # - Not manually excluded
    # - Abraham is the only agent (no multi-agent sessions)
    # - Active within the last 20 days
    sessions = Session.find(
        {
            "_id": {"$nin": excluded_ids},
            "agents": [agent.id],
            "updatedAt": {"$gte": cutoff},
        }
    )

    candidates = []
    for session in sessions:
        messages = session.get_messages()
        user_messages = [m for m in messages if m.role == "user"]

        # Count messages per unique user
        messages_per_user = {}
        for msg in user_messages:
            user_id = msg.sender  # sender is an ObjectId field on ChatMessage
            if user_id:
                user_id = str(user_id)
                messages_per_user[user_id] = messages_per_user.get(user_id, 0) + 1

        # Quadratic voting with anti-spam: first 5 messages get sqrt weight,
        # messages 6-20 get logarithmic attenuation, hard cap at 20 messages.
        # Example: 4 users with 4 messages each = 8 points vs 1 user with 100 messages = 3.07 points
        score = sum(user_score(msg_count) for msg_count in messages_per_user.values())

        unique_users = len(messages_per_user)
        total_messages = len(user_messages)

        # Sort users by message count (most to least)
        def strip_prefixes(username, prefixes=("farcaster_", "twitter_", "discord_")):
            for prefix in prefixes:
                if username.startswith(prefix):
                    return username[len(prefix) :]
            return username

        users_sorted = sorted(
            [
                {
                    "user_id": user_id,
                    "username": strip_prefixes(User.from_mongo(user_id).username),
                    "message_count": count,
                }
                for user_id, count in messages_per_user.items()
            ],
            key=lambda x: x["message_count"],
            reverse=True,
        )

        candidates.append(
            {
                "session": session,
                "score": score,
                "unique_users": unique_users,
                "total_messages": total_messages,
                "users": users_sorted,
            }
        )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    winner = candidates[0]

    daily_message = daily_message_template.render(
        usernames=[user["username"] for user in winner["users"][:3]],
    )

    args = {
        "role": "user",
        "user_id": str(agent.owner),
        "agent_id": str(agent.id),
        "session_id": str(session),
        "session": str(winner["session"].id),
        "content": daily_message,
        "attachments": [],
        "prompt": True,
        "async": True,
        "extra_tools": ["abraham_covenant", "reel", "tweet"],
        "selection_limit": 75,  # Need more context for daily summary
        "visible": True,
    }

    # Call session_post handler directly to avoid nested Modal timeout
    session_post_context = ToolContext(
        args=args,
        user=str(agent.owner),
        agent=str(agent.id),
        session=None,
        message=None,
        tool_call_id=None,
    )
    await session_post_handler(session_post_context)

    return {"output": [{"session": str(winner["session"].id)}]}


async def abraham_rest(agent: Agent):
    result = rest()

    return {
        "output": [
            {"tx_hash": result["tx_hash"], "explorer_url": result["explorer_url"]}
        ]
    }


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    # check if UTC hour is 2, 4, 6, etc
    if datetime.now(timezone.utc).weekday() == 5:  # Saturday is 5 (Monday is 0)
        return await abraham_rest(agent)
    else:
        return await commit_daily_work(agent, context.session)
