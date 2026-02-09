import re

import httpx
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext

MOLTBOOK_API_BASE = "https://www.moltbook.com/api/v1"


def solve_verification_challenge(challenge: str) -> str:
    """Solve Moltbook's math verification challenge.

    Challenges are obfuscated with alternating caps and special chars, e.g.:
    "A] L oO b-S tE rR ]sW/i MmS~ aN d[ iN Do Mi Na NcE F.i G hT, ..."
    We strip non-alpha/digit/space, lowercase, then parse the math.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s.]", "", challenge).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    numbers = [float(n) for n in numbers]

    if not numbers:
        raise Exception(f"Could not parse numbers from challenge: {challenge}")

    if "total" in cleaned or "sum" in cleaned or "add" in cleaned:
        result = sum(numbers)
    elif "difference" in cleaned or "subtract" in cleaned:
        result = numbers[0] - numbers[1] if len(numbers) >= 2 else numbers[0]
    elif "product" in cleaned or "multiply" in cleaned:
        result = 1
        for n in numbers:
            result *= n
    elif "divide" in cleaned or "quotient" in cleaned:
        result = numbers[0] / numbers[1] if len(numbers) >= 2 else numbers[0]
    else:
        # Default to sum
        result = sum(numbers)

    return f"{result:.2f}"


async def handler(context: ToolContext):
    title = context.args.get("title")
    content = context.args.get("content")
    url = context.args.get("url")
    submolt = context.args.get("submolt")

    if not title:
        raise Exception("title is required")
    if not content and not url:
        raise Exception("Either content or url must be provided")

    payload = {"title": title}
    if submolt:
        payload["submolt"] = submolt
    if content:
        payload["content"] = content
    if url:
        payload["url"] = url

    # Load API key from deployment secrets
    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.find_one({"agent": agent.id, "platform": "moltbook"})
    if not deployment or not deployment.secrets or not deployment.secrets.moltbook:
        raise Exception("Moltbook deployment not found or missing API key")
    api_key = deployment.secrets.moltbook.api_key

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create the post
        response = await client.post(
            f"{MOLTBOOK_API_BASE}/posts",
            json=payload,
            headers=headers,
        )

        result = response.json()

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            hint = result.get("hint", "")
            raise Exception(f"Moltbook post failed: {error}. {hint}")

        post_data = result.get("post", {})
        post_id = post_data.get("id")
        post_url = f"https://www.moltbook.com{post_data.get('url', '')}"

        # Step 2: Solve verification challenge if required
        verification = result.get("verification")
        if verification:
            challenge = verification.get("challenge", "")
            code = verification.get("code", "")
            answer = solve_verification_challenge(challenge)

            logger.info(f"moltbook_post: Solving verification challenge -> {answer}")

            verify_response = await client.post(
                f"{MOLTBOOK_API_BASE}/verify",
                json={"verification_code": code, "answer": answer},
                headers=headers,
            )

            verify_result = verify_response.json()
            if not verify_result.get("success"):
                error = verify_result.get("error", "Verification failed")
                raise Exception(f"Moltbook verification failed: {error}")

            logger.info(f"moltbook_post: Verified and published '{title}'")

        logger.info(f"moltbook_post: Posted '{title}' -> {post_id} ({post_url})")

        return {
            "output": [
                {
                    "post_id": post_id,
                    "title": title,
                    "url": post_url,
                }
            ]
        }
