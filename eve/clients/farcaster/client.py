from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hmac
import hashlib
import logging
import os
from farcaster import Warpcast
from dotenv import load_dotenv
from fastapi.background import BackgroundTasks
from ably import AblyRealtime
import aiohttp
from contextlib import asynccontextmanager
import traceback

from eve.agent import Agent
from eve.llm import UpdateType
from eve.thread import Thread
from eve.user import User
from eve.eden_utils import prepare_result
from eve.clients import common
from eve.models import ClientType

logger = logging.getLogger(__name__)


class MentionedProfile(BaseModel):
    fid: int
    username: str
    custody_address: str
    display_name: str
    pfp_url: str


class CastWebhook(BaseModel):
    created_at: int
    type: str
    data: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
    app.state.channel_name = common.get_ably_channel_name(
        app.state.agent.name, ClientType.FARCASTER
    )
    channel = ably_client.channels.get(app.state.channel_name)

    async def async_callback(message):
        logger.info(f"Received update in Farcaster client: {message.data}")

        data = message.data
        if not isinstance(data, dict) or "type" not in data:
            logger.error("Invalid message format:", data)
            return

        update_type = data["type"]
        update_config = data.get("update_config", {})
        cast_hash = update_config.get("cast_hash")
        author_fid = update_config.get("author_fid")

        if not cast_hash or not author_fid:
            logger.error("Missing cast_hash or author_fid in update_config:", data)
            return

        logger.info(f"Processing update type: {update_type} for cast: {cast_hash}")

        try:
            if update_type == UpdateType.START_PROMPT:
                pass

            elif update_type == UpdateType.ERROR:
                error_msg = data.get("error", "Unknown error occurred")
                app.state.client.post_cast(
                    text=f"Error: {error_msg}",
                    parent={"hash": cast_hash, "fid": author_fid},
                )

            elif update_type == UpdateType.ASSISTANT_MESSAGE:
                content = data.get("content")
                if content:
                    app.state.client.post_cast(
                        text=content,
                        parent={"hash": cast_hash, "fid": author_fid},
                    )

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = data.get("result", {})
                result["result"] = prepare_result(result["result"], db=app.state.db)
                url = result["result"][0]["output"][0]["url"]
                app.state.client.post_cast(
                    text="",
                    embeds=[url],
                    parent={"hash": cast_hash, "fid": author_fid},
                )

        except Exception as e:
            logger.error(
                f"Error processing Ably update: {str(e)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            try:
                app.state.client.post_cast(
                    text=f"Sorry, I encountered an error: {str(e)}",
                    parent={"hash": cast_hash, "fid": author_fid},
                )
            except Exception as post_error:
                logger.error(
                    f"Failed to send error message to Farcaster: {str(post_error)}\n"
                    f"Stack trace:\n{traceback.format_exc()}"
                )

    # Subscribe using the async callback
    await channel.subscribe(async_callback)
    logger.info(f"Subscribed to Ably channel: {app.state.channel_name}")

    yield  # Server is running

    # Cleanup
    if ably_client:
        ably_client.close()
        logger.info("Closed Ably connection")


def create_app(env: str, db: str = "STAGE"):
    app = FastAPI(lifespan=lifespan)

    load_dotenv(env)

    mnemonic = os.environ.get("CLIENT_FARCASTER_MNEMONIC")
    db = os.environ.get("DB", "STAGE")
    agent_name = os.getenv("EDEN_AGENT_USERNAME")

    # Store these in app.state for access in lifespan and routes
    app.state.client = Warpcast(mnemonic=mnemonic)
    app.state.agent = Agent.load(agent_name, db=db)
    app.state.db = db

    logger.info("Initialized Farcaster client")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_neynar_signature(signature: str, raw_body: bytes) -> bool:
        webhook_secret = os.environ.get("CLIENT_FARCASTER_NEYNAR_WEBHOOK_SECRET")
        computed_signature = hmac.new(
            webhook_secret.encode(), raw_body, hashlib.sha512
        ).hexdigest()
        return hmac.compare_digest(computed_signature, signature)

    @app.post("/")
    async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
        body = await request.body()

        signature = request.headers.get("X-Neynar-Signature")
        if not signature:
            raise HTTPException(status_code=400, detail="Missing signature header")

        if not verify_neynar_signature(signature, body):
            raise HTTPException(status_code=400, detail="Invalid signature")

        webhook_data = CastWebhook.model_validate(await request.json())

        cast_data = webhook_data.data
        if not cast_data or "hash" not in cast_data:
            raise HTTPException(status_code=400, detail="Invalid cast data")

        # Add the background task with channel_name
        background_tasks.add_task(
            process_webhook,
            cast_data,
            app.state.client,
            app.state.agent,
            app.state.db,
            app.state.channel_name,
        )

        return {"status": "accepted"}

    return app


async def process_webhook(
    cast_data: dict, client: Warpcast, agent: Agent, db: str, channel_name: str
):
    """Process the webhook data in the background"""
    logger.info(f"Processing webhook for cast {cast_data['hash']}")
    try:
        cast_hash = cast_data["hash"]
        author = cast_data["author"]
        author_username = author["username"]
        author_fid = author["fid"]

        # Get or create user
        user = User.from_farcaster(author_fid, author_username, db=db)

        # Get or create thread
        thread_key = f"farcaster-{author_fid}-{cast_hash}"
        thread = agent.request_thread(
            key=thread_key,
            db=db,
        )

        # Make API request
        api_url = os.getenv("EDEN_API_URL")
        request_data = {
            "user_id": str(user.id),
            "agent_id": str(agent.id),
            "thread_id": str(thread.id),
            "user_message": {
                "content": cast_data["text"],
                "name": author_username,
            },
            "update_config": {
                "sub_channel_name": channel_name,
                "cast_hash": cast_hash,
                "author_fid": author_fid,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/chat",
                json=request_data,
            ) as response:
                if response.status != 200:
                    raise Exception("Failed to process request")

    except Exception as e:
        logger.error(
            f"Error processing webhook in background: {str(e)}\n"
            f"Stack trace:\n{traceback.format_exc()}"
        )
        try:
            client.post_cast(
                text=f"Sorry, I encountered an error: {str(e)}",
                parent={"hash": cast_hash, "fid": author_fid},
            )
        except Exception as post_error:
            logger.error(
                f"Failed to send error message to Farcaster: {str(post_error)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )


def start(env: str, db: str = "STAGE"):
    """Start the FastAPI server locally"""
    import uvicorn

    app = create_app(env, db)
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    start()
