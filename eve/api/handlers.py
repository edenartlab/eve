import json
import logging
import os
import time
import uuid
from bson import ObjectId
from typing import Dict, List, Optional
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import aiohttp

from eve.agent.session.models import (
    PromptSessionContext,
    Session,
    ChatMessage,
    EdenMessageType,
    EdenMessageData,
    EdenMessageAgentData,
    Trigger,
)
from eve.agent.session.session import run_prompt_session, run_prompt_session_stream
from eve.agent.session.triggers import create_trigger_fn, stop_trigger
from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CancelRequest,
    CancelSessionRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    PromptSessionRequest,
    TaskRequest,
    PlatformUpdateRequest,
    UpdateConfig,
    UpdateDeploymentRequest,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
)
from eve.api.helpers import (
    emit_update,
    setup_chat,
    create_telegram_chat_request,
    update_busy_state,
)
from eve.deploy import (
    stop_client,
    DeploymentConfig,
    DeploymentSettingsDiscord,
    ClientType,
    DeploymentSecrets,
    DeploymentSettingsFarcaster,
    deploy_client_modal,
)
from eve.eden_utils import prepare_result, dumps_json
from eve.tools.replicate_tool import replicate_update_task
from eve.agent.llm import UpdateType
from eve.agent.run_thread import async_prompt_thread
from eve.mongo import get_collection
from eve.task import Task
from eve.tool import Tool
from eve.agent import Agent
from eve.user import User
from eve.agent.thread import Thread, UserMessage
from eve.deploy import Deployment
from eve.tools.twitter import X
from eve.api.helpers import get_eden_creation_url

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()


@handle_errors
async def handle_create(request: TaskRequest):
    tool = Tool.load(key=request.tool)

    print("#### create ####")
    print(request)

    # if USE_RATE_LIMITS:
    #     await RateLimiter().check_create_rate_limit(user, tool)

    print("### run the tool ###")
    result = await tool.async_start_task(
        user_id=request.user_id, agent_id=None, args=request.args, public=request.public
    )

    print("### return the result ###")
    print(result)

    return dumps_json(result.model_dump(by_alias=True))


@handle_errors
async def handle_cancel(request: CancelRequest):
    task = Task.from_mongo(request.taskId)
    if str(task.user) != request.user:
        raise APIError(
            "Unauthorized: Task user does not match user_id", status_code=403
        )

    if task.status in ["completed", "failed", "cancelled"]:
        return {"status": task.status}

    tool = Tool.load(key=task.tool)
    await tool.async_cancel(task)
    return {"status": task.status}


async def handle_replicate_webhook(body: dict):
    task = Task.from_handler_id(body["id"])
    tool = Tool.load(task.tool)
    _ = replicate_update_task(
        task, body["status"], body["error"], body["output"], tool.output_handler
    )
    return {"status": "success"}


async def run_chat_request(
    user: User,
    agent: Agent,
    thread: Thread,
    tools: List[Tool],
    user_message: UserMessage,
    update_config: UpdateConfig,
    force_reply: bool,
    use_thinking: bool,
    model: str,
    user_is_bot: bool = False,
    metadata: Optional[Dict] = None,
):
    request_id = str(uuid.uuid4())

    is_client_platform = True if update_config else False
    request_processed = (
        False  # Flag to ensure stop signal isn't sent prematurely on error
    )

    try:
        async for update in async_prompt_thread(
            user=user,
            agent=agent,
            thread=thread,
            user_messages=user_message,
            tools=tools,
            force_reply=force_reply,
            use_thinking=use_thinking,
            model=model,
            user_is_bot=user_is_bot,
            stream=False,
            is_client_platform=is_client_platform,
        ):
            print("UPDATE", update)
            data = {
                "type": update.type.value,
                "update_config": update_config.model_dump() if update_config else {},
            }

            if update.type == UpdateType.START_PROMPT:
                await update_busy_state(update_config, request_id, True)
            elif update.type == UpdateType.ASSISTANT_MESSAGE:
                data["content"] = update.message.content
            elif update.type == UpdateType.TOOL_COMPLETE:
                data["tool"] = update.tool_name
                data["result"] = dumps_json(update.result)
            elif update.type == UpdateType.ERROR:
                data["error"] = update.error if hasattr(update, "error") else None
            elif update.type == UpdateType.END_PROMPT:
                await update_busy_state(update_config, request_id, False)
                request_processed = True  # Mark as processed

            await emit_update(update_config, data)

        # If the loop finishes without error, mark as processed
        request_processed = True
        await update_busy_state(update_config, request_id, False)

    except Exception as e:
        logger.error("Error in run_prompt", exc_info=True)
        # Update busy state immediately on error
        await update_busy_state(update_config, request_id, False)
        request_processed = True  # Mark as processed even on error to prevent finally block double-sending
        await emit_update(
            update_config,
            {"type": UpdateType.ERROR.value, "error": str(e)},
        )
    finally:
        # Ensure busy state is set to False if it hasn't been already
        # by END_PROMPT or the except block.
        if not request_processed:
            logger.warning(
                f"run_chat_request for {request_id} finished without END_PROMPT or error, ensuring busy state is cleared."
            )
            await update_busy_state(update_config, request_id, False)


async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    user, agent, thread, tools = await setup_chat(
        request, cache=True, background_tasks=background_tasks
    )

    background_tasks.add_task(
        run_chat_request,
        user,
        agent,
        thread,
        tools,
        request.user_message,
        request.update_config,
        request.force_reply,
        request.use_thinking,
        request.model,
        request.user_is_bot,
    )

    return {"thread_id": str(thread.id)}


@handle_errors
async def handle_stream_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    user, agent, thread, tools = await setup_chat(
        request, cache=True, background_tasks=background_tasks
    )

    async def event_generator():
        try:
            async for update in async_prompt_thread(
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                use_thinking=request.use_thinking,
                model=request.model,
                user_is_bot=request.user_is_bot,
                stream=True,
            ):
                data = {"type": update.type}
                if update.type == UpdateType.ASSISTANT_TOKEN:
                    data["text"] = update.text
                elif update.type == UpdateType.ASSISTANT_MESSAGE:
                    data["content"] = update.message.content
                    if update.message.tool_calls:
                        data["tool_calls"] = [
                            dumps_json(t.model_dump())
                            for t in update.message.tool_calls
                        ]
                elif update.type == UpdateType.TOOL_COMPLETE:
                    data["tool"] = update.tool_name
                    data["result"] = dumps_json(update.result)
                elif update.type == UpdateType.ERROR:
                    data["error"] = update.error or "Unknown error occurred"

                yield f"data: {json.dumps({'event': 'update', 'data': data})}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
        except Exception as e:
            logger.error("Error in event_generator", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@handle_errors
async def handle_deployment_create(request: CreateDeploymentRequest):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {agent.id}", status_code=404)

    # Predeploy: platform-specific validation and setup
    secrets, config = await predeploy_platform(
        agent, request.secrets, request.config, request.platform
    )

    # Create the deployment object
    deployment = Deployment(
        agent=agent.id,
        user=ObjectId(request.user),
        platform=request.platform,
        secrets=secrets,
        config=config,
        valid=True,
    )

    deployment.save(
        upsert_filter={"agent": agent.id, "platform": request.platform.value}
    )

    try:
        # Postdeploy: platform-specific actions that need the deployment object
        await postdeploy_platform(deployment, request.platform, secrets)
    except Exception as e:
        logger.error(f"Failed in postdeploy: {str(e)}")
        deployment.delete()
        raise APIError(f"Failed to deploy client: {str(e)}", status_code=500)

    return {"deployment_id": str(deployment.id)}


async def predeploy_platform(
    agent: Agent,
    secrets: DeploymentSecrets,
    config: DeploymentConfig,
    platform: ClientType,
):
    """Platform-specific validation, secret modification, and agent tool setup"""
    if platform == ClientType.DISCORD:
        return await predeploy_discord(agent, secrets, config)
    elif platform == ClientType.TELEGRAM:
        return await predeploy_telegram(agent, secrets, config)
    elif platform == ClientType.FARCASTER:
        return await predeploy_farcaster(agent, secrets, config)
    elif platform == ClientType.TWITTER:
        return await predeploy_twitter(agent, secrets, config)
    else:
        raise APIError(f"Invalid platform: {platform}", status_code=400)


async def postdeploy_platform(
    deployment: Deployment,
    platform: ClientType,
    secrets: DeploymentSecrets,
):
    """Platform-specific actions that require the deployment object"""
    if platform == ClientType.DISCORD:
        await postdeploy_discord(deployment)
    elif platform == ClientType.TELEGRAM:
        await postdeploy_telegram(deployment, secrets)
    elif platform == ClientType.FARCASTER:
        await postdeploy_farcaster(deployment, secrets)
    elif platform == ClientType.TWITTER:
        await postdeploy_twitter(deployment)
    else:
        raise APIError(f"Invalid platform: {platform}", status_code=400)


async def predeploy_discord(
    agent: Agent, secrets: DeploymentSecrets, config: DeploymentConfig
):
    """Validate Discord token and setup OAuth"""
    headers = {"Authorization": f"Bot {secrets.discord.token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://discord.com/api/v10/users/@me", headers=headers
        ) as response:
            if response.status != 200:
                raise APIError("Invalid Discord token", status_code=400)

            # Get application ID if not provided
            if not secrets.discord.application_id:
                bot_data = await response.json()
                application_id = bot_data.get("id")
                if application_id:
                    secrets.discord.application_id = application_id

                    # Setup Discord config
                    if not config:
                        config = DeploymentConfig()
                    if not config.discord:
                        config.discord = DeploymentSettingsDiscord()

                    # Create OAuth URL with the same permissions integer
                    permissions_integer = "309237771264"
                    oauth_url = f"https://discord.com/oauth2/authorize?client_id={application_id}&permissions={permissions_integer}&integration_type=0&scope=bot"

                    config.discord.oauth_client_id = application_id
                    config.discord.oauth_url = oauth_url

    # Add Discord tools to agent
    if not agent.tools:
        agent.tools = {}
        agent.add_base_tools = True

    agent.tools["discord_search"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.tools["discord_post"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.save()

    return secrets, config


async def predeploy_telegram(
    agent: Agent, secrets: DeploymentSecrets, config: DeploymentConfig
):
    """Validate Telegram token, generate webhook secret and add Telegram tools"""
    from telegram import Bot
    import secrets as python_secrets

    # Validate bot token
    try:
        bot = Bot(secrets.telegram.token)
        bot_info = await bot.get_me()
        print(f"Verified Telegram bot: {bot_info.username}")
    except Exception as e:
        raise APIError(f"Invalid Telegram token: {str(e)}", status_code=400)

    webhook_secret = python_secrets.token_urlsafe(32)
    secrets.telegram.webhook_secret = webhook_secret

    # Add Telegram tools to agent
    if not agent.tools:
        agent.tools = {}
        agent.add_base_tools = True

    agent.tools["telegram_post"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.save()

    return secrets, config


async def predeploy_farcaster(
    agent: Agent, secrets: DeploymentSecrets, config: DeploymentConfig
):
    """Verify Farcaster credentials"""
    try:
        from farcaster import Warpcast

        client = Warpcast(mnemonic=secrets.farcaster.mnemonic)

        # Test the credentials by getting user info
        user_info = client.get_me()
        print(f"Verified Farcaster credentials for user: {user_info}")
    except Exception as e:
        raise APIError(f"Invalid Farcaster credentials: {str(e)}", status_code=400)

    # Generate webhook secret if not provided
    if not secrets.farcaster.neynar_webhook_secret:
        import secrets as python_secrets

        webhook_secret = python_secrets.token_urlsafe(32)
        secrets.farcaster.neynar_webhook_secret = webhook_secret

    return secrets, config


async def predeploy_twitter(
    agent: Agent, secrets: DeploymentSecrets, config: DeploymentConfig
):
    """Validate Twitter credentials and add Twitter tools to agent"""
    import tweepy

    # Validate Twitter credentials
    try:
        # Create Twitter client with OAuth 1.0a
        auth = tweepy.OAuth1UserHandler(
            consumer_key=secrets.twitter.consumer_key,
            consumer_secret=secrets.twitter.consumer_secret,
            access_token=secrets.twitter.access_token,
            access_token_secret=secrets.twitter.access_token_secret,
        )
        api = tweepy.API(auth)

        # Test the credentials by getting user info
        user = api.verify_credentials()
        print(f"Verified Twitter credentials for user: @{user.screen_name}")

        # Also test with v2 API using bearer token
        client = tweepy.Client(
            bearer_token=secrets.twitter.bearer_token,
            consumer_key=secrets.twitter.consumer_key,
            consumer_secret=secrets.twitter.consumer_secret,
            access_token=secrets.twitter.access_token,
            access_token_secret=secrets.twitter.access_token_secret,
        )

        # Test v2 API
        me = client.get_me()
        print(f"Verified Twitter v2 API for user: @{me.data.username}")

    except Exception as e:
        raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

    if not agent.tools:
        agent.tools = {}
        agent.add_base_tools = True

    agent.tools["tweet"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.tools["twitter_mentions"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.tools["twitter_search"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }
    agent.save()

    return secrets, config


async def postdeploy_discord(deployment: Deployment):
    """Notify Discord gateway service via Ably"""
    try:
        from ably import AblyRest

        ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        channel = ably_client.channels.get(f"discord-gateway-{db}")

        await channel.publish(
            "command", {"command": "start", "deployment_id": str(deployment.id)}
        )
        print(f"Sent start command for deployment {deployment.id} via Ably")
    except Exception as e:
        raise Exception(f"Failed to notify gateway service: {e}")


async def postdeploy_telegram(deployment: Deployment, secrets: DeploymentSecrets):
    """Set Telegram webhook and notify gateway"""
    from telegram import Bot

    webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/telegram"

    # Update bot webhook
    response = await Bot(secrets.telegram.token).set_webhook(
        url=webhook_url,
        secret_token=secrets.telegram.webhook_secret,
        drop_pending_updates=True,
        max_connections=100,
    )

    if not response:
        raise Exception("Failed to set Telegram webhook")

    # Notify gateway about the new deployment
    try:
        from ably import AblyRest

        ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        channel = ably_client.channels.get(f"discord-gateway-{db}")

        await channel.publish(
            "command",
            {
                "command": "register_telegram",
                "deployment_id": str(deployment.id),
                "token": secrets.telegram.token,
            },
        )
        print(
            f"Sent Telegram registration command for deployment {deployment.id} via Ably"
        )
    except Exception as e:
        raise Exception(f"Failed to notify gateway service for Telegram: {e}")


async def postdeploy_farcaster(deployment: Deployment, secrets: DeploymentSecrets):
    """Register webhook with Neynar"""
    webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/farcaster"

    async with aiohttp.ClientSession() as session:
        # Get Neynar API key from environment
        neynar_api_key = os.getenv("NEYNAR_API_KEY")
        if not neynar_api_key:
            raise Exception("NEYNAR_API_KEY not found in environment")

        headers = {
            "x-api-key": f"{neynar_api_key}",
            "Content-Type": "application/json",
        }

        # Get user info for webhook registration
        from farcaster import Warpcast

        client = Warpcast(mnemonic=secrets.farcaster.mnemonic)
        user_info = client.get_me()

        webhook_data = {
            "name": f"eden-{deployment.id}",
            "url": webhook_url,
            "subscription": {"cast.created": {"mentioned_fids": [user_info.fid]}},
        }

        async with session.post(
            "https://api.neynar.com/v2/farcaster/webhook",
            headers=headers,
            json=webhook_data,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to register Neynar webhook: {error_text}")

            webhook_response = await response.json()
            webhook_id = webhook_response.get("webhook", {}).get("webhook_id")
            webhook_secret = (
                webhook_response.get("webhook", {}).get("secrets", [{}])[0].get("value")
            )

            if not webhook_id:
                raise Exception("No webhook_id in response")

            # Update the webhook secret in deployment secrets
            secrets.farcaster.neynar_webhook_secret = webhook_secret

            # Store webhook ID in deployment for later cleanup
            if not deployment.config:
                deployment.config = DeploymentConfig()
            if not deployment.config.farcaster:
                deployment.config.farcaster = DeploymentSettingsFarcaster()

            deployment.config.farcaster.webhook_id = webhook_id
            deployment.secrets = secrets
            deployment.save()

            print(
                f"Registered Neynar webhook {webhook_id} for deployment {deployment.id}"
            )


async def postdeploy_twitter(deployment: Deployment):
    """No post-deployment actions needed for Twitter"""
    pass


async def postdeploy_modal(
    deployment: Deployment,
    agent: Agent,
    platform: ClientType,
    secrets: DeploymentSecrets,
    env: str,
    repo_branch: str = None,
):
    """Deploy Modal client"""
    deploy_client_modal(agent, platform, secrets, env, repo_branch)


@handle_errors
async def handle_deployment_update(request: UpdateDeploymentRequest):
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.deployment_id}", status_code=404
        )

    # Handle partial config updates by merging with existing config
    if request.config:
        existing_config = deployment.config or DeploymentConfig()
        new_config = request.config.model_dump(exclude_unset=True)

        # Merge the configs at the platform level
        updated_config_dict = existing_config.model_dump() if existing_config else {}

        for platform, platform_config in new_config.items():
            if platform_config is not None:
                if platform in updated_config_dict:
                    # Merge platform-specific configs
                    updated_config_dict[platform].update(platform_config)
                else:
                    # Add new platform config
                    updated_config_dict[platform] = platform_config

        # Convert to dict for MongoDB storage
        deployment.update(config=updated_config_dict)

    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_deployment_delete(request: DeleteDeploymentRequest):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent}", status_code=404)

    try:
        await stop_client(agent, request.platform)

        # Delete deployment record
        Deployment.delete_deployment(agent.id, request.platform.value)

        return {"success": True}
    except Exception as e:
        raise APIError(f"Failed to stop client: {str(e)}", status_code=500)


@handle_errors
async def handle_trigger_create(
    request: CreateTriggerRequest, background_tasks: BackgroundTasks
):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent}", status_code=404)

    user = User.from_mongo(ObjectId(request.user))
    if not user:
        raise APIError(f"User not found: {request.user}", status_code=404)

    trigger_id = f"{str(user.id)}_{int(time.time())}"

    background_tasks.add_task(
        create_trigger_fn,
        schedule=request.schedule.to_cron_dict(),
        trigger_id=trigger_id,
    )

    trigger = Trigger(
        trigger_id=trigger_id,
        user=ObjectId(user.id),
        agent=ObjectId(agent.id),
        schedule=request.schedule.to_cron_dict(),
        instruction=request.instruction,
        posting_instructions=request.posting_instructions.model_dump() if request.posting_instructions else None,
        update_config=request.update_config.model_dump()
        if request.update_config
        else None,
        session=ObjectId(request.session) if request.session else None,
        session_type=request.session_type,
    )
    trigger.save()

    return {
        "id": str(trigger.id),
        "trigger_id": trigger_id,
    }


@handle_errors
async def handle_trigger_stop(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if not trigger:
        raise APIError(f"Trigger not found: {request.id}", status_code=404)
    await stop_trigger(trigger.trigger_id)
    trigger.status = "finished"
    trigger.save()

    return {"id": str(request.id)}


@handle_errors
async def handle_trigger_delete(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if trigger.status != "finished":
        await stop_trigger(trigger.trigger_id)

    trigger.delete()

    return {"id": str(trigger.id)}


@handle_errors
async def handle_twitter_update(request: PlatformUpdateRequest):
    """Handle Twitter updates from async_prompt_thread"""

    deployment_id = request.update_config.deployment_id

    # Get deployment
    deployment = Deployment.from_mongo(ObjectId(deployment_id))
    if not deployment:
        raise APIError(f"Deployment not found: {deployment_id}")

    # Initialize Twitter client
    twitter = X(deployment)
    reply_to = request.update_config.twitter_tweet_to_reply_id

    # Post tweet
    tweet_id = None
    if request.type == UpdateType.ASSISTANT_MESSAGE:
        if reply_to:
            # Reply to specpific tweet
            response = twitter.post(
                text=request.content,
                reply_to=reply_to,
            )
        else:
            # Regular tweet
            response = twitter.post(text=request.content)
        tweet_id = response.get("data", {}).get("id")

    return {"status": "success", "tweet_id": tweet_id}


@handle_errors
async def handle_trigger_get(trigger_id: str):
    trigger = Trigger.load(trigger_id=trigger_id)
    if not trigger:
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    return {
        "id": str(trigger.id) if trigger.id else None,
        "user": str(trigger.user) if trigger.user else None,
        "agent": str(trigger.agent) if trigger.agent else None,
        "session": str(trigger.session) if trigger.session else None,
        "instruction": trigger.instruction,
        "update_config": trigger.update_config,
        "schedule": trigger.schedule,
    }


@handle_errors
async def handle_farcaster_update(request: Request):
    """Handle webhook updates from Neynar for Farcaster"""
    try:
        import hmac
        import hashlib

        # Verify Neynar webhook signature
        body = await request.body()
        signature = request.headers.get("X-Neynar-Signature")
        if not signature:
            return JSONResponse(
                status_code=401, content={"error": "Missing signature header"}
            )

        # Find deployment by webhook secret - we'll store this in the deployment
        # For now, let's extract the webhook secret from headers or find another way
        webhook_data = await request.json()
        cast_data = webhook_data.get("data", {})

        if not cast_data or "hash" not in cast_data:
            return JSONResponse(status_code=400, content={"error": "Invalid cast data"})

        # For now, we'll need to find the deployment differently
        # We could use the cast author or have Neynar include a custom field
        # Let's assume we can match by the webhook signature for now
        deployment = None
        for d in Deployment.find({"platform": "farcaster"}):
            if (
                d.secrets
                and d.secrets.farcaster
                and d.secrets.farcaster.neynar_webhook_secret
            ):
                # Verify signature
                computed_signature = hmac.new(
                    d.secrets.farcaster.neynar_webhook_secret.encode(),
                    body,
                    hashlib.sha512,
                ).hexdigest()
                if hmac.compare_digest(computed_signature, signature):
                    deployment = d
                    break

        if not deployment:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid signature or deployment not found"},
            )
        if not deployment.config.farcaster.auto_reply:
            return JSONResponse(status_code=200, content={"ok": True})

        # Create chat request similar to Telegram
        cast_hash = cast_data["hash"]
        author = cast_data["author"]
        author_username = author["username"]
        author_fid = author["fid"]

        # Get or create user
        user = User.from_farcaster(author_fid, author_username)

        # Get agent
        agent = Agent.from_mongo(deployment.agent)
        if not agent:
            return JSONResponse(status_code=404, content={"error": "Agent not found"})

        # Get or create thread
        thread_key = f"farcaster-{author_fid}-{cast_hash}"
        thread = agent.request_thread(key=thread_key)

        chat_request_data = {
            "user_id": str(user.id),
            "agent_id": str(agent.id),
            "thread_id": str(thread.id),
            "force_reply": True,
            "user_message": {
                "content": cast_data.get("text", ""),
                "name": author_username,
            },
            "update_config": {
                "deployment_id": str(deployment.id),
                "update_endpoint": f"{os.getenv('EDEN_API_URL')}/emissions/platform/farcaster",
                "farcaster_hash": cast_hash,
                "farcaster_author_fid": author_fid,
            },
        }

        # Make async HTTP POST to /chat
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/chat",
                json=chat_request_data,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"Failed to process chat request: {error_text}"
                        },
                    )

        return JSONResponse(status_code=200, content={"ok": True})

    except Exception as e:
        logger.error("Error processing Farcaster update", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@handle_errors
async def handle_farcaster_emission(request: Request):
    """Handle updates from async_prompt_thread for Farcaster"""
    try:
        data = await request.json()
        print("FARCASTER EMISSION DATA:", data)

        update_type = data.get("type")
        update_config = data.get("update_config", {})
        deployment_id = update_config.get("deployment_id")
        cast_hash = update_config.get("farcaster_hash")
        author_fid = update_config.get("farcaster_author_fid")

        if not deployment_id or not cast_hash or not author_fid:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Deployment ID, cast hash, and author FID are required"
                },
            )

        # Find deployment
        deployment = Deployment.from_mongo(ObjectId(deployment_id))
        if not deployment:
            return JSONResponse(
                status_code=404, content={"error": "No Farcaster deployment found"}
            )

        # Initialize Farcaster client
        from farcaster import Warpcast

        client = Warpcast(mnemonic=deployment.secrets.farcaster.mnemonic)

        if update_type == UpdateType.ASSISTANT_MESSAGE:
            content = data.get("content")
            if content:
                try:
                    client.post_cast(
                        text=content,
                        parent={"hash": cast_hash, "fid": author_fid},
                    )
                except Exception as e:
                    logger.error(f"Failed to post cast: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Failed to post cast: {str(e)}"},
                    )

        elif update_type == UpdateType.TOOL_COMPLETE:
            result = data.get("result", {})
            if not result:
                return JSONResponse(status_code=200, content={"ok": True})

            result["result"] = prepare_result(result["result"])
            outputs = result["result"][0]["output"]
            urls = [output["url"] for output in outputs[:4]]  # Get up to 4 URLs

            try:
                client.post_cast(
                    text="",
                    embeds=urls,
                    parent={"hash": cast_hash, "fid": author_fid},
                )
            except Exception as e:
                logger.error(f"Failed to post cast with embeds: {str(e)}")
                return JSONResponse(
                    status_code=500, content={"error": f"Failed to post cast: {str(e)}"}
                )

        elif update_type == UpdateType.ERROR:
            error_msg = data.get("error", "Unknown error occurred")
            try:
                client.post_cast(
                    text=f"Error: {error_msg}",
                    parent={"hash": cast_hash, "fid": author_fid},
                )
            except Exception as e:
                logger.error(f"Failed to post error cast: {str(e)}")

        return JSONResponse(status_code=200, content={"ok": True})

    except Exception as e:
        logger.error("Error handling Farcaster emission", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@handle_errors
async def handle_telegram_emission(request: Request):
    """Handle updates from async_prompt_thread for Telegram"""
    try:
        data = await request.json()
        print("TELEGRAM EMISSION DATA:", data)

        update_type = data.get("type")
        update_config = data.get("update_config", {})
        deployment_id = update_config.get("deployment_id")

        if not deployment_id:
            return JSONResponse(
                status_code=400, content={"error": "Deployment ID is required"}
            )

        # Convert chat_id to int
        chat_id = int(update_config.get("telegram_chat_id"))
        message_id = (
            int(update_config.get("telegram_message_id"))
            if update_config.get("telegram_message_id")
            else None
        )
        thread_id = (
            int(update_config.get("telegram_thread_id"))
            if update_config.get("telegram_thread_id")
            else None
        )

        # Find deployment
        deployment = Deployment.from_mongo(ObjectId(deployment_id))
        if not deployment:
            return JSONResponse(
                status_code=404, content={"error": "No Telegram deployment found"}
            )

        # Initialize bot
        from telegram import Bot

        bot = Bot(deployment.secrets.telegram.token)

        # Verify bot info
        try:
            me = await bot.get_me()
            print("BOT INFO:", me.to_dict())
        except Exception as e:
            print("Failed to get bot info:", str(e))
            return JSONResponse(
                status_code=500,
                content={"error": f"Bot authentication failed: {str(e)}"},
            )

        if update_type == UpdateType.ASSISTANT_MESSAGE:
            content = data.get("content")
            if content:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=content,
                        reply_to_message_id=message_id,
                        message_thread_id=thread_id,
                    )
                except Exception as e:
                    print(f"Failed to send message: {str(e)}")
                    print(
                        f"Params: chat_id={chat_id}, text={content}, reply_to={message_id}, thread_id={thread_id}"
                    )
                    raise

        elif update_type == UpdateType.TOOL_COMPLETE:
            result = data.get("result", {})
            if not result:
                return JSONResponse(status_code=200, content={"ok": True})

            result["result"] = prepare_result(result["result"])
            outputs = result["result"][0]["output"]
            urls = [output["url"] for output in outputs[:4]]  # Get up to 4 URLs

            # Send each URL as appropriate media type
            for url in urls:
                video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
                if any(url.lower().endswith(ext) for ext in video_extensions):
                    await bot.send_video(
                        chat_id=chat_id,
                        video=url,
                        reply_to_message_id=message_id,
                        message_thread_id=thread_id,
                    )
                else:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=url,
                        reply_to_message_id=message_id,
                        message_thread_id=thread_id,
                    )

        return JSONResponse(status_code=200, content={"ok": True})

    except Exception as e:
        logger.error("Error handling Telegram emission", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@handle_errors
async def handle_telegram_update(request: Request):
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if not secret_token:
        return JSONResponse(status_code=401, content={"error": "Missing secret token"})

    try:
        update_data = await request.json()
        print("TELEGRAM UPDATE DATA:", update_data)

        # Find deployment by webhook secret
        deployment = next(
            (
                d
                for d in Deployment.find({"platform": "telegram"})
                if d.secrets
                and d.secrets.telegram
                and d.secrets.telegram.webhook_secret == secret_token
            ),
            None,
        )

        if not deployment:
            return JSONResponse(
                status_code=401, content={"error": "Invalid secret token"}
            )

        # Create chat request with endpoint for updates
        chat_request = await create_telegram_chat_request(update_data, deployment)
        if not chat_request:
            return JSONResponse(status_code=200, content={"ok": True})

        # Make async HTTP POST to /chat
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/chat",
                json=chat_request,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                print("CHAT RESPONSE:", response.status)
                if response.status != 200:
                    error_text = await response.text()
                    print("CHAT ERROR:", error_text)
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"Failed to process chat request: {error_text}"
                        },
                    )

        return JSONResponse(status_code=200, content={"ok": True})

    except Exception as e:
        logger.error("Error processing Telegram update", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@handle_errors
async def handle_discord_emission(request: Request):
    """Handle updates from async_prompt_thread for Discord"""
    try:
        data = await request.json()

        update_type = data.get("type")
        update_config = data.get("update_config", {})
        deployment_id = update_config.get("deployment_id")
        channel_id = update_config.get("discord_channel_id")
        message_id = update_config.get("discord_message_id")

        if not deployment_id or not channel_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Deployment ID and channel ID are required"},
            )

        payload = {}
        if message_id:
            payload["message_reference"] = {
                "message_id": message_id,
                "channel_id": channel_id,
                "fail_if_not_exists": False,
            }

        # Find deployment
        deployment = Deployment.from_mongo(ObjectId(deployment_id))
        if not deployment:
            return JSONResponse(
                status_code=404, content={"error": "No Discord deployment found"}
            )

        # Initialize Discord REST client for sending messages
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bot {deployment.secrets.discord.token}",
                "Content-Type": "application/json",
            }

            if update_type == UpdateType.ASSISTANT_MESSAGE:
                content = data.get("content")
                if content:
                    payload["content"] = content

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = data.get("result", {})
                if not result:
                    return JSONResponse(status_code=200, content={"ok": True})

                result["result"] = prepare_result(result["result"])
                print("RESULT", result)
                outputs = result["result"][0]["output"]
                urls = [
                    output["url"] for output in outputs[:4] if "url" in output
                ]  # Get up to 4 URLs with valid urls

                # Get creation ID from the first output
                creation_id = None
                if isinstance(outputs, list) and len(outputs) > 0:
                    creation_id = str(outputs[0].get("creation"))

                # Prepare message content with URLs
                content = "\n".join(urls)

                payload["content"] = content

                # Add components for Eden link if creation_id exists
                if creation_id:
                    eden_url = get_eden_creation_url(creation_id)
                    payload["components"] = [
                        {
                            "type": 1,  # Action Row
                            "components": [
                                {
                                    "type": 2,  # Button
                                    "style": 5,  # Link
                                    "label": "View on Eden",
                                    "url": eden_url,
                                }
                            ],
                        }
                    ]

            if payload["content"]:
                async with session.post(
                    f"https://discord.com/api/v10/channels/{channel_id}/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to send Discord message: {error_text}")
                        return JSONResponse(
                            status_code=500,
                            content={"error": f"Failed to send message: {error_text}"},
                        )

        return JSONResponse(status_code=200, content={"ok": True})

    except Exception as e:
        logger.error("Error handling Discord emission", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@handle_errors
async def handle_agent_tools_update(request: AgentToolsUpdateRequest):
    agent = Agent.from_mongo(ObjectId(request.agent_id))
    if not agent:
        raise APIError(f"Agent not found: {request.agent_id}", status_code=404)
    # Upsert tools
    tools = agent.tools or {}
    tools.update(request.tools)
    update = {"tools": tools, "add_base_tools": True}
    agents = get_collection("users3")
    agents.update_one({"_id": agent.id}, {"$set": update})
    return {"tools": tools}


@handle_errors
async def handle_agent_tools_delete(request: AgentToolsDeleteRequest):
    agent = Agent.from_mongo(ObjectId(request.agent_id))
    if not agent:
        raise APIError(f"Agent not found: {request.agent_id}", status_code=404)
    tools = agent.tools or {}
    for tool in request.tools:
        tools.pop(tool, None)
    update = {"tools": tools}
    agents = get_collection("users3")
    agents.update_one({"_id": agent.id}, {"$set": update})
    return {"tools": tools}


def create_eden_message(
    session_id: ObjectId, message_type: EdenMessageType, agents: List[Agent]
) -> ChatMessage:
    """Create an eden message for agent operations"""
    eden_message = ChatMessage(
        session=session_id,
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content="",
        eden_message_data=EdenMessageData(
            message_type=message_type,
            agents=[
                EdenMessageAgentData(
                    id=agent.id,
                    name=agent.name or agent.username,
                    avatar=agent.userImage,
                )
                for agent in agents
            ],
        ),
    )
    eden_message.save()
    return eden_message


def generate_session_title(
    session: Session, request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    from eve.agent.session.session import async_title_session

    if session.title:
        return

    if request.creation_args and request.creation_args.title:
        return

    background_tasks.add_task(async_title_session, session, request.message.content)


def setup_session(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request: PromptSessionRequest = None,
):
    if session_id:
        session = Session.from_mongo(ObjectId(session_id))
        if not session:
            raise APIError(f"Session not found: {session_id}", status_code=404)
        generate_session_title(session, request, background_tasks)
        return session

    if not request.creation_args:
        raise APIError(
            "Session creation requires additional parameters", status_code=400
        )

    # Create new session
    agent_object_ids = [ObjectId(agent_id) for agent_id in request.creation_args.agents]
    session_kwargs = {
        "owner": ObjectId(request.creation_args.owner_id or user_id),
        "agents": agent_object_ids,
        "title": request.creation_args.title,
        "scenario": request.creation_args.scenario,
        "status": "active",
        "trigger": ObjectId(request.creation_args.trigger)
        if request.creation_args.trigger
        else None,
    }

    # Only include budget if it's not None, so default factory can work
    if request.creation_args.budget is not None:
        session_kwargs["budget"] = request.creation_args.budget

    session = Session(**session_kwargs)
    session.save()

    # Update trigger with session ID
    if request.creation_args.trigger:
        trigger = Trigger.from_mongo(ObjectId(request.creation_args.trigger))
        if trigger:
            trigger.session = session.id
            trigger.save()

    # Create eden message for initial agent additions
    agents = [Agent.from_mongo(agent_id) for agent_id in agent_object_ids]
    agents = [agent for agent in agents if agent]  # Filter out None values
    if agents:
        eden_message = create_eden_message(
            session.id, EdenMessageType.AGENT_ADD, agents
        )
        session.messages.append(eden_message.id)
        session.save()

    # Generate title for new sessions if no title provided and we have background tasks
    generate_session_title(session, request, background_tasks)

    return session


@handle_errors
async def handle_prompt_session(
    request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    session = setup_session(
        background_tasks, request.session_id, request.user_id, request
    )
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        actor_agent_id=request.actor_agent_id,
        message=request.message,
        update_config=request.update_config,
    )

    if request.stream:

        async def event_generator():
            try:
                async for data in run_prompt_session_stream(context, background_tasks):
                    yield f"data: {json.dumps({'event': 'update', 'data': data})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
            except Exception as e:
                logger.error("Error in event_generator", exc_info=True)
                yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    background_tasks.add_task(
        run_prompt_session,
        context=context,
        background_tasks=background_tasks,
    )

    return {"session_id": str(session.id)}


@handle_errors
async def handle_session_cancel(request: CancelSessionRequest):
    """Cancel a running prompt session by sending a cancel signal via Ably."""
    try:
        from ably import AblyRest

        # Verify session exists and user has permission
        session = Session.from_mongo(ObjectId(request.session_id))
        if not session:
            raise APIError(f"Session not found: {request.session_id}", status_code=404)

        # Check if user has permission to cancel this session
        if str(session.owner) != request.user_id:
            raise APIError(
                "Unauthorized: User does not own this session", status_code=403
            )

        # Send cancel signal via Ably
        ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        channel_name = f"{os.getenv('DB')}-session-cancel-{request.session_id}"
        channel = ably_client.channels.get(channel_name)

        await channel.publish(
            "cancel",
            {
                "session_id": request.session_id,
                "user_id": request.user_id,
                "timestamp": time.time(),
            },
        )

        logger.info(f"Sent cancellation signal for session {request.session_id}")
        return {"status": "cancel_signal_sent", "session_id": request.session_id}

    except Exception as e:
        logger.error(f"Error sending session cancel signal: {e}", exc_info=True)
        raise APIError(f"Failed to send cancel signal: {str(e)}", status_code=500)
