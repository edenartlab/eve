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

from langfuse.decorators import observe, langfuse_context
from eve import LANGFUSE_ENV
from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CancelRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    TaskRequest,
    PlatformUpdateRequest,
    UpdateConfig,
    UpdateDeploymentRequest,
)
from eve.api.helpers import (
    emit_update,
    serialize_for_json,
    setup_chat,
    create_telegram_chat_request,
    update_busy_state,
)
from eve.deploy import (
    deploy_client,
    modify_secrets,
    stop_client,
)
from eve.eden_utils import prepare_result
from eve.tools.replicate_tool import replicate_update_task
from eve.trigger import create_chat_trigger, delete_trigger, Trigger
from eve.agent.llm import UpdateType
from eve.agent.run_thread import async_prompt_thread
from eve.mongo import serialize_document
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

USE_RATE_LIMITS = os.getenv("USE_RATE_LIMITS", "false").lower() == "true"


@handle_errors
async def handle_create(request: TaskRequest):
    tool = Tool.load(key=request.tool)

    print("#### create ####")
    print(request)

    # if USE_RATE_LIMITS:
    #     await RateLimiter().check_create_rate_limit(user, tool)

    print("### run the tool ###")
    result = await tool.async_start_task(
        requester_id=request.user_id, user_id=request.user_id, args=request.args
    )

    print("### return the result ###")
    print(result)

    return serialize_document(result.model_dump(by_alias=True))


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


@observe()
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

    metadata = {
        "user_id": str(user.id),
        "agent_id": str(agent.id),
        "thread_id": str(thread.id),
        "request_id": request_id,
        "environment": LANGFUSE_ENV,
    }

    langfuse_context.update_current_trace(user_id=str(user.id))
    langfuse_context.update_current_observation(metadata=metadata)

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
        ):
            print("UPDATE", update)
            data = {
                "type": update.type.value,
                "update_config": update_config.model_dump() if update_config else {},
            }

            if update.type == UpdateType.START_PROMPT:
                update_busy_state(update_config, request_id, True)
            elif update.type == UpdateType.ASSISTANT_MESSAGE:
                data["content"] = update.message.content
            elif update.type == UpdateType.TOOL_COMPLETE:
                data["tool"] = update.tool_name
                data["result"] = serialize_for_json(update.result)
            elif update.type == UpdateType.ERROR:
                data["error"] = update.error if hasattr(update, "error") else None
            elif update.type == UpdateType.END_PROMPT:
                update_busy_state(update_config, request_id, False)
            await emit_update(update_config, data)

    except Exception as e:
        logger.error("Error in run_prompt", exc_info=True)
        update_busy_state(update_config, request_id, False)
        await emit_update(
            update_config,
            {"type": "error", "error": str(e)},
        )


async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    print("handle_chat")
    print(request)

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
                            serialize_for_json(t.model_dump())
                            for t in update.message.tool_calls
                        ]
                elif update.type == UpdateType.TOOL_COMPLETE:
                    data["tool"] = update.tool_name
                    data["result"] = serialize_for_json(update.result)
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

    secrets = await modify_secrets(request.secrets, request.platform)

    deployment = Deployment(
        agent=agent.id,
        user=ObjectId(request.user),
        platform=request.platform,
        secrets=secrets,
        config=request.config,
    )
    deployment.save(
        upsert_filter={"agent": agent.id, "platform": request.platform.value}
    )

    try:
        await deploy_client(
            deployment,
            agent,
            request.platform,
            request.secrets,
            db.lower(),
            request.repo_branch,
        )
    except Exception as e:
        logger.error(f"Failed to deploy client: {str(e)}")
        deployment.delete()
        raise APIError(f"Failed to deploy client: {str(e)}", status_code=500)

    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_deployment_update(request: UpdateDeploymentRequest):
    print("deployment update request", request)

    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.deployment_id}", status_code=404
        )

    deployment.update(config=request.config.model_dump())

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
async def handle_trigger_create(request: CreateTriggerRequest):
    trigger_id = f"{request.user_id}_{request.agent_id}_{int(time.time())}"

    await create_chat_trigger(
        schedule=request.schedule.to_cron_dict(),
        trigger_id=trigger_id,
    )

    agent = Agent.from_mongo(ObjectId(request.agent_id))
    thread = agent.request_thread(user=ObjectId(request.user_id), key=trigger_id)

    trigger = Trigger(
        trigger_id=trigger_id,
        user=ObjectId(request.user_id),
        agent=ObjectId(request.agent_id),
        thread=thread.id,
        schedule=request.schedule.to_cron_dict(),
        message=request.message,
        update_config=request.update_config.model_dump()
        if request.update_config
        else {},
    )
    trigger.save()

    return {
        "id": str(trigger.id),
        "trigger_id": trigger_id,
    }


@handle_errors
async def handle_trigger_delete(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    await delete_trigger(trigger.trigger_id)
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
        "user": str(trigger.user),
        "agent": str(trigger.agent),
        "thread": str(trigger.thread),
        "message": trigger.message,
        "update_config": trigger.update_config,
    }


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
        print("DISCORD EMISSION DATA:", data)

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
                    payload = {
                        "content": content,
                        "message_reference": {
                            "message_id": message_id,
                            "channel_id": channel_id,
                            "fail_if_not_exists": False,
                        },
                    }

                    async with session.post(
                        f"https://discord.com/api/v10/channels/{channel_id}/messages",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Failed to send Discord message: {error_text}"
                            )
                            return JSONResponse(
                                status_code=500,
                                content={
                                    "error": f"Failed to send message: {error_text}"
                                },
                            )

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = data.get("result", {})
                if not result:
                    return JSONResponse(status_code=200, content={"ok": True})

                result["result"] = prepare_result(result["result"])
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

                # Basic message payload
                payload = {
                    "content": content,
                    "message_reference": {
                        "message_id": message_id,
                        "channel_id": channel_id,
                        "fail_if_not_exists": False,
                    },
                }

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


























################################################################################
# Below this is experimental

################################################################################
################################################################################
# Sessions (v2)
################################################################################

import traceback
from pydantic import BaseModel
from eve.agent.session import Session, SessionMessage
from eve.agent.run_thread import async_run_session

class SessionMessageRequest(BaseModel):
    user_id: str
    user_message: UserMessage
    session_id: Optional[str] = None
    update_config: Optional[UpdateConfig] = None


async def handle_session_message(
    request: SessionMessageRequest,
    # background_tasks: BackgroundTasks,
):
    print("handle_session_message")
    print(request)

    user, session = await setup_session(
        request
    )

    # background_tasks.add_task(
    #     run_session_request,
    #     user,
    #     session,
    #     request.user_message,
    #     request.update_config,
    # )

    await run_session_request(
        user,
        session,
        request.user_message,
        request.update_config,
    )

    return {"session_id": str(session.id)}



async def setup_session(
    request: SessionMessageRequest,
    # cache: bool = False,
    # background_tasks: BackgroundTasks = None,
    # metadata: Optional[Dict] = None,
) -> tuple[User, Session]:
    try:
        user = User.from_mongo(request.user_id)
    except Exception as e:
        logger.error(f"Error loading user: {traceback.format_exc()}")
        raise APIError(f"Invalid user_id: {request.user_id}", status_code=400) from e

    # try:
    #     agent = Agent.from_mongo(request.agent_id, cache=False)
    # except Exception as e:
    #     logger.error(f"Error loading agent: {traceback.format_exc()}")
    #     raise APIError(f"Invalid agent_id: {request.agent_id}", status_code=400) from e

    # tools = agent.get_tools(cache=cache)

    if request.session_id:
        try:
            session = Session.from_mongo(request.session_id)
        except Exception as e:
            logger.error(f"Error loading thread: {traceback.format_exc()}")
            raise APIError(
                f"Invalid thread_id: {request.thread_id}", status_code=400
            ) from e
    else:
        # thread = agent.request_thread(user=user.id, message_limit=25)
        session = Session(
            scenario="example scenario",
            agents=[]
        )
        session.save()

    return user, session


@observe()
async def run_session_request(
    user: User,
    session: Session,
    user_message: UserMessage,
    update_config: UpdateConfig,
):
    request_id = str(uuid.uuid4())

    metadata = {
        "user_id": str(user.id),
        "session_id": str(session.id),
        "request_id": request_id,
        "environment": LANGFUSE_ENV,
    }

    langfuse_context.update_current_trace(user_id=str(user.id))
    langfuse_context.update_current_observation(metadata=metadata)

    try:
        result = await async_run_session(
            user=user,
            session=session,
            user_messages=user_message,
        )
        print(result)

    except Exception as e:
        logger.error("Error in run_session", exc_info=True)
        update_busy_state(update_config, request_id, False)
        await emit_update(
            update_config,
            {"type": "error", "error": str(e)},
        )