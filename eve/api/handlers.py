import json
import logging
import os
import time
from bson import ObjectId
from typing import List
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import aiohttp

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
)
from eve.clients.common import get_ably_channel_name
from eve.deploy import (
    deploy_client,
    stop_client,
)
from eve.eden_utils import prepare_result
from eve.tools.replicate_tool import replicate_update_task
from eve.trigger import create_chat_trigger, delete_trigger, Trigger
from eve.agent.llm import UpdateType
from eve.agent.run import async_prompt_thread
from eve.mongo import serialize_document
from eve.task import Task
from eve.tool import Tool
from eve.agent.agent import Agent
from eve.user import User
from eve.agent.thread import Thread, UserMessage
from eve.deploy import Deployment
from eve.tools.twitter import X

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
    print("___handle_replicate_webhook")
    task = Task.from_handler_id(body["id"])
    tool = Tool.load(task.tool)
    _ = replicate_update_task(
        task, body["status"], body["error"], body["output"], tool.output_handler
    )
    print("___handle_replicate_webhook success !")
    return {"status": "success"}


async def run_chat_request(
    user: User,
    agent: Agent,
    thread: Thread,
    tools: List[Tool],
    user_message: UserMessage,
    update_config: UpdateConfig,
    force_reply: bool,
    model: str,
    user_is_bot: bool = False,
):
    try:
        async for update in async_prompt_thread(
            user=user,
            agent=agent,
            thread=thread,
            user_messages=user_message,
            tools=tools,
            force_reply=force_reply,
            model=model,
            user_is_bot=user_is_bot,
            stream=False,
        ):
            print("UPDATE", update)
            data = {
                "type": update.type.value,
                "update_config": update_config.model_dump() if update_config else {},
            }

            if update.type == UpdateType.ASSISTANT_MESSAGE:
                data["content"] = update.message.content
            elif update.type == UpdateType.TOOL_COMPLETE:
                data["tool"] = update.tool_name
                data["result"] = serialize_for_json(update.result)
            elif update.type == UpdateType.ERROR:
                data["error"] = update.error if hasattr(update, "error") else None

            await emit_update(update_config, data)

    except Exception as e:
        logger.error("Error in run_prompt", exc_info=True)
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

    secrets = await deploy_client(
        agent, request.platform, request.secrets, db.lower(), request.repo_branch
    )

    # Create/update deployment record
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
    agent = Agent.from_mongo(deployment.agent)

    try:
        channel_name = get_ably_channel_name(agent.username, deployment.platform)
        await emit_update(
            UpdateConfig(sub_channel_name=channel_name), {"type": "RELOAD_DEPLOYMENT"}
        )
    except Exception as e:
        logger.error(f"Failed to emit deployment reload message: {str(e)}")

    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_deployment_delete(request: DeleteDeploymentRequest):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent}", status_code=404)

    try:
        stop_client(agent, request.platform)

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
