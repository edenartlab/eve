import json
import logging
import os
import time
from bson import ObjectId
from typing import List
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from modal import Function

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
)
from eve.clients.common import get_ably_channel_name
from eve.deploy import (
    stop_client,
)
from eve.tools.replicate_tool import replicate_update_task
from eve.trigger import create_chat_trigger, delete_trigger
from eve.llm import UpdateType, async_prompt_thread
from eve.mongo import serialize_document
from eve.task import Task
from eve.tool import Tool
from eve.agent import Agent
from eve.user import User
from eve.thread import Thread, UserMessage
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
    print("update_config", update_config)
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
    user, agent, thread, tools = await setup_chat(request, background_tasks)

    print("chat request")
    print(request)

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
    user, agent, thread, tools = await setup_chat(request, background_tasks)

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

    try:
        deploy_func = Function.lookup(
            f"api-{db.lower()}", "deploy_client_modal", environment_name="main"
        )
        deploy_func.remote(
            agent_id=str(agent.id),
            agent_key=agent.username,
            platform=request.platform.value,
            secrets=request.secrets.model_dump(),
            env=db.lower(),
            repo_branch=request.repo_branch,
        )
    except Exception as e:
        raise APIError(f"Failed to deploy client: {str(e)}", status_code=500)

    # Create/update deployment record
    deployment = Deployment(
        agent=agent.id,
        user=ObjectId(request.user),
        platform=request.platform,
        secrets=request.secrets,
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
        stop_client(agent, request.platform.value)

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
