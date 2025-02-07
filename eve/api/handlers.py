import json
import logging
import os
import time
from bson import ObjectId
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from apscheduler.schedulers.background import BackgroundScheduler

from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CancelRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    TaskRequest,
    ConfigureDeploymentRequest,
    PlatformUpdateRequest,
)
from eve.api.helpers import (
    emit_update,
    serialize_for_json,
    setup_chat,
)
from eve.deploy import (
    create_modal_secrets,
    stop_client,
)
from eve.tools.replicate_tool import replicate_update_task
from eve.trigger import Trigger
from eve.llm import UpdateType, async_prompt_thread
from eve.mongo import serialize_document
from eve.task import Task
from eve.tool import Tool
from eve.agent import Agent
from eve.deploy import Deployment
from eve.tools.twitter import X

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()

USE_RATE_LIMITS = os.getenv("USE_RATE_LIMITS", "false").lower() == "true"


@handle_errors
async def handle_create(request: TaskRequest):
    tool = Tool.load(key=request.tool)

    # if USE_RATE_LIMITS:
    #     await RateLimiter().check_create_rate_limit(user, tool)

    result = await tool.async_start_task(
        requester_id=request.user_id, user_id=request.user_id, args=request.args
    )

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


async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    user, agent, thread, tools = await setup_chat(request, background_tasks)

    async def run_prompt():
        try:
            async for update in async_prompt_thread(
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                dont_reply=request.dont_reply,
                model=request.model,
                stream=False,
            ):
                data = {
                    "type": update.type.value,
                    "update_config": request.update_config.model_dump()
                    if request.update_config
                    else {},
                }

                if update.type == UpdateType.ASSISTANT_MESSAGE:
                    data["content"] = update.message.content
                elif update.type == UpdateType.TOOL_COMPLETE:
                    data["tool"] = update.tool_name
                    data["result"] = serialize_for_json(update.result)
                elif update.type == UpdateType.ERROR:
                    data["error"] = update.error if hasattr(update, "error") else None

                await emit_update(request.update_config, data)
        except Exception as e:
            logger.error("Error in run_prompt", exc_info=True)
            await emit_update(
                request.update_config,
                {"type": "error", "error": str(e)},
            )

    background_tasks.add_task(run_prompt)
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
                dont_reply=request.dont_reply,
                model=request.model,
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
async def handle_deployment_configure(request: ConfigureDeploymentRequest):
    agent = Agent.load(request.agent_username)
    if not agent:
        raise APIError(f"Agent not found: {request.agent_username}", status_code=404)

    env = db.lower()

    # Update secrets if provided
    if request.secrets:
        secrets_dict = request.secrets.model_dump(exclude_none=True)
        secrets_dict["EDEN_AGENT_USERNAME"] = request.agent_username
        if secrets_dict:
            create_modal_secrets(
                secrets_dict,
                f"{request.agent_username}-secrets-{env}",
            )

    # Update agent config if provided
    if request.deployment_config:
        config_dict = request.deployment_config.model_dump(exclude_none=True)
        if config_dict:
            for key, value in config_dict.items():
                setattr(agent, key, value)
            agent.save()

    return {"message": "Deployment configuration updated"}


@handle_errors
async def handle_deployment_create(request: CreateDeploymentRequest):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent_username}", status_code=404)

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

    # Deploy the Modal container with optional repo branch
    # deploy_client(
    #     request.agent_username,
    #     request.platform.value,
    #     db.lower(),
    #     repo_branch=request.repo_branch,
    # )

    return {"message": f"Deployed {request.platform.value} client"}


@handle_errors
async def handle_deployment_delete(request: DeleteDeploymentRequest):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent}", status_code=404)

    try:
        # Stop the Modal container
        stop_client(request.agent_username, request.platform.value)

        # Delete deployment record
        Deployment.delete_deployment(agent.id, request.platform.value)

        return {"message": f"Stopped {request.platform.value} client"}
    except Exception as e:
        raise APIError(f"Failed to stop client: {str(e)}", status_code=500)


@handle_errors
async def handle_trigger_create(
    request: CreateTriggerRequest,
    scheduler: BackgroundScheduler,
):
    from eve.trigger import create_chat_trigger

    trigger_id = f"{request.user_id}_{request.agent_id}_{int(time.time())}"

    job = await create_chat_trigger(
        user_id=request.user_id,
        agent_id=request.agent_id,
        thread_id=request.thread_id,
        message=request.message,
        schedule=request.schedule.to_cron_dict(),
        update_config=request.update_config,
        scheduler=scheduler,
        trigger_id=trigger_id,
        handle_chat_fn=handle_chat,
    )

    trigger = Trigger(
        trigger_id=trigger_id,
        user=ObjectId(request.user_id),
        agent=ObjectId(request.agent_id),
        thread=ObjectId(request.thread_id),
        schedule=request.schedule.to_cron_dict(),
        message=request.message,
        update_config=request.update_config.model_dump()
        if request.update_config
        else {},
    )
    trigger.save()

    return {
        "id": str(trigger.id),
        "job_id": job.id,
        "next_run_time": str(job.next_run_time),
    }


@handle_errors
async def handle_trigger_delete(
    request: DeleteTriggerRequest, scheduler: BackgroundScheduler
):
    trigger = Trigger.from_mongo(request.id)
    scheduler.remove_job(trigger.trigger_id)
    trigger.delete()
    return {"message": f"Deleted job {trigger.trigger_id}"}


@handle_errors
async def handle_twitter_update(request: PlatformUpdateRequest):
    """Handle Twitter updates from async_prompt_thread"""

    print("REQUEST", request)
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
