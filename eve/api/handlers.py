import json
import logging
import os
import traceback
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from ably import AblyRealtime

from eve.api.requests import CancelRequest, ChatRequest, ScheduleRequest, TaskRequest
from eve.api.helpers import (
    emit_update,
    get_update_channel,
    serialize_for_json,
    setup_chat,
)
from eve.deploy import (
    DeployCommand,
    DeployRequest,
    create_modal_secrets,
    deploy_client,
    stop_client,
)
from eve.llm import UpdateType, async_prompt_thread
from eve.mongo import serialize_document
from eve.task import Task
from eve.tool import Tool

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()


async def handle_task(request: TaskRequest):
    tool = Tool.load(key=request.tool)
    result = await tool.async_start_task(
        requester_id=request.user_id, user_id=request.user_id, args=request.args
    )
    return serialize_document(result.model_dump(by_alias=True))


async def handle_cancel(request: CancelRequest):
    task = Task.from_mongo(request.task_id)
    assert str(task.user) == request.user_id, "Task user does not match user_id"
    if task.status in ["completed", "failed", "cancelled"]:
        return {"status": task.status}
    tool = Tool.load(key=task.tool)
    tool.cancel(task)
    return {"status": task.status}


async def handle_chat(
    request: ChatRequest, background_tasks: BackgroundTasks, ably_client: AblyRealtime
):
    try:
        user, agent, thread, tools = await setup_chat(request, background_tasks)
        update_channel = (
            await get_update_channel(request.update_config, ably_client)
            if request.update_config and request.update_config.sub_channel_name
            else None
        )

        async def run_prompt():
            async for update in async_prompt_thread(
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                model="claude-3-5-sonnet-20241022",
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

                await emit_update(request.update_config, update_channel, data)

        background_tasks.add_task(run_prompt)
        return {"status": "success", "thread_id": str(thread.id)}

    except Exception as e:
        logger.error(f"Error in handle_chat: {str(e)}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


async def handle_stream_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        user, agent, thread, tools = await setup_chat(request, background_tasks)

        async def event_generator():
            async for update in async_prompt_thread(
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                model="claude-3-5-sonnet-20241022",
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

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


async def handle_deploy(request: DeployRequest):
    try:
        if request.credentials:
            create_modal_secrets(
                request.credentials,
                f"{request.agent_key}-secrets-{db}",
            )

        if request.command == DeployCommand.DEPLOY:
            deploy_client(request.agent_key, request.platform.value)
            return {
                "status": "success",
                "message": f"Deployed {request.platform.value} client",
            }
        elif request.command == DeployCommand.STOP:
            stop_client(request.agent_key, request.platform.value)
            return {
                "status": "success",
                "message": f"Stopped {request.platform.value} client",
            }
        else:
            raise Exception("Invalid command")

    except Exception as e:
        return {"status": "error", "message": str(e)}


async def handle_schedule(request: ScheduleRequest):
    # TODO: Gene, translate natural language instruction into modal-compatible cron? Are we doing this?

    # Schedule the modal cron

    # If successful, save cron to db
    pass
