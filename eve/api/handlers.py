import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid

import aiohttp
import modal
from bson import ObjectId
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse

from eve.agent import Agent
from eve.agent.deployments.farcaster import FarcasterClient
from eve.agent.deployments.utils import get_api_url
from eve.agent.llm.llm import async_prompt
from eve.agent.memory.memory_models import messages_to_text, select_messages
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    ClientType,
    Deployment,
    DeploymentConfig,
    DeploymentSecrets,
    EmailDomain,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
    Notification,
    NotificationChannel,
    Reaction,
    Session,
    SessionUpdateConfig,
)
from eve.agent.session.service import (
    create_prompt_session_handle,
)
from eve.api.api_requests import (
    AgentToolsDeleteRequest,
    AgentToolsUpdateRequest,
    CancelRequest,
    CancelSessionRequest,
    CreateDeploymentRequestV2,
    CreateNotificationRequest,
    DeleteDeploymentRequestV2,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    GetDiscordChannelsRequest,
    PromptSessionRequest,
    ReactionRequest,
    RealtimeToolRequest,
    RefreshDiscordChannelsRequest,
    SessionCreationArgs,
    SyncDiscordChannelsRequest,
    TaskRequest,
    UpdateDeploymentRequestV2,
)
from eve.api.errors import APIError, handle_errors
from eve.api.helpers import (
    get_platform_client,
)
from eve.mongo import MongoDocumentNotFound, get_collection
from eve.task import Task
from eve.tool import Tool
from eve.tools.replicate_tool import replicate_update_task
from eve.user import User
from eve.utils import serialize_json

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()


def compute_llm_cost_simple(input_text: str, output_text: str) -> float:
    """
    Compute manna cost for LLM calls based on token count estimates.

    Args:
        input_text: The input text (system message + user messages)
        output_text: The output text from the LLM

    Returns:
        The manna cost
    """
    # Make sure output_text is a str (sometimes it's a dict)
    if isinstance(output_text, dict):
        output_text = json.dumps(output_text)

    # Estimate token count (4 chars ≈ 1 token)
    input_tokens = len(input_text) / 4
    output_tokens = len(output_text) / 4

    # Cost parameters (hardcoded to claude-sonnet-4-5)
    manna_cost_per_dollar = 1 / 0.01  # 0.01 dollars = 1 manna, so 100 manna per dollar
    input_cost_per_M_tokens = 3  # dollars per million tokens
    output_cost_per_M_tokens = 15  # dollars per million tokens

    # Calculate dollar cost
    input_cost_dollars = (input_tokens / 1_000_000) * input_cost_per_M_tokens
    output_cost_dollars = (output_tokens / 1_000_000) * output_cost_per_M_tokens
    total_cost_dollars = input_cost_dollars + output_cost_dollars

    # Convert to manna
    manna_cost = total_cost_dollars * manna_cost_per_dollar

    return manna_cost


@handle_errors
async def handle_create(request: TaskRequest):
    tool = Tool.load(key=request.tool)

    result = await tool.async_start_task(
        user_id=request.user_id,
        agent_id=None,
        args=request.args,
        public=request.public,
        metadata=request.metadata,
    )

    if request.metadata:
        result.update(metadata=request.metadata)

    return serialize_json(result.model_dump(by_alias=True))


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


@handle_errors
async def handle_realtime_tool(
    request: RealtimeToolRequest, background_tasks: BackgroundTasks
):
    """
    Handle realtime tool calls from ElevenLabs client tools.

    Supported tools:
    - "create": Run the create tool with default args (blocking)
    - "display": Run the display tool (blocking)
    - "create_async": Run create tool in background, return immediately

    For blocking tools: waits for completion and returns result.
    For async tools: returns immediately with task_id for tracking.
    """
    # Log full request for debugging
    logger.info(
        f"[REALTIME_TOOL] Received request: tool={request.tool_name}, "
        f"wait={request.wait_for_response}, session={request.session_id}, "
        f"args={json.dumps(request.args, default=str)}"
    )

    # Get agent_id from session
    agent_id = None
    if request.session_id:
        try:
            session = Session.from_mongo(ObjectId(request.session_id))
            if session and session.agents:
                agent_id = str(session.agents[0])
                logger.info(f"[REALTIME_TOOL] Found agent_id from session: {agent_id}")
        except Exception as e:
            logger.warning(f"[REALTIME_TOOL] Could not load session: {e}")

    # Map tool names to actual tool keys
    tool_key_map = {
        "create": "create",
        "create_async": "create",
        "display": "display",
        "eden_search": "eden_search",
    }

    actual_tool_key = tool_key_map.get(request.tool_name)
    if not actual_tool_key:
        raise APIError(f"Unknown tool: {request.tool_name}", status_code=400)

    # Load the tool
    tool = Tool.load(key=actual_tool_key)

    # Prepare args - for create tool, use defaults if prompt not provided
    args = request.args.copy()
    if actual_tool_key == "create" and "prompt" not in args:
        args["prompt"] = "A beautiful abstract image"
        args["output"] = args.get("output", "image")

    # Add context IDs
    args["user_id"] = request.user_id
    args["agent_id"] = agent_id
    args["session_id"] = request.session_id

    # Log prepared args
    logger.info(
        f"[REALTIME_TOOL] Prepared args for {actual_tool_key}: {json.dumps(args, default=str)}"
    )

    # Determine if this is an async (fire-and-forget) request
    is_async = request.tool_name == "create_async" or not request.wait_for_response

    # Generate a task_id for tracking (frontend expects this)
    task_id = str(ObjectId())

    if is_async:
        # Non-blocking: start task and return immediately
        async def run_tool_background():
            try:
                result = await tool.async_run(args)
                logger.info(
                    f"[REALTIME_TOOL] Background task {request.tool_name} completed: {json.dumps(result, default=str)}"
                )
            except Exception as e:
                logger.error(
                    f"[REALTIME_TOOL] Background task failed: {e}", exc_info=True
                )

        background_tasks.add_task(run_tool_background)

        logger.info(f"[REALTIME_TOOL] Tool {request.tool_name} started in background")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "I've started working on that. I'll let you know when it's ready.",
        }
    else:
        # Blocking: run and wait for result
        logger.info(f"[REALTIME_TOOL] Running tool {request.tool_name} (blocking)")

        try:
            result = await tool.async_run(args)

            # Log full result for debugging
            logger.info(
                f"[REALTIME_TOOL] Tool {request.tool_name} raw result: {json.dumps(result, default=str)}"
            )

            if result.get("status") == "failed":
                error_msg = result.get("error", "Unknown error")
                logger.error(
                    f"[REALTIME_TOOL] Tool {request.tool_name} failed: {error_msg}. Args: {json.dumps(args, default=str)}"
                )
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error_msg,
                    "args": args,  # Include args in error response for debugging
                }

            logger.info(
                f"[REALTIME_TOOL] Tool {request.tool_name} completed successfully"
            )

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
            }

        except Exception as e:
            logger.error(
                f"[REALTIME_TOOL] Tool execution failed: {e}. Args: {json.dumps(args, default=str)}",
                exc_info=True,
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "args": args,  # Include args in error response for debugging
            }


async def handle_replicate_webhook(body: dict):
    task = Task.from_handler_id(body["id"])
    tool = Tool.load(task.tool)
    _ = replicate_update_task(
        task, body["status"], body["error"], body["output"], tool.output_handler
    )
    return {"status": "success"}


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


@handle_errors
async def handle_prompt_session(
    request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    """
    Prompt a session using the unified orchestrator.

    This handler uses the new orchestrate() function which provides
    full observability (Sentry, Langfuse, structured logging).
    """
    from eve.agent.session.orchestrator import (
        OrchestrationMode,
        OrchestrationRequest,
        orchestrate,
    )
    from eve.agent.session.setup import setup_session

    # Setup session first to get session_id
    session = setup_session(
        background_tasks=background_tasks,
        session_id=request.session_id,
        user_id=request.user_id,
        request=request,
    )
    session_id = str(session.id)

    # For automatic sessions that are actively running, just add the message
    # without running orchestration - the moderator will see it on its next turn
    if session.session_type == "automatic" and session.status in ("running", "active"):
        logger.info(
            f"[PROMPT] Automatic session {session_id} is {session.status}, "
            "adding message only (no orchestration)"
        )
        # Use the message-only handler
        return await handle_session_message(request, background_tasks)

    # Build orchestration request
    orch_request = OrchestrationRequest(
        initiating_user_id=request.user_id,
        session=session,
        actor_agent_ids=request.actor_agent_ids,
        message=request.message,
        llm_config=request.llm_config,
        update_config=request.update_config,
        notification_config=request.notification_config,
        thinking_override=request.thinking,
        acting_user_id=request.acting_user_id,
        api_key_id=request.api_key_id,
        trigger_id=request.trigger,
        mode=OrchestrationMode.API_REQUEST,
        stream=request.stream,
        background_tasks=background_tasks,
    )

    if request.stream:

        async def event_generator():
            try:
                from eve.api.sse_manager import sse_manager
                from eve.utils import dumps_json

                async for data in orchestrate(orch_request):
                    # Broadcast to SSE manager for connected clients
                    try:
                        await sse_manager.broadcast(session_id, data)
                    except Exception as sse_error:
                        logger.error(f"Failed to broadcast to SSE: {sse_error}")
                    yield f"data: {dumps_json({'event': 'update', 'data': data})}\n\n"
                yield f"data: {dumps_json({'event': 'done', 'data': ''})}\n\n"
            except Exception as e:
                logger.error("Error in event_generator", exc_info=True)
                from eve.utils import dumps_json

                yield f"data: {dumps_json({'event': 'error', 'data': {'error': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run in background
    async def run_orchestration():
        from eve.api.helpers import emit_update

        async for data in orchestrate(orch_request):
            await emit_update(request.update_config, data, session_id=session_id)

    background_tasks.add_task(run_orchestration)

    return {"session_id": session_id}


@handle_errors
async def handle_session_message(
    request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    """Add a message to a session without running orchestration.

    Same interface as /sessions/prompt, but only adds the message and returns.
    Creates a new session if session_id is not provided.
    """
    handle = create_prompt_session_handle(request, background_tasks)
    session_id = handle.session_id

    # Add user message only (no orchestration)
    await handle.add_message()

    return {"session_id": session_id, "message": "Message added successfully"}


@handle_errors
async def handle_session_run(
    request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    """Run orchestration on a session without adding a message.

    Same interface as /sessions/prompt, but only runs orchestration.
    Requires session_id to be provided.

    Uses the unified orchestrator for full observability.
    """
    from eve.agent.session.orchestrator import (
        OrchestrationMode,
        OrchestrationRequest,
        orchestrate,
    )
    from eve.agent.session.setup import setup_session

    if not request.session_id:
        raise APIError("session_id is required for /sessions/run", status_code=400)

    # Load existing session
    session = setup_session(
        background_tasks=background_tasks,
        session_id=request.session_id,
        user_id=request.user_id,
        request=request,
    )
    session_id = str(session.id)

    # Build orchestration request WITHOUT a message
    orch_request = OrchestrationRequest(
        initiating_user_id=request.user_id,
        session=session,
        actor_agent_ids=request.actor_agent_ids,
        message=None,  # No message - just run orchestration
        llm_config=request.llm_config,
        update_config=request.update_config,
        notification_config=request.notification_config,
        thinking_override=request.thinking,
        acting_user_id=request.acting_user_id,
        api_key_id=request.api_key_id,
        mode=OrchestrationMode.API_REQUEST,
        stream=request.stream,
        background_tasks=background_tasks,
    )

    if request.stream:

        async def event_generator():
            try:
                from eve.api.sse_manager import sse_manager
                from eve.utils import dumps_json

                async for data in orchestrate(orch_request):
                    try:
                        await sse_manager.broadcast(session_id, data)
                    except Exception as sse_error:
                        logger.error(f"Failed to broadcast to SSE: {sse_error}")
                    yield f"data: {dumps_json({'event': 'update', 'data': data})}\n\n"
                yield f"data: {dumps_json({'event': 'done', 'data': ''})}\n\n"
            except Exception as e:
                logger.error("Error in event_generator", exc_info=True)
                from eve.utils import dumps_json

                yield f"data: {dumps_json({'event': 'error', 'data': {'error': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run in background
    async def run_orchestration():
        from eve.api.helpers import emit_update

        async for data in orchestrate(orch_request):
            await emit_update(request.update_config, data, session_id=session_id)

    background_tasks.add_task(run_orchestration)

    return {"session_id": session_id}


@handle_errors
async def handle_session_stream(session_id: str):
    """Stream SSE updates for a session."""
    import uuid

    from fastapi.responses import StreamingResponse

    from eve.api.sse_manager import sse_manager

    # Verify session exists
    try:
        session = Session.from_mongo(ObjectId(session_id))
        if not session:
            raise APIError(f"Session not found: {session_id}", status_code=404)
    except Exception:
        raise APIError(f"Invalid session_id: {session_id}", status_code=400)

    # Generate unique client ID for this connection
    client_id = str(uuid.uuid4())

    async def event_generator():
        # Register connection
        connection = await sse_manager.connect(session_id, client_id)

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'event': 'connected', 'session_id': session_id, 'client_id': client_id})}\n\n"

            # Stream updates from queue
            while True:
                try:
                    # Wait for message with timeout for keep-alive
                    message = await asyncio.wait_for(
                        connection.queue.get(),
                        timeout=30.0,  # 30 second timeout
                    )
                    yield f"data: {message}\n\n"

                except asyncio.TimeoutError:
                    # Send keep-alive ping
                    yield ": keep-alive\n\n"
                    continue

        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error in SSE stream for session {session_id}: {e}")
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
        finally:
            # Clean up connection
            await sse_manager.disconnect(session_id, connection)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


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

        cancel_message = {
            "session_id": request.session_id,
            "user_id": request.user_id,
            "timestamp": time.time(),
        }

        # Include trace_id if provided for trace-specific cancellation
        if request.trace_id:
            cancel_message["trace_id"] = request.trace_id

        # Include tool call specific cancellation
        if request.tool_call_id:
            cancel_message["tool_call_id"] = request.tool_call_id
        if request.tool_call_index is not None:
            cancel_message["tool_call_index"] = request.tool_call_index

        await channel.publish("cancel", cancel_message)

        logger.info(
            f"Sent cancellation signal for session {request.session_id}"
            + (f" trace {request.trace_id}" if request.trace_id else "")
        )
        return {
            "status": "cancel_signal_sent",
            "session_id": request.session_id,
            "trace_id": request.trace_id,
        }

    except Exception as e:
        logger.error(f"Error sending session cancel signal: {e}", exc_info=True)
        raise APIError(f"Failed to send cancel signal: {str(e)}", status_code=500)


@handle_errors
async def handle_session_fields_update(request):
    """Update session fields like context, title, etc.

    This endpoint properly handles empty strings - an empty string will clear the field,
    while None (omitted field) will leave it unchanged.
    """
    from bson import ObjectId

    logger.info(f"[SESSION_UPDATE] Updating session {request.session_id}")

    try:
        session = Session.from_mongo(ObjectId(request.session_id))

        # Build update dict - only include fields that were explicitly provided
        # Use model_dump to get all fields, then filter based on what was set
        request_data = request.model_dump(exclude={"session_id"}, exclude_unset=True)

        # IMPORTANT: For fields that can be empty strings, we need special handling
        # model_dump(exclude_unset=True) will include fields set to "" or None if explicitly provided
        if request_data:
            logger.info(
                f"[SESSION_UPDATE] Updating fields: {list(request_data.keys())}"
            )
            logger.info(f"[SESSION_UPDATE] Values: {request_data}")
            session.update(**request_data)
            logger.info(f"[SESSION_UPDATE] Successfully updated session {session.id}")
        else:
            logger.info("[SESSION_UPDATE] No fields to update")

        return {"success": True, "session_id": str(session.id)}
    except Exception as e:
        logger.error(f"[SESSION_UPDATE] Error updating session: {e}")
        raise


@handle_errors
async def handle_session_status_update(request):
    """Update the status of a session.

    For automatic sessions, setting status to 'active' will start the automatic
    session loop (runs orchestration, then schedules next run after delay).
    """
    logger.info(
        f"[STATUS] Session status update: session={request.session_id}, new_status={request.status}"
    )

    try:
        session = Session.from_mongo(ObjectId(request.session_id))
        old_status = session.status
        new_status = request.status

        logger.info(
            f"[STATUS] Session {session.id}: type={session.session_type}, {old_status} -> {new_status}"
        )

        # Reject attempts to reactivate a finished session
        if old_status == "finished" and new_status == "active":
            logger.warning(f"[STATUS] Cannot reactivate finished session {session.id}")
            return {
                "success": False,
                "error": "Cannot reactivate a finished session. Create a new session instead.",
            }

        # Update the status
        session.update(status=new_status)

        # Check if we need to start an automatic session
        should_start_automatic = (
            session.session_type == "automatic"
            and new_status == "active"
            and old_status != "active"  # Only if transitioning to active
        )

        if should_start_automatic:
            # Check if running on Modal or locally
            if os.getenv("MODAL_SERVE") == "1":
                # Production: spawn Modal function
                try:
                    db = os.getenv("DB", "STAGE").upper()
                    func = modal.Function.from_name(
                        f"api-{db.lower()}",
                        "handle_session_status_change_fn",
                        environment_name="main",
                    )
                    func.spawn(request.session_id, request.status)
                except Exception as e:
                    logger.warning(
                        f"Failed to spawn Modal function for automatic session: {e}"
                    )
                    session.update(status="paused")
                    return {"success": False, "error": str(e)}
            else:
                # Local development: start automatic session in-process
                import asyncio

                from eve.agent.session.automatic import start_automatic_session

                asyncio.create_task(start_automatic_session(request.session_id))

        return {
            "success": True,
            "session_id": request.session_id,
            "status": new_status,
            "automatic_session_started": should_start_automatic,
        }

    except APIError:
        raise

    except Exception as e:
        logger.error(f"Error updating session status: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@handle_errors
async def handle_v2_deployment_create(request: CreateDeploymentRequestV2):
    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {agent.id}", status_code=404)

    # Create deployment object for client
    deployment = Deployment(
        agent=agent.id,
        user=ObjectId(request.user),
        platform=request.platform,
        secrets=request.secrets,
        config=request.config,
    )

    # Get platform client and run predeploy
    client = get_platform_client(
        agent=agent, platform=request.platform, deployment=deployment
    )
    secrets, config = await client.predeploy(
        secrets=request.secrets, config=request.config
    )

    # Update deployment with validated secrets and config
    deployment.secrets = secrets
    deployment.config = config
    deployment.valid = True
    deployment.save(
        upsert_filter={"agent": agent.id, "platform": request.platform.value}
    )

    try:
        # Run postdeploy
        await client.postdeploy()
    except Exception as e:
        logger.error(f"Failed in postdeploy: {str(e)}")
        deployment.delete()
        raise APIError(f"Failed to deploy client: {str(e)}", status_code=500)

    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_v2_deployment_update(request: UpdateDeploymentRequestV2):
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.deployment_id}", status_code=404
        )

    # Store old config and secrets for platform update hook
    # MongoDB returns nested objects as dicts, convert to proper Pydantic models
    old_config = (
        DeploymentConfig(**deployment.config)
        if isinstance(deployment.config, dict)
        else deployment.config
    )
    old_secrets = (
        DeploymentSecrets(**deployment.secrets)
        if isinstance(deployment.secrets, dict)
        else deployment.secrets
    )

    update_dict = {}

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

        update_dict["config"] = updated_config_dict

    # Handle secrets updates by merging with existing secrets
    if request.secrets:
        existing_secrets = deployment.secrets or DeploymentSecrets()
        new_secrets = request.secrets.model_dump(exclude_unset=True)

        # Merge the secrets at the platform level
        updated_secrets_dict = existing_secrets.model_dump() if existing_secrets else {}

        for platform, platform_secrets in new_secrets.items():
            if platform_secrets is not None:
                if platform in updated_secrets_dict:
                    # Merge platform-specific secrets
                    updated_secrets_dict[platform].update(platform_secrets)
                else:
                    # Add new platform secrets
                    updated_secrets_dict[platform] = platform_secrets

        update_dict["secrets"] = updated_secrets_dict

    # Update deployment with both config and secrets if provided
    if update_dict:
        # Apply updates to deployment object and use save() to ensure
        # secrets are encrypted via convert_to_mongo() hook
        # (deployment.update() bypasses encryption)
        if "config" in update_dict:
            deployment.config = DeploymentConfig(**update_dict["config"])
        if "secrets" in update_dict:
            deployment.secrets = DeploymentSecrets(**update_dict["secrets"])

        deployment.save()

        # Call platform-specific update hook if it exists
        try:
            agent = Agent.from_mongo(ObjectId(deployment.agent))
            client = get_platform_client(
                agent=agent, platform=deployment.platform, deployment=deployment
            )

            # Call platform-specific update hook
            await client.update(
                old_config=old_config,
                new_config=deployment.config,
                old_secrets=old_secrets,
                new_secrets=deployment.secrets,
            )
        except Exception as e:
            logger.error(f"Error calling platform update hook: {e}")
            # Don't fail the update if the hook fails

    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_v2_deployment_delete(request: DeleteDeploymentRequestV2):
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.deployment_id}",
            status_code=404,
        )

    try:
        # Get platform client and run stop
        agent = Agent.from_mongo(ObjectId(deployment.agent))
        client = get_platform_client(
            agent=agent, platform=deployment.platform, deployment=deployment
        )
        await client.stop()
        deployment.delete()

        return {"success": True}
    except Exception as e:
        raise APIError(f"Failed to stop client: {str(e)}", status_code=500)


@handle_errors
async def handle_v2_deployment_interact(request: DeploymentInteractRequest):
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.deployment_id}", status_code=404
        )
    agent = Agent.from_mongo(ObjectId(deployment.agent))
    if not agent:
        raise APIError(f"Agent not found: {deployment.agent}", status_code=404)
    client = get_platform_client(
        agent=agent, platform=deployment.platform, deployment=deployment
    )
    await client.interact(request)
    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_v2_deployment_farcaster_neynar_webhook(request: Request):
    client = FarcasterClient()
    await client.handle_neynar_webhook(request)


@handle_errors
async def handle_v2_deployment_emission(request: DeploymentEmissionRequest):
    deployment = Deployment.from_mongo(ObjectId(request.update_config.deployment_id))
    if not deployment:
        raise APIError(
            f"Deployment not found: {request.update_config.deployment_id}",
            status_code=404,
        )
    agent = Agent.from_mongo(ObjectId(deployment.agent))
    if not agent:
        raise APIError(f"Agent not found: {deployment.agent}", status_code=404)
    client = get_platform_client(
        agent=agent, platform=deployment.platform, deployment=deployment
    )
    await client.handle_emission(request)
    return {"deployment_id": str(deployment.id)}


@handle_errors
async def handle_v2_deployment_email_inbound(request: Request):
    form = await request.form()

    timestamp = form.get("timestamp")
    token = form.get("token")
    signature = form.get("signature")

    signing_key = os.getenv("MAILGUN_WEBHOOK_SIGNING_KEY") or os.getenv(
        "MAILGUN_API_KEY"
    )

    if signing_key and timestamp and token and signature:
        expected_signature = hmac.new(
            signing_key.encode(), f"{timestamp}{token}".encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_signature, signature):
            logger.warning("Invalid email webhook signature")
            return JSONResponse(status_code=401, content={"error": "invalid signature"})
    else:
        logger.warning("Missing email webhook signature data")
        return JSONResponse(
            status_code=400, content={"error": "missing signature parameters"}
        )

    recipient = form.get("recipient")
    sender = form.get("sender")
    subject = form.get("subject") or ""

    if not recipient or not sender:
        return JSONResponse(status_code=400, content={"error": "invalid payload"})

    domain_part = recipient.split("@")[-1].lower()
    email_domain = EmailDomain.find_by_domain(domain_part)

    if not email_domain:
        logger.info(f"No email domain configured for {domain_part}")
        return JSONResponse(status_code=200, content={"ok": True})

    if email_domain.provider != "mailgun":
        logger.info(f"Email domain {domain_part} not managed by configured provider")
        return JSONResponse(status_code=200, content={"ok": True})

    deployment = None
    if email_domain.deployment:
        deployment = Deployment.from_mongo(ObjectId(email_domain.deployment))
    else:
        try:
            deployment = Deployment.load(agent=email_domain.agent, platform="email")
        except Exception:
            deployment = None

    if not deployment:
        logger.info(f"No deployment associated with domain {domain_part}")
        return JSONResponse(status_code=200, content={"ok": True})

    config_email = deployment.config.email if deployment.config else None

    if not config_email or not config_email.autoreply_enabled:
        logger.info(f"Autoreply disabled for deployment {deployment.id}")
        return JSONResponse(status_code=200, content={"ok": True})

    text_body = form.get("stripped-text") or form.get("body-plain") or ""
    html_body = form.get("stripped-html") or form.get("body-html")
    message_id = (
        form.get("Message-Id") or form.get("message-id") or form.get("message_id")
    )
    in_reply_to = (
        form.get("In-Reply-To") or form.get("in-reply-to") or form.get("in_reply_to")
    )

    thread_identifier = (
        (in_reply_to or message_id or str(uuid.uuid4()))
        .replace("<", "")
        .replace(">", "")
    )

    session_key = f"email-{deployment.id}-{thread_identifier}"

    sender_name = sender
    if "<" in sender and ">" in sender:
        sender_name = sender.split("<")[0].strip().strip('"')

    try:
        session = Session.load(session_key=session_key)
        session_id = str(session.id)
    except MongoDocumentNotFound:
        session = None
        session_id = None

    user = User.from_email(sender)

    content_lines = []
    if subject:
        content_lines.append(f"Subject: {subject}")
    if text_body:
        content_lines.extend(["", text_body])
    elif html_body:
        content_lines.extend(["", html_body])
    else:
        content_lines.append("")

    message_content = "\n".join(content_lines).strip()

    avg_delay = config_email.reply_delay_average_minutes or 0
    variance_delay = config_email.reply_delay_variance_minutes or 0

    delay_minutes = avg_delay
    if variance_delay:
        delay_minutes = max(0, random.normalvariate(avg_delay, variance_delay))

    delay_seconds = max(0, delay_minutes * 60)

    update_config = SessionUpdateConfig(
        deployment_id=str(deployment.id),
        update_endpoint=f"{get_api_url()}/v2/deployments/emission",
        email_sender=sender,
        email_recipient=recipient,
        email_subject=subject,
        email_message_id=message_id,
        email_thread_id=thread_identifier,
    )

    prompt_request = PromptSessionRequest(
        user_id=str(user.id),
        actor_agent_ids=[str(deployment.agent)],
        message=ChatMessageRequestInput(
            content=message_content or "(no content)",
            sender_name=sender_name,
        ),
        update_config=update_config,
    )

    if session_id:
        prompt_request.session_id = session_id
    else:
        prompt_request.creation_args = SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(deployment.agent)],
            title=f"Email thread with {sender_name}",
            session_key=session_key,
            platform="email",
        )

    async def dispatch_prompt():
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        eden_api_url = get_api_url()
        prompt_url = f"{eden_api_url}/sessions/prompt"

        async with aiohttp.ClientSession() as session_client:
            async with session_client.post(
                prompt_url,
                json=prompt_request.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to dispatch email prompt ({response.status}): {error_text}"
                    )

    asyncio.create_task(dispatch_prompt())

    return JSONResponse(status_code=200, content={"ok": True})


# Notification handlers
@handle_errors
async def handle_create_notification(request: CreateNotificationRequest):
    """Create a new notification"""

    # Validate user exists
    user = User.from_mongo(ObjectId(request.user_id))
    if not user:
        raise APIError(f"User not found: {request.user_id}", status_code=404)

    # Set default channels if not provided
    channels = request.channels or [NotificationChannel.IN_APP]

    # Create notification
    notification = Notification(
        user=ObjectId(request.user_id),
        type=request.type,
        title=request.title,
        message=request.message,
        priority=request.priority,
        channels=channels,
        trigger=ObjectId(request.trigger_id) if request.trigger_id else None,
        session=ObjectId(request.session_id) if request.session_id else None,
        agent=ObjectId(request.agent_id) if request.agent_id else None,
        metadata=request.metadata,
        action_url=request.action_url,
        expires_at=request.expires_at,
    )

    notification.save()

    # Mark as delivered for in-app channel immediately
    if NotificationChannel.IN_APP in channels:
        notification.mark_delivered(NotificationChannel.IN_APP)

    return {"id": str(notification.id), "message": "Notification created successfully"}


import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only load model on Modal, not on localhost
# MODAL_IS_REMOTE is automatically set by Modal when running remotely
if os.getenv("MODAL_IS_REMOTE") == "1":
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
else:
    model = None
    proc = None


# Embed handler
@handle_errors
async def handle_embed(request):
    """Embed images with CLIP"""

    inputs = proc(
        text=[request.query], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    v = model.get_text_features(**inputs)
    qv = F.normalize(v, p=2, dim=-1)[0].cpu().tolist()

    return {"embedding": qv}


# Embed search handler
@handle_errors
async def handle_embedsearch(request):
    """Search images using CLIP embeddings"""

    qv_result = await handle_embed(request)
    qv = qv_result["embedding"]

    creations = get_collection("creations3")
    filt = {}

    if request.agent_id:
        filt["agent"] = ObjectId(request.agent_id)
    if request.user_id:
        filt["user"] = ObjectId(request.user_id)
    if request.tool:
        filt["tool"] = ObjectId(request.tool)

    base_nc = max(50 * request.limit, 10_000)
    exact = False
    if filt:
        subset_cap = 10_000
        n = creations.count_documents(filt, limit=subset_cap + 1)
        exact = n <= subset_cap

    pipeline = [
        {
            "$vectorSearch": {
                "index": "img_vec_idx",
                "path": "embedding",
                "queryVector": qv,
                # "numCandidates": 50 * request.limit, # ~20x limit is recommended starting point
                "limit": int(request.limit),
                **({"filter": filt} if filt else {}),
            }
        },
        {
            "$project": {
                "_id": 1,
                "filename": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    if exact:
        # ENN: omit numCandidates
        pipeline[0]["$vectorSearch"]["exact"] = True
    else:
        # ANN: start at ~20x limit
        pipeline[0]["$vectorSearch"]["numCandidates"] = base_nc

    results = list(creations.aggregate(pipeline))

    return {"results": results}


@handle_errors
async def handle_extract_agent_prompts(request):
    """
    Extract agent persona, description, and memory instructions from a conversation session.

    Takes a session_id, fetches all messages from that session, and uses LLM to extract:
    - agent_instructions: Core personality traits and characteristics
    - agent_description: Short summary of agent's purpose and capabilities
    - memory_instructions: Instructions for how agent should store/recall memories
    """
    from pydantic import BaseModel, Field

    from eve.agent.session.models import Session

    class AgentPromptsResponse(BaseModel):
        agent_instructions: str = Field(
            description="Core personality traits and characteristics"
        )
        agent_description: str = Field(description="Short summary of agent's purpose")
        memory_instructions: str = Field(
            description="Instructions for memory extraction"
        )

    # Get user_id and verify user exists
    user_id = request.user_id
    user = User.from_mongo(ObjectId(user_id))
    if not user:
        raise APIError(f"User not found: {user_id}", status_code=404)

    # Fetch session
    session = Session.from_mongo(ObjectId(request.session_id))
    if not session:
        raise APIError(f"Session not found: {request.session_id}", status_code=404)

    session_messages = select_messages(session, selection_limit=50)
    conversation_text, _ = messages_to_text(session_messages)

    # Get agent name from request, default to "<agent-name>" if not provided
    agent_name = request.agent_name or "<agent-name>"

    # Load prompt template
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "prompt_templates", "extract_agent_prompts.txt"
    )
    with open(template_path, "r") as f:
        prompt_template = f.read()

    # Inject conversation and agent name into template
    prompt = prompt_template.replace("{conversation}", conversation_text)
    prompt = prompt.replace("{agent_name}", agent_name)

    # Make single LLM call with structured output using LLMContext with tracing
    context = LLMContext(
        messages=[ChatMessage(role="user", content=prompt)],
        config=LLMConfig(
            model="gpt-5.1",
            response_format=AgentPromptsResponse,
        ),
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(session.id)}",
            trace_name="extract_agent_prompts",
            trace_id=str(uuid.uuid4()),
            generation_name="extract_agent_prompts",
            trace_metadata=LLMTraceMetadata(
                session_id=str(session.id),
                user_id=str(user_id),
                agent_id=str(session.agents[0]) if session.agents else None,
            ),
        ),
        enable_tracing=True,
    )

    response = await async_prompt(context)

    # Parse the structured response
    if hasattr(response, "parsed"):
        result = response.parsed
    else:
        result = AgentPromptsResponse.model_validate_json(response.content)

    # Extract fields from response model
    agent_instructions = result.agent_instructions
    agent_description = result.agent_description
    memory_instructions = result.memory_instructions

    # Calculate cost (estimate based on prompt length and typical response)
    # Since we're using response_model, we get a structured object back, not text
    content_text = json.dumps(result.model_dump())
    cost = compute_llm_cost_simple(prompt, content_text)

    # Check and spend manna
    try:
        user.check_manna(cost)
    except Exception as e:
        raise APIError(f"Insufficient manna: {str(e)}", status_code=402)

    return {
        "agent_instructions": agent_instructions.strip(),
        "agent_description": agent_description.strip(),
        "memory_instructions": memory_instructions.strip(),
        "cost": cost,
    }


@handle_errors
async def handle_regenerate_agent_memory(request):
    """Regenerate fully_formed_memory for an agent memory shard."""
    from eve.agent.memory.memory import _regenerate_fully_formed_agent_memory
    from eve.agent.memory.memory_models import AgentMemory

    try:
        shard_id = ObjectId(request.shard_id)
        logger.info(f"Regenerating agent memory for shard {shard_id}")

        # Load shard
        shard = AgentMemory.from_mongo(shard_id)
        if not shard:
            logger.error(f"Shard not found: {shard_id}")
            raise APIError("Shard not found", status_code=404)

        # Regenerate
        await _regenerate_fully_formed_agent_memory(shard)

        return {"success": True, "shard_id": str(shard_id)}
    except APIError:
        raise
    except Exception as e:
        logger.error(
            f"Error regenerating agent memory for shard {request.shard_id}: {e}"
        )
        return {"success": False, "error": str(e)}


@handle_errors
async def handle_regenerate_user_memory(request):
    """Regenerate fully_formed_memory for a user memory document."""
    from eve.agent.memory.memory import _regenerate_fully_formed_user_memory
    from eve.agent.memory.memory_models import UserMemory

    try:
        agent_id = ObjectId(request.agent_id)
        user_id = ObjectId(request.user_id)
        logger.info(f"Regenerating user memory for agent {agent_id}, user {user_id}")

        # Load user memory
        user_memory = UserMemory.find_one({"agent_id": agent_id, "user_id": user_id})
        if not user_memory:
            logger.error(f"User memory not found for agent {agent_id}, user {user_id}")
            raise APIError("User memory not found", status_code=404)

        # Regenerate
        await _regenerate_fully_formed_user_memory(user_memory)

        return {"success": True, "user_id": str(user_id)}
    except APIError:
        raise
    except Exception as e:
        logger.error(
            f"Error regenerating user memory for agent {request.agent_id}, user {request.user_id}: {e}"
        )
        return {"success": False, "error": str(e)}


@handle_errors
async def handle_reaction(request: ReactionRequest):
    """
    Handle reactions to messages or tool calls.

    If the reacted tool has a hook.py file with a `hook` function,
    it will be called with the reaction context.
    """
    import importlib.util

    from eve.tool import get_api_files

    logger.info(
        f"[handle_reaction] Received reaction request:\n"
        f"  message_id: {request.message_id}\n"
        f"  tool_call_id: {request.tool_call_id}\n"
        f"  reaction: {request.reaction}\n"
        f"  user_id: {request.user_id}"
    )

    # Load the message
    message = ChatMessage.from_mongo(ObjectId(request.message_id))
    if not message:
        raise APIError(f"Message not found: {request.message_id}", status_code=404)

    tool_name = None

    # If reacting to a specific tool call, find it
    if request.tool_call_id:
        tool_call = None
        for tc in message.tool_calls or []:
            if tc.id == request.tool_call_id:
                tool_call = tc
                break

        if not tool_call:
            raise APIError(
                f"Tool call not found: {request.tool_call_id}", status_code=404
            )

        tool_name = tool_call.tool

        # Add reaction to the tool call
        if not tool_call.reactions:
            tool_call.reactions = []
        user_key = request.user_id or "anonymous"
        # Check if this user already has this reaction
        existing = any(
            r.user_id == user_key and r.reaction == request.reaction
            for r in tool_call.reactions
        )
        if not existing:
            tool_call.reactions.append(
                Reaction(user_id=user_key, reaction=request.reaction)
            )

        message.save()

        # Try to call the tool's hook if it exists
        if tool_name:
            api_files = get_api_files()
            if tool_name in api_files:
                tool_dir = os.path.dirname(api_files[tool_name])
                hook_path = os.path.join(tool_dir, "hook.py")

                if os.path.exists(hook_path):
                    try:
                        # Dynamically load the hook module
                        spec = importlib.util.spec_from_file_location(
                            f"{tool_name}_hook", hook_path
                        )
                        hook_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(hook_module)

                        # Call the hook function if it exists
                        if hasattr(hook_module, "hook"):
                            hook_result = hook_module.hook(
                                message_id=request.message_id,
                                tool_call_id=request.tool_call_id,
                                reaction=request.reaction,
                                user_id=request.user_id,
                            )
                            # Support async hooks
                            if asyncio.iscoroutine(hook_result):
                                await hook_result

                            logger.info(
                                f"Hook executed for tool '{tool_name}' on reaction '{request.reaction}'"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error executing hook for tool '{tool_name}': {e}"
                        )

        return {
            "status": "success",
            "message_id": request.message_id,
            "tool_call_id": request.tool_call_id,
            "reactions": tool_call.reactions,
        }
    else:
        # Add reaction to the message itself
        if not message.reactions:
            message.reactions = []
        user_key = request.user_id or "anonymous"
        # Check if this user already has this reaction
        existing = any(
            r.user_id == user_key and r.reaction == request.reaction
            for r in message.reactions
        )
        if not existing:
            message.reactions.append(
                Reaction(user_id=user_key, reaction=request.reaction)
            )

        message.save()

        return {
            "status": "success",
            "message_id": request.message_id,
            "reactions": message.reactions,
        }


@handle_errors
async def handle_get_discord_channels(request: GetDiscordChannelsRequest):
    """Get all guilds and channels for a Discord deployment."""
    import logging

    logger = logging.getLogger(__name__)
    from eve.agent.deployments.discord_gateway import DiscordGuild

    logger.info(
        f"[handle_get_discord_channels] Request: deployment_id={request.deployment_id}, user_id={request.user_id}"
    )

    # Load deployment
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        logger.warning(
            f"[handle_get_discord_channels] Deployment not found: {request.deployment_id}"
        )
        raise APIError("Deployment not found", status_code=404)

    logger.info(
        f"[handle_get_discord_channels] Deployment found: user={deployment.user}, platform={deployment.platform}"
    )

    # Verify ownership - skip for admin calls (user_id from trusted backend)
    # Note: The Eden API passes the requesting user's ID
    # if str(deployment.user) != request.user_id:
    #     logger.warning(f"[handle_get_discord_channels] Unauthorized: deployment.user={deployment.user}, request.user_id={request.user_id}")
    #     raise APIError("Unauthorized", status_code=403)

    # Verify it's a Discord deployment
    if deployment.platform != ClientType.DISCORD:
        logger.warning(
            f"[handle_get_discord_channels] Not a Discord deployment: {deployment.platform}"
        )
        raise APIError("Not a Discord deployment", status_code=400)

    # Query guilds from MongoDB
    logger.info(
        f"[handle_get_discord_channels] Querying guilds for deployment_id={request.deployment_id}"
    )
    guilds = list(DiscordGuild.find({"deployment_id": ObjectId(request.deployment_id)}))
    logger.info(f"[handle_get_discord_channels] Found {len(guilds)} guilds")

    # Get latest refresh time
    last_refreshed = None
    if guilds:
        refresh_times = [g.last_refreshed_at for g in guilds if g.last_refreshed_at]
        if refresh_times:
            last_refreshed = max(refresh_times)

    # Format response
    guild_list = []
    for guild in guilds:
        guild_list.append(
            {
                "id": guild.guild_id,
                "name": guild.name,
                "icon": guild.icon,
                "member_count": guild.member_count,
                "channels": guild.channels,
            }
        )

    logger.info(f"[handle_get_discord_channels] Returning {len(guild_list)} guilds")
    return {
        "guilds": guild_list,
        "last_refreshed_at": last_refreshed.isoformat() if last_refreshed else None,
    }


@handle_errors
async def handle_refresh_discord_channels(request: RefreshDiscordChannelsRequest):
    """Refresh guilds and channels from Discord API."""
    from eve.agent.deployments.discord_gateway import (
        refresh_discord_guilds_and_channels,
    )

    # Load deployment
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError("Deployment not found", status_code=404)

    # Verify ownership
    if str(deployment.user) != request.user_id:
        raise APIError("Unauthorized", status_code=403)

    # Verify it's a Discord deployment
    if deployment.platform != ClientType.DISCORD:
        raise APIError("Not a Discord deployment", status_code=400)

    # Call refresh function
    result = await refresh_discord_guilds_and_channels(request.deployment_id)

    return {
        "success": True,
        "guilds_count": result["guilds_count"],
        "channels_count": result["channels_count"],
        "guilds": result["guilds"],
    }


@handle_errors
async def handle_sync_discord_channels(request: SyncDiscordChannelsRequest):
    """
    Find Discord channels without sessions, create sessions, and backfill messages.
    """
    import aiohttp

    from eve.agent.agent import Agent
    from eve.agent.deployments.gateway_v2 import backfill_discord_channel
    from eve.agent.session.models import Channel

    # Load deployment
    deployment = Deployment.from_mongo(ObjectId(request.deployment_id))
    if not deployment:
        raise APIError("Deployment not found", status_code=404)

    # Verify ownership
    if str(deployment.user) != request.user_id:
        raise APIError("Unauthorized", status_code=403)

    # Verify it's a Discord deployment
    if deployment.platform != ClientType.DISCORD:
        raise APIError("Not a Discord deployment", status_code=400)

    # Get the agent for this deployment
    agent = Agent.from_mongo(deployment.agent)
    if not agent:
        raise APIError("Agent not found for deployment", status_code=404)

    # Get all subscribed channel IDs (both read and write access)
    all_channel_ids: set[str] = set()

    discord_config = deployment.config.discord if deployment.config else None
    if discord_config:
        if discord_config.channel_allowlist:
            for item in discord_config.channel_allowlist:
                all_channel_ids.add(item.id)
        if discord_config.read_access_channels:
            for item in discord_config.read_access_channels:
                all_channel_ids.add(item.id)

    if not all_channel_ids:
        logger.info(
            f"No subscribed Discord channels for deployment {request.deployment_id}"
        )
        return {
            "success": True,
            "total_channels": 0,
            "channels_with_sessions": {},
            "channels_without_sessions": [],
            "sessions_created": [],
            "backfill_results": {},
        }

    # Find sessions with matching Discord channels
    sessions_collection = Session.get_collection()
    sessions_with_channels = sessions_collection.find(
        {
            "channel.type": "discord",
            "channel.key": {"$in": list(all_channel_ids)},
        },
        {"_id": 1, "channel.key": 1},
    )

    # Get the set of channel IDs that have sessions, and map channel -> session
    channels_with_sessions: set[str] = set()
    channel_to_session: dict[str, str] = {}
    for session in sessions_with_channels:
        channel = session.get("channel", {})
        if channel and channel.get("key"):
            channel_key = channel["key"]
            channels_with_sessions.add(channel_key)
            channel_to_session[channel_key] = str(session["_id"])

    # Find channels without sessions
    channels_without_sessions = all_channel_ids - channels_with_sessions

    # Log the results
    logger.info(
        f"Discord sync for deployment {request.deployment_id}: "
        f"{len(all_channel_ids)} total channels, "
        f"{len(channels_with_sessions)} with sessions, "
        f"{len(channels_without_sessions)} without sessions"
    )

    if channel_to_session:
        logger.info(f"Channels with sessions: {channel_to_session}")

    if channels_without_sessions:
        logger.info(f"Channels without sessions: {list(channels_without_sessions)}")

    # Create sessions and backfill for channels without sessions
    sessions_created: list[dict] = []
    backfill_results: dict[str, int] = {}
    bot_token = deployment.secrets.discord.token

    async with aiohttp.ClientSession() as http_session:
        for channel_id in channels_without_sessions:
            try:
                # Fetch channel info from Discord API
                headers = {"Authorization": f"Bot {bot_token}"}
                url = f"https://discord.com/api/v10/channels/{channel_id}"

                async with http_session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to fetch channel {channel_id}: {response.status}"
                        )
                        continue

                    channel_data = await response.json()

                guild_id = channel_data.get("guild_id")
                channel_name = channel_data.get("name", f"channel-{channel_id}")

                # Build session title
                if guild_id:
                    # Fetch guild name
                    guild_url = f"https://discord.com/api/v10/guilds/{guild_id}"
                    async with http_session.get(
                        guild_url, headers=headers
                    ) as guild_response:
                        if guild_response.status == 200:
                            guild_data = await guild_response.json()
                            guild_name = guild_data.get("name", "Unknown Server")
                            session_title = f"{guild_name}: #{channel_name}"
                        else:
                            session_title = f"#{channel_name}"
                else:
                    session_title = f"#{channel_name}"

                # Create session key (same format as gateway)
                session_key = f"discord:{channel_id}"

                # Create new session
                new_session = Session(
                    owner=agent.owner,
                    agents=[agent.id],
                    title=session_title,
                    session_key=session_key,
                    platform="discord",
                    channel=Channel(type="discord", key=channel_id),
                    discord_channel_id=channel_id,
                    session_type="passive",
                    status="active",
                )
                new_session.save()

                logger.info(
                    f"Created session {new_session.id} for channel {channel_id} ({session_title})"
                )

                sessions_created.append(
                    {
                        "session_id": str(new_session.id),
                        "channel_id": channel_id,
                        "title": session_title,
                    }
                )

                # Update tracking
                channel_to_session[channel_id] = str(new_session.id)

                # Backfill messages (only for guild channels, not DMs)
                if guild_id:
                    try:
                        backfill_count = await backfill_discord_channel(
                            session=new_session,
                            channel_id=channel_id,
                            guild_id=guild_id,
                            token=bot_token,
                            agent=agent,
                            deployment=deployment,
                        )
                        backfill_results[channel_id] = backfill_count
                        logger.info(
                            f"Backfilled {backfill_count} messages for channel {channel_id}"
                        )
                    except Exception as e:
                        logger.error(f"Error backfilling channel {channel_id}: {e}")
                        backfill_results[channel_id] = -1  # Indicate error

            except Exception as e:
                logger.error(f"Error creating session for channel {channel_id}: {e}")
                continue

    return {
        "success": True,
        "total_channels": len(all_channel_ids),
        "channels_with_sessions": channel_to_session,
        "channels_without_sessions": [],  # All processed now
        "sessions_created": sessions_created,
        "backfill_results": backfill_results,
    }
