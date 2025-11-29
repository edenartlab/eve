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
    PromptSessionRequest,
    SessionCreationArgs,
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
        user_id=request.user_id, agent_id=None, args=request.args, public=request.public
    )

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
    handle = create_prompt_session_handle(request, background_tasks)
    session_id = handle.session_id

    # Add user message first (decoupled from orchestration)
    await handle.add_message()

    if request.stream:

        async def event_generator():
            try:
                from eve.utils import dumps_json

                async for data in handle.stream_updates():
                    yield f"data: {dumps_json({'event': 'update', 'data': data})}\n\n"
                yield f"data: {dumps_json({'event': 'done', 'data': ''})}\n\n"
            except Exception as e:
                logger.error("Error in event_generator", exc_info=True)
                yield f"data: {dumps_json({'event': 'error', 'data': {'error': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    background_tasks.add_task(
        handle.run,
    )

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
    """
    if not request.session_id:
        raise APIError("session_id is required for /sessions/run", status_code=400)

    handle = create_prompt_session_handle(request, background_tasks)
    session_id = handle.session_id

    # Run orchestration only (no message addition)
    if request.stream:

        async def event_generator():
            try:
                from eve.utils import dumps_json

                async for data in handle.stream_updates():
                    yield f"data: {dumps_json({'event': 'update', 'data': data})}\n\n"
                yield f"data: {dumps_json({'event': 'done', 'data': ''})}\n\n"
            except Exception as e:
                logger.error("Error in event_generator", exc_info=True)
                yield f"data: {dumps_json({'event': 'error', 'data': {'error': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    background_tasks.add_task(
        handle.run,
    )

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
        deployment.update(**update_dict)

        # Call platform-specific update hook if it exists
        try:
            agent = Agent.from_mongo(ObjectId(deployment.agent))
            client = get_platform_client(
                agent=agent, platform=deployment.platform, deployment=deployment
            )

            # Reload deployment to get updated values from MongoDB
            deployment.reload()

            # Convert reloaded config/secrets to proper objects (MongoDB returns dicts)
            new_config = (
                DeploymentConfig(**deployment.config)
                if isinstance(deployment.config, dict)
                else deployment.config
            )
            new_secrets = (
                DeploymentSecrets(**deployment.secrets)
                if isinstance(deployment.secrets, dict)
                else deployment.secrets
            )

            # Call platform-specific update hook
            await client.update(
                old_config=old_config,
                new_config=new_config,
                old_secrets=old_secrets,
                new_secrets=new_secrets,
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
            model="gpt-5",
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


# =============================================================================
# Artifact Handlers
# =============================================================================


@handle_errors
async def handle_artifact_create(request):
    """Create a new artifact."""
    from eve.artifact import Artifact

    user_id = ObjectId(request.user_id)
    session_id = ObjectId(request.session_id) if request.session_id else None

    artifact = Artifact(
        type=request.type,
        name=request.name,
        description=request.description,
        data=request.data,
        owner=user_id,
        session=session_id if request.link_to_session else None,
        sessions=[session_id] if session_id and request.link_to_session else [],
    )
    artifact.save()

    # Link artifact to session if specified
    if session_id and request.link_to_session:
        session = Session.from_mongo(session_id)
        session.link_artifact(artifact.id)

    return {
        "success": True,
        "artifact_id": str(artifact.id),
        "type": artifact.type,
        "name": artifact.name,
        "version": artifact.version,
    }


@handle_errors
async def handle_artifact_get(request):
    """Get an artifact by ID."""
    from eve.artifact import Artifact

    artifact = Artifact.from_mongo(ObjectId(request.artifact_id))

    if request.view == "summary":
        result = {
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "name": artifact.name,
            "description": artifact.description,
            "version": artifact.version,
            "summary": artifact.get_summary(),
            "updated_at": artifact.updatedAt.isoformat() if artifact.updatedAt else None,
            "created_at": artifact.createdAt.isoformat() if artifact.createdAt else None,
            "archived": artifact.archived,
        }
    else:
        result = {
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "name": artifact.name,
            "description": artifact.description,
            "version": artifact.version,
            "data": artifact.data,
            "updated_at": artifact.updatedAt.isoformat() if artifact.updatedAt else None,
            "created_at": artifact.createdAt.isoformat() if artifact.createdAt else None,
            "archived": artifact.archived,
            "owner": str(artifact.owner),
            "session": str(artifact.session) if artifact.session else None,
            "sessions": [str(s) for s in artifact.sessions],
        }

    if request.include_history and artifact.versions:
        result["history"] = [
            {
                "version": v.version,
                "timestamp": v.timestamp.isoformat(),
                "actor_type": v.actor_type,
                "message": v.message,
                "operations_count": len(v.operations),
            }
            for v in artifact.versions[-20:]
        ]

    return result


@handle_errors
async def handle_artifact_update(request):
    """Update an artifact with structured operations."""
    from eve.artifact import Artifact

    artifact = Artifact.from_mongo(ObjectId(request.artifact_id))

    # Verify ownership
    user_id = ObjectId(request.user_id)
    if artifact.owner != user_id:
        raise APIError("Unauthorized: You do not own this artifact", status_code=403)

    actor_id = ObjectId(request.actor_id) if request.actor_id else None
    previous_version = artifact.version

    artifact.apply_operations(
        operations=request.operations,
        actor_type=request.actor_type,
        actor_id=actor_id,
        message=request.message,
        save=True,
    )

    return {
        "success": True,
        "artifact_id": str(artifact.id),
        "name": artifact.name,
        "previous_version": previous_version,
        "new_version": artifact.version,
        "operations_applied": len(request.operations),
        "data": artifact.data,
    }


@handle_errors
async def handle_artifact_list(request):
    """List artifacts for a user or session."""
    from eve.artifact import Artifact

    if request.session_id:
        session_id = ObjectId(request.session_id)
        artifacts = Artifact.find_for_session(
            session_id, include_archived=request.include_archived
        )
    elif request.user_id:
        user_id = ObjectId(request.user_id)
        artifacts = Artifact.find_for_user(
            user_id,
            artifact_type=request.type,
            include_archived=request.include_archived,
            limit=request.limit,
        )
    else:
        raise APIError("Either user_id or session_id is required", status_code=400)

    # Filter by type if session query and type specified
    if request.session_id and request.type:
        artifacts = [a for a in artifacts if a.type == request.type]

    # Apply limit
    artifacts = artifacts[: request.limit]

    return {
        "count": len(artifacts),
        "artifacts": [
            {
                "artifact_id": str(a.id),
                "type": a.type,
                "name": a.name,
                "description": a.description,
                "version": a.version,
                "summary": a.get_summary(max_length=100),
                "updated_at": a.updatedAt.isoformat() if a.updatedAt else None,
                "archived": a.archived,
            }
            for a in artifacts
        ],
    }


@handle_errors
async def handle_artifact_delete(request):
    """Archive (soft delete) an artifact."""
    from eve.artifact import Artifact

    artifact = Artifact.from_mongo(ObjectId(request.artifact_id))

    # Verify ownership
    user_id = ObjectId(request.user_id)
    if artifact.owner != user_id:
        raise APIError("Unauthorized: You do not own this artifact", status_code=403)

    artifact.archive()

    return {
        "success": True,
        "artifact_id": str(artifact.id),
        "archived": True,
    }


@handle_errors
async def handle_artifact_link_session(request):
    """Link an artifact to a session."""
    from eve.artifact import Artifact

    artifact = Artifact.from_mongo(ObjectId(request.artifact_id))
    session = Session.from_mongo(ObjectId(request.session_id))

    # Verify ownership
    user_id = ObjectId(request.user_id)
    if artifact.owner != user_id:
        raise APIError("Unauthorized: You do not own this artifact", status_code=403)

    artifact.link_to_session(session.id)
    session.link_artifact(artifact.id)

    return {
        "success": True,
        "artifact_id": str(artifact.id),
        "session_id": str(session.id),
    }


@handle_errors
async def handle_artifact_rollback(request):
    """Rollback an artifact to a previous version."""
    from eve.artifact import Artifact

    artifact = Artifact.from_mongo(ObjectId(request.artifact_id))

    # Verify ownership
    user_id = ObjectId(request.user_id)
    if artifact.owner != user_id:
        raise APIError("Unauthorized: You do not own this artifact", status_code=403)

    previous_version = artifact.version

    success = artifact.rollback_to_version(request.target_version)

    if not success:
        raise APIError(
            f"Version {request.target_version} not found in artifact history",
            status_code=404,
        )

    return {
        "success": True,
        "artifact_id": str(artifact.id),
        "previous_version": previous_version,
        "new_version": artifact.version,
        "rolled_back_to": request.target_version,
    }
