import asyncio
import json
import logging
import modal
import os
import time
import uuid
import hashlib
import hmac
import random

import aiohttp
from bson import ObjectId
from typing import List, Optional
from fastapi import BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse

from eve.agent.deployments.farcaster import FarcasterClient
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.models import (
    PromptSessionContext,
    Session,
    ChatMessage,
    EdenMessageType,
    EdenMessageData,
    EdenMessageAgentData,
    Deployment,
    DeploymentConfig,
    DeploymentSecrets,
    Notification,
    NotificationChannel,
    NotificationConfig,
    SessionUpdateConfig,
    ChatMessageRequestInput,
    EmailDomain,
)
from eve.agent.memory.memory_models import messages_to_text, select_messages
from eve.agent.session.session import run_prompt_session, run_prompt_session_stream
from eve.trigger import (
    Trigger,
)
from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CancelRequest,
    CancelSessionRequest,
    CreateDeploymentRequestV2,
    DeleteDeploymentRequestV2,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    PromptSessionRequest,
    TaskRequest,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
    UpdateDeploymentRequestV2,
    CreateNotificationRequest,
    SessionCreationArgs,
)
from eve.api.helpers import (
    get_platform_client,
)
from eve.utils import serialize_json
from eve.tools.replicate_tool import replicate_update_task
from eve.agent.session.session_llm import LLMContext, LLMConfig, async_prompt
from eve.agent.session.models import LLMContextMetadata, LLMTraceMetadata
from eve.mongo import get_collection, MongoDocumentNotFound
from eve.task import Task
from eve.tool import Tool
from eve.agent import Agent
from eve.user import User

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

    if background_tasks:
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

        # TODO: titling
        if background_tasks:
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
        "session_key": request.creation_args.session_key,
        "platform": request.creation_args.platform,
        "status": "active",
        "trigger": ObjectId(request.creation_args.trigger)
        if request.creation_args.trigger
        else None,
    }

    # Only include budget if it's not None, so default factory can work
    if request.creation_args.budget is not None:
        session_kwargs["budget"] = request.creation_args.budget

    if request.creation_args.parent_session:
        session_kwargs["parent_session"] = ObjectId(
            request.creation_args.parent_session
        )

    if request.creation_args.extras:
        session_kwargs["extras"] = request.creation_args.extras

    session = Session(**session_kwargs)
    session.save()

    # Update trigger with session ID
    if request.creation_args.trigger:
        trigger = Trigger.from_mongo(ObjectId(request.creation_args.trigger))
        if trigger and not trigger.deleted:
            trigger.session = session.id
            trigger.save()

    # Create eden message for initial agent additions
    agents = [Agent.from_mongo(agent_id) for agent_id in agent_object_ids]
    agents = [agent for agent in agents if agent]  # Filter out None values
    if agents:
        create_eden_message(session.id, EdenMessageType.AGENT_ADD, agents)

    # Generate title for new sessions if no title provided and we have background tasks
    # TODO: titling
    if background_tasks:
        generate_session_title(session, request, background_tasks)

    return session


@handle_errors
async def handle_prompt_session(
    request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    session = setup_session(
        background_tasks, request.session_id, request.user_id, request
    )
    # Convert notification_config dict to NotificationConfig object if present
    notification_config = None
    if request.notification_config:
        notification_config = NotificationConfig(**request.notification_config)

    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        actor_agent_ids=request.actor_agent_ids,
        message=request.message,
        update_config=request.update_config,
        llm_config=request.llm_config,
        notification_config=notification_config,
        thinking_override=request.thinking,  # Pass thinking override
        acting_user_id=request.acting_user_id or request.user_id,
        api_key_id=request.api_key_id,  # Pass API key ID to context
        trigger=ObjectId(request.trigger)
        if request.trigger
        else None,  # Pass trigger ID to mark automated messages
    )

    if request.stream:

        async def event_generator():
            try:
                from eve.utils import dumps_json

                async for data in run_prompt_session_stream(context, background_tasks):
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
        run_prompt_session,
        context=context,
        background_tasks=background_tasks,
    )

    return {"session_id": str(session.id)}


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
    """Update the status of a session."""
    try:
        # Request is already an UpdateSessionStatusRequest from FastAPI
        session = Session.from_mongo(ObjectId(request.session_id))

        try:
            db = os.getenv("DB", "STAGE").upper()
            func = modal.Function.from_name(
                f"api-{db.lower()}",
                "handle_session_status_change_fn",
                environment_name="main",
            )
            func.spawn(request.session_id, request.status)
            session.update(status=request.status)
            return {
                "success": True,
                "session_id": request.session_id,
                "status": request.status,
            }
        except Exception as e:
            logger.warning(
                f"Failed to spawn Modal function for session status change: {e}"
            )
            session.update(status="paused")
            return {"success": False, "error": str(e)}

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


import torch, torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

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
