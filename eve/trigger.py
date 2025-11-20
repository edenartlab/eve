import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pytz
import sentry_sdk
from apscheduler.triggers.cron import CronTrigger
from bson import ObjectId
from fastapi import BackgroundTasks
from jinja2 import Template

from eve.agent import Agent
from eve.agent.session.models import (
    ChatMessageRequestInput,
    NotificationConfig,
    PromptSessionContext,
    Session,
    SessionUpdateConfig,
)
from eve.agent.session.session import (
    add_chat_message,
    async_prompt_session,
    build_llm_context,
)
from eve.agent.session.tracing import (
    add_breadcrumb,
    trace_async_operation,
)
from eve.api.api_requests import (
    CreateTriggerRequest,
    DeleteTriggerRequest,
    PromptSessionRequest,
    RunTriggerRequest,
    SessionCreationArgs,
)
from eve.api.errors import APIError, handle_errors
from eve.mongo import Collection, Document
from eve.user import User

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()


@Collection("triggers2")
class Trigger(Document):
    name: Optional[str] = "Untitled Task"
    schedule: Dict[str, Any]
    user: ObjectId
    agent: Optional[ObjectId] = None
    context: Optional[str] = None
    trigger_prompt: str
    posting_instructions: Optional[List[Dict[str, Any]]] = None
    # think: Optional[bool] = None
    session_type: Optional[Literal["new", "another"]] = "new"
    session: Optional[ObjectId] = None
    update_config: Optional[Dict[str, Any]] = None
    status: Optional[Literal["active", "paused", "running", "finished"]] = "active"
    deleted: Optional[bool] = False
    last_run_time: Optional[datetime] = None
    next_scheduled_run: Optional[datetime] = None


def calculate_next_scheduled_run(schedule: dict) -> datetime:
    """Calculate the next scheduled run time based on the cron schedule"""
    # Extract schedule parameters
    hour = schedule.get("hour", "*")
    minute = schedule.get("minute", "*")
    day_of_month = schedule.get("day_of_month") or schedule.get("day", "*")
    month = schedule.get("month", "*")
    day_of_week = schedule.get("day_of_week", "*")
    timezone_str = schedule.get("timezone", "UTC")
    end_date = schedule.get("end_date")

    # Create CronTrigger
    trigger = CronTrigger(
        hour=hour,
        minute=minute,
        day=day_of_month,
        month=month,
        day_of_week=day_of_week,
        timezone=timezone_str,
        end_date=end_date,
    )

    # Get next fire time
    next_time = trigger.get_next_fire_time(
        None, datetime.now(pytz.timezone(timezone_str))
    )
    if next_time:
        # Convert to UTC for storage
        return next_time.astimezone(pytz.UTC).replace(tzinfo=timezone.utc)
    return None


@handle_errors
async def handle_trigger_create(
    request: CreateTriggerRequest, background_tasks: BackgroundTasks
):
    # Log the incoming request to check for name field
    logger.info(f"Creating trigger with request: {request}")
    logger.info(f"Request model fields: {request.model_fields_set}")
    logger.info(f"Request dict: {request.model_dump()}")

    agent = Agent.from_mongo(ObjectId(request.agent))
    if not agent:
        raise APIError(f"Agent not found: {request.agent}", status_code=404)

    user = User.from_mongo(ObjectId(request.user))
    if not user:
        raise APIError(f"User not found: {request.user}", status_code=404)

    # Calculate next scheduled run
    schedule_dict = request.schedule.to_cron_dict()
    next_run = calculate_next_scheduled_run(schedule_dict)

    if not next_run:
        raise APIError("Failed to calculate next scheduled run time", status_code=400)
    logger.info(f"New Trigger next scheduled run: {next_run}")

    trigger_name = request.name or "Untitled Task"
    think = False  # TODO

    # Create trigger in database
    logger.info(f"Creating trigger with name: '{trigger_name}'")
    trigger = Trigger(
        name=trigger_name,  # Add the name field
        user=ObjectId(user.id),
        agent=ObjectId(agent.id),
        schedule=schedule_dict,
        context=request.context,
        trigger_prompt=request.trigger_prompt,
        posting_instructions=[p.model_dump() for p in request.posting_instructions],
        think=think,
        update_config=request.update_config.model_dump()
        if request.update_config
        else None,
        session=ObjectId(request.session) if request.session else None,
        session_type="another",  # request.session_type,
        next_scheduled_run=next_run,
    )
    trigger.save()

    return {
        "id": str(trigger.id),
        "next_scheduled_run": next_run.isoformat(),
    }


@handle_errors
async def handle_trigger_stop(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if not trigger or trigger.deleted:
        raise APIError(f"Trigger not found: {request.id}", status_code=404)
    trigger.status = "finished"
    trigger.save()
    return {"id": str(request.id)}


@handle_errors
async def handle_trigger_delete(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if not trigger:
        raise APIError(f"Trigger not found: {request.id}", status_code=404)
    # Soft delete by setting deleted flag
    trigger.deleted = True
    trigger.save()
    return {"id": str(trigger.id)}


@handle_errors
async def handle_trigger_get(trigger_id: str):
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger or trigger.deleted:
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


trigger_prompt_template = Template("""
{{trigger_prompt}}
{% if posting_instructions %}
<Posting Instructions>
Post the result of your last task to the following channels:
{{posting_instructions}}
</Posting Instructions>
{% endif %}
""")


async def prepare_trigger_prompt(
    trigger: Trigger,
    agent: Agent,
    session: Session,
    user_id: str,
) -> PromptSessionContext:
    extra_tools = []
    posting_instructions = ""

    for i, p in enumerate(trigger.posting_instructions or []):
        post_to = p.get("post_to")
        channel_id = p.get("channel_id")
        session_id = p.get("session_id")
        custom_instructions = p.get("custom_instructions")

        platform = {
            "discord": "discord_post",
            "telegram": "telegram_post",
            "x": "tweet",
            "farcaster": "farcaster_post",
        }

        if post_to in ["same", "another"]:
            if post_to == "same":
                posting_instructions += f"\n{i + 1}): {custom_instructions}"
            else:
                posting_instructions += f"\n{i + 1}) Post to {post_to}, channel '{session_id}': {custom_instructions}"

        elif post_to in platform:
            tool = platform.get(post_to)
            extra_tools.append(tool)
            posting_instructions += f"\n{i + 1}) Post to {post_to}, channel '{channel_id}': {custom_instructions}"

        else:
            raise APIError(f"Invalid post_to: {post_to}", status_code=400)

    trigger_prompt = trigger_prompt_template.render(
        trigger_prompt=trigger.trigger_prompt, posting_instructions=posting_instructions
    )

    # load posting tools even if they are not active by default in the agent
    tools = agent.get_tools(auth_user=user_id, extra_tools=extra_tools)
    posting_tools = {
        k: v
        for k, v in tools.items()
        if k in ["discord_post", "telegram_post", "farcaster_post", "tweet"]
    }

    return trigger_prompt, posting_tools


@handle_errors
async def execute_trigger(
    trigger_id: str,
    # background_tasks: BackgroundTasks,
) -> Session:
    # Start distributed tracing transaction
    import sentry_sdk

    transaction = sentry_sdk.start_transaction(
        name="trigger_execution",
        op="trigger.execute",
    )

    if transaction:
        transaction.set_tag("trigger_id", trigger_id)
        # Set as active span (correct API)
        sentry_sdk.Hub.current.scope.span = transaction

    try:
        from eve.api.handlers import setup_session

        trigger = Trigger.from_mongo(trigger_id)
        if not trigger:
            raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

        if trigger.status in ["finished", "running", "paused"]:
            logger.info(f"Trigger {trigger.id} is {trigger.status}, skipping...")
            return

        session = None
        user = User.from_mongo(trigger.user)
        agent = Agent.from_mongo(trigger.agent)

        # Add context to transaction
        if transaction:
            transaction.set_data("trigger_name", trigger.name)
            transaction.set_tag("user_id", str(user.id))
            transaction.set_tag("agent_id", str(agent.id))
        add_breadcrumb(f"Executing trigger: {trigger.name}", category="trigger")

        current_time = datetime.now(timezone.utc)

        if trigger.session:
            session = Session.from_mongo(trigger.session)
            request = PromptSessionRequest(
                user_id=str(user.id),
                session_id=str(trigger.session),
            )
            trigger.update(
                status="running",
                last_run_time=current_time,
            )

        else:
            # Create session request
            session_id = ObjectId()
            request = PromptSessionRequest(
                user_id=str(user.id),
                creation_args=SessionCreationArgs(
                    session_id=str(session_id),
                    owner_id=str(user.id),
                    agents=[str(agent.id)],
                    trigger=str(trigger.id),
                    title=trigger.name,
                ),
            )
            trigger.update(
                session=session_id, status="running", last_run_time=current_time
            )

        if trigger.update_config:
            try:
                request.update_config = SessionUpdateConfig(**trigger.update_config)
            except Exception as exc:
                logger.warning(
                    f"[TRIGGER] Failed to hydrate update_config for trigger {trigger.id}: {exc}"
                )

        request.notification_config = NotificationConfig(
            user_id=str(user.id),
            notification_type="trigger_complete",
            title="Task Completed",
            message=f'Your task "{trigger.name}" has completed successfully',
            trigger_id=str(trigger.id),
            agent_id=str(trigger.agent),
            priority="normal",
            # metadata={"trigger_id": trigger.trigger_id},
            success_notification=True,
            failure_notification=True,
            failure_title="Task Failed",
            failure_message=f'Your task "{trigger.name}" has failed',
        )

        # Setup session
        async with trace_async_operation(
            "trigger.setup_session",
            session_id=str(request.session_id) if request.session_id else None,
        ):
            session = setup_session(
                # background_tasks,
                None,
                request.session_id,
                request.user_id,
                request,
            )
            if transaction:
                transaction.set_data("session_id", str(session.id))

        # Prepare trigger prompt
        async with trace_async_operation("trigger.prepare_prompt"):
            trigger_prompt, extra_tools = await prepare_trigger_prompt(
                trigger, agent, session, request.user_id
            )

        # Create artwork generation message
        message = ChatMessageRequestInput(role="user", content=trigger_prompt)

        # Create context with selected model
        prompt_context = PromptSessionContext(
            session=session,
            initiating_user_id=request.user_id,
            message=message,
            update_config=request.update_config,
            # thinking_override=trigger.think,
            thinking_override=False,
            extra_tools=extra_tools,
            trigger=trigger.id,
        )

        # Add user message to session
        async with trace_async_operation("trigger.add_message"):
            await add_chat_message(session, prompt_context)

        # Build LLM context
        async with trace_async_operation("trigger.build_context"):
            llm_context = await build_llm_context(
                session,
                agent,
                prompt_context,
            )

        # Execute the prompt session (this will have its own transaction)
        add_breadcrumb("Starting prompt session", category="trigger")
        async for _ in async_prompt_session(
            session, llm_context, agent, context=prompt_context
        ):
            pass

        return session

    except Exception as e:
        logger.error(f"Error executing trigger {trigger.id}: {str(e)}")
        sentry_sdk.capture_exception(e)
        return session

    finally:
        logger.info(f"Trigger execution cleanup completed for {trigger.id}")

        next_run = calculate_next_scheduled_run(trigger.schedule)

        if next_run:
            trigger.update(status="active", next_scheduled_run=next_run)
        else:
            trigger.update(status="finished", next_scheduled_run=None)

        # Finish transaction
        if transaction:
            transaction.finish()


@handle_errors
async def handle_trigger_run(
    request: RunTriggerRequest,
    # background_tasks: BackgroundTasks,
):
    trigger_id = request.trigger_id
    trigger = Trigger.from_mongo(trigger_id)

    if not trigger or trigger.deleted:
        raise APIError(f"Trigger {trigger_id} not found", status_code=404)

    if trigger.status == "running":
        raise APIError(
            f"Trigger {trigger_id} already running, try later", status_code=400
        )

    if trigger.status != "active":
        raise APIError(
            f"Trigger {trigger_id} is not active (status: {trigger.status})",
            status_code=400,
        )

    try:
        # session_id = background_tasks.add_task(execute_trigger, trigger, background_tasks)
        # todo: use the modal func or the function directly?
        from eve.trigger import execute_trigger

        session = await execute_trigger(trigger_id)
        session_id = str(session.id)

        return {
            "trigger_id": trigger_id,
            "session_id": session_id,
            "executed": True,
        }

    except Exception as e:
        raise APIError(f"Failed to execute trigger: {str(e)}", status_code=500)


# TODO
async def handle_trigger_posting_deprecated(trigger, session_id):
    """Handle posting instructions for a trigger"""
    import aiohttp

    posting_instructions = trigger.posting_instructions
    if not posting_instructions:
        return

    try:
        request_data = {
            "session_id": session_id,
            "user_id": str(trigger.user),
            "actor_agent_ids": [str(trigger.agent)],
            "message": {
                "role": "system",
                "content": f"""## Posting instructions
{posting_instructions.get("post_to", "")} channel {posting_instructions.get("channel_id", "")}

{posting_instructions.get("custom_instructions", "")}
""",
            },
            "update_config": trigger.update_config,
        }

        # Add custom tools based on platform
        platform = posting_instructions.get("post_to")
        if platform == "discord" and posting_instructions.get("channel_id"):
            request_data["tools"] = {
                "discord_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }
        elif platform == "telegram" and posting_instructions.get("channel_id"):
            request_data["tools"] = {
                "telegram_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }

        # Make async HTTP POST to prompt session endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to run posting instructions for trigger {trigger.trigger_id}: {error_text}"
                    )

    except Exception as e:
        logger.error(
            f"Error handling posting instructions for trigger {trigger.trigger_id}: {str(e)}"
        )
        sentry_sdk.capture_exception(e)
