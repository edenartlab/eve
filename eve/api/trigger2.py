from eve.api.modal_app import app, image


import logging
import aiohttp
import os
import pytz
from datetime import datetime, timezone
from apscheduler.triggers.cron import CronTrigger

from eve.api.api_requests import CronSchedule
from eve.api.errors import handle_errors


from eve.agent.session.models import Trigger

from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs



# from eve.api.handlers import setup_session
from eve.session2 import (
    setup_session,
    add_user_message, 
    async_prompt_session, 
    build_llm_context
)


from eve.agent.agent import Agent
from eve.user import User
from eve.agent.session.models import Session
from eve.agent.session.models import (
    Session,
    ChatMessageRequestInput, 
    LLMConfig,  
    PromptSessionContext, 
    Session,
    Trigger
)

from eve.agent.session.models import PromptSessionContext
from fastapi import FastAPI, Depends, BackgroundTasks, Request
import uuid
import sentry_sdk
from bson import ObjectId


from eve import auth
import time
import modal
# from eve.api.api import app, web_app, image, pre_modal_setup
from typing import List

from eve.api.errors import handle_errors, APIError

from eve.api.api_requests import (
    CreateTriggerRequest,
    DeleteTriggerRequest,
    RunTriggerRequest
)





logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()


















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
        end_date=end_date
    )
    
    # Get next fire time
    next_time = trigger.get_next_fire_time(None, datetime.now(pytz.timezone(timezone_str)))
    if next_time:
        # Convert to UTC for storage
        return next_time.astimezone(pytz.UTC).replace(tzinfo=timezone.utc)
    return None


@handle_errors
async def create_trigger_fn(
    schedule: CronSchedule,
    trigger_id: str,
) -> datetime:
    """Calculate and return the next scheduled run time"""
    print(f"Creating session trigger {trigger_id} with schedule {schedule}")
    schedule_dict = schedule
    
    # Calculate next scheduled run
    next_run = calculate_next_scheduled_run(schedule_dict)
    
    if next_run:
        logger.info(f"Trigger {trigger_id} next scheduled run: {next_run}")
    else:
        logger.warning(f"Could not calculate next run time for trigger {trigger_id}")
    
    return next_run


async def execute_trigger_old_deprecated(trigger, is_immediate: bool = False):
    """
    Execute a trigger using the current session-based system.
    Used by both scheduled and immediate trigger execution.
    
    Args:
        trigger: Trigger model instance
        is_immediate: Whether this is an immediate execution (affects notifications)
    
    Returns:
        dict: Response from the session prompt endpoint
    """
    
    
    logger.info(f"Executing trigger {trigger.trigger_id} (immediate={is_immediate})")
    
    # Prepare the prompt session request
    request_data = {
        "session_id": str(trigger.session) if trigger.session else None,
        "user_id": str(trigger.user),
        "actor_agent_ids": [str(trigger.agent)],
        "message": {
            "role": "system",
            "content": f"""## Task

You have been given the following instructions. Do not ask for clarification, or stop until you have completed the task.

{trigger.instruction}

""",
        },
        "update_config": trigger.update_config,
    }
    
    # If no session, add creation args
    if not trigger.session:
        request_data["creation_args"] = {
            "owner_id": str(trigger.user),
            "agents": [str(trigger.agent)],
            "trigger": str(trigger.id),
        }
    
    # Add notification configuration
    notification_type = "Immediate Task" if is_immediate else "Task"
    request_data["notification_config"] = {
        "user_id": str(trigger.user),
        "notification_type": "trigger_complete",
        "title": f"{notification_type} Completed Successfully",
        "message": f"Your {'immediate ' if is_immediate else 'scheduled '}task has completed successfully: {trigger.instruction[:100]}...",
        "trigger_id": str(trigger.id),
        "agent_id": str(trigger.agent),
        "priority": "normal",
        "metadata": {
            "trigger_id": trigger.trigger_id,
            "immediate_run": is_immediate,
        },
        "success_notification": True,
        "failure_notification": True,
        "failure_title": f"{notification_type} Failed",
        "failure_message": f"Your {'immediate ' if is_immediate else 'scheduled '}task failed: {trigger.instruction[:100]}...",
    }
    
    # Make async HTTP POST to prompt session endpoint
    api_url = os.getenv("EDEN_API_URL")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/sessions/prompt",
            json=request_data,
            headers={
                "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(
                    f"Failed to execute trigger {trigger.trigger_id}: {error_text}"
                )
                raise Exception(f"Failed to execute trigger: {response.status} - {error_text}")
            
            result = await response.json()
            session_id = result.get("session_id")
            
            logger.info(f"Successfully executed trigger {trigger.trigger_id}, session: {session_id}")
            return result


async def stop_trigger(trigger_id: str) -> None:
    """Mark trigger as stopped in database"""
    logger.info(f"Stopping trigger {trigger_id}")
    # No need to stop Modal app anymore since we're using a centralized scheduler





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

    trigger_id = f"{db}_{str(user.id)}_{int(time.time())}"

    # Calculate next scheduled run
    schedule_dict = request.schedule.to_cron_dict()
    next_run = await create_trigger_fn(
        schedule=schedule_dict,
        trigger_id=trigger_id,
    )

    if not next_run:
        raise APIError("Failed to calculate next scheduled run time", status_code=400)

    # Create trigger in database  
    logger.info(f"Creating trigger with name: '{request.name}'")
    trigger = Trigger(
        trigger_id=trigger_id,
        name=request.name,  # Add the name field
        user=ObjectId(user.id),
        agent=ObjectId(agent.id),
        schedule=schedule_dict,
        instruction=request.instruction,
        posting_instructions=request.posting_instructions.model_dump()
        if request.posting_instructions
        else None,
        think=request.think,
        update_config=request.update_config.model_dump()
        if request.update_config
        else None,
        session=ObjectId(request.session) if request.session else None,
        session_type=request.session_type,
        next_scheduled_run=next_run,
    )
    trigger.save()

    return {
        "id": str(trigger.id),
        "trigger_id": trigger_id,
        "next_scheduled_run": next_run.isoformat(),
    }


@handle_errors
async def handle_trigger_stop(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if not trigger or trigger.deleted:
        raise APIError(f"Trigger not found: {request.id}", status_code=404)
    await stop_trigger(trigger.trigger_id)
    trigger.status = "finished"
    trigger.save()

    return {"id": str(request.id)}


@handle_errors
async def handle_trigger_delete(request: DeleteTriggerRequest):
    trigger = Trigger.from_mongo(request.id)
    if not trigger:
        raise APIError(f"Trigger not found: {request.id}", status_code=404)

    if trigger.status != "finished":
        await stop_trigger(trigger.trigger_id)

    # Soft delete by setting deleted flag
    trigger.deleted = True
    trigger.save()

    return {"id": str(trigger.id)}



@handle_errors
async def handle_trigger_get(trigger_id: str):
    trigger = Trigger.load(trigger_id=trigger_id)
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











@handle_errors
@app.function(image=image, max_containers=4, timeout=3600)
async def execute_trigger(trigger: Trigger) -> Session:    
    try: 
        if trigger.status in ["finished", "running", "paused"]:
            logger.info(f"Trigger {trigger.id} is {trigger.status}, skipping...")
            return
        
        user = User.from_mongo(trigger.user)
        agent = Agent.from_mongo(trigger.agent)
        task_instruction = trigger.instruction

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
                    title=trigger.name
                )
            )
            trigger.update(
                session=session_id,
                status="running",
                last_run_time=current_time
            )
        
        # Setup session
        session = setup_session(
            request.session_id, 
            request.user_id, 
            request
        )

        # Create artwork generation message
        message = ChatMessageRequestInput(
            role="user",
            content=task_instruction
        )
        
        # Create context with selected model
        context = PromptSessionContext(
            session=session,
            initiating_user_id=request.user_id,
            message=message,
            thinking_override=trigger.think,
        )
        
        # Add user message to session
        add_user_message(session, context, pin=True)
        
        # Build LLM context
        context = await build_llm_context(
            session, 
            agent, 
            context, 
            trace_id=str(uuid.uuid4()), 
        )
        
        # Execute the prompt session
        async for _ in async_prompt_session(session, context, agent):
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
            trigger.update(
                status="active",
                next_scheduled_run=next_run
            )
        else:
            trigger.update(
                status="finished",
                next_scheduled_run=None
            )



@app.function(image=image, max_containers=1)
async def run_scheduled_triggers():
    """Check for and run scheduled triggers every minute"""

    # Find active triggers which should be run now
    triggers = Trigger.find({
        "status": "active",
        "deleted": {"$ne": True},
        "next_scheduled_run": {"$lte": datetime.now(timezone.utc)},
    })
    triggers = [Trigger.from_mongo(ObjectId(t)) for t in ["68b3bca333da060a73cef02a", "68b3bc8f33da060a73cef029"]]

    if not triggers:
        logger.info("No triggers to run")
        return
    
    logger.info(f"Running {len(triggers)} triggers")

    await execute_trigger.spawn_map.aio(triggers)





@handle_errors
async def handle_trigger_run(
    request: RunTriggerRequest,
    background_tasks: BackgroundTasks,
):
    trigger_id = request.trigger_id    
    trigger = Trigger.from_mongo(trigger_id)

    if not trigger or trigger.deleted:
        raise APIError(f"Trigger {trigger_id} not found", status_code=404)

    if trigger.status == "running":
        raise APIError(f"Trigger {trigger_id} already running, try later", status_code=400)

    if trigger.status != "active":
        raise APIError(f"Trigger {trigger_id} is not active (status: {trigger.status})", status_code=400)
    
    try:
        session_id = background_tasks.add_task(execute_trigger, trigger, background_tasks)
        
        return {
            "trigger_id": trigger_id,
            "session_id": session_id,
            "executed": True,
        }
        
    except Exception as e:
        raise APIError(f"Failed to execute trigger: {str(e)}", status_code=500)


