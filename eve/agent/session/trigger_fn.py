import os
from fastapi import BackgroundTasks
import requests
import asyncio
from datetime import datetime, timezone

from eve.agent import Agent
from eve.tool import Tool
from eve.agent.session.models import Trigger
from eve.api.api_requests import (
    PromptSessionRequest,
    SessionCreationArgs,
)
from eve.agent.session.session import (
    PromptSessionContext,
    build_llm_context,
    async_prompt_session,
    validate_prompt_session,
)


trigger_message = """
## Task

You have been given the following instructions. Do not ask for clarification, or stop until you have completed the task.

{instruction}

"""

trigger_message_post = """
## Posting instructions
{channel_info}

{post_instruction}
"""


async def trigger_fn():

    background_tasks = BackgroundTasks()

    trigger_id = os.getenv("TRIGGER_ID")
    trigger = Trigger.find_one({"trigger_id": trigger_id})

    request = PromptSessionRequest(
        session_id=str(trigger.session) if trigger.session else None,
        user_id=str(trigger.user),
        actor_agent_id=str(trigger.agent),
        message={"role": "system", "content": trigger_message.format(instruction=trigger.instruction)},
        update_config=trigger.update_config,
    )
    if not trigger.session:
        request.creation_args = SessionCreationArgs(
            owner_id=str(trigger.user),
            agents=[str(trigger.agent)],
            trigger=str(trigger.id),
        )

    from eve.api.handlers import setup_session
    session = setup_session(background_tasks, request.session_id, request.user_id, request)

    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        actor_agent_id=request.actor_agent_id,
        message=request.message,
        update_config=request.update_config,
    )
    
    validate_prompt_session(session, context)
    actor = Agent.from_mongo(trigger.agent)
    llm_context = await build_llm_context(session, actor, context)

    async for update in async_prompt_session(
        session, llm_context, actor, stream=True
    ):
        print(update)

    print(f"Completed trigger {trigger_id}")

    posting_instructions = trigger.posting_instructions

    if not posting_instructions:
        print("No posting instructions")
        return
    
    request = PromptSessionRequest(
        session_id=str(session.id),
        user_id=str(trigger.user),
        actor_agent_id=str(trigger.agent),
        message={"role": "system", "content": trigger_message_post.format(
            platform=posting_instructions["post_to"],
            channel_info=f"Post the following to {posting_instructions['post_to']}, channel {posting_instructions.get('channel_id', '')}" if posting_instructions.get('channel_id') else "",
            post_instruction=posting_instructions["custom_instructions"]
        )},
        update_config=trigger.update_config,
    )

    custom_tools = None
    
    platform = posting_instructions.get("post_to")
    if platform == "discord":
        channel_id = posting_instructions.get("channel_id", None)
        custom_tools = {"discord_post": Tool.from_raw_yaml({"parent_tool": "discord_post", **{"parameters": {"channel_id": {"default": channel_id}}}})}
    elif platform == "telegram":
        channel_id = posting_instructions.get("channel_id", None)
        custom_tools = {"telegram_post": {"parameters": {"channel_id": {"default": channel_id}}}}
    else:
        print("No platform specified")
        channel_id = None
        
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        actor_agent_id=request.actor_agent_id,
        message=request.message,
        update_config=request.update_config,
        custom_tools=custom_tools,
    )

    llm_context = await build_llm_context(session, actor, context)

    async for update in async_prompt_session(
        session, llm_context, actor, stream=True
    ):
        print(update)

    print(f"Completed posting instructions {trigger_id}")
    



    print("end date?", trigger.schedule)

    if trigger.schedule.get("end_date"):
        # Get current time with full precision
        current_time = datetime.now(timezone.utc)
        end_date_str = trigger.schedule["end_date"]

        # Parse the date string and ensure it's timezone aware
        try:
            # If using fromisoformat, the timezone info should be preserved
            end_date = datetime.fromisoformat(end_date_str)

            # If somehow end_date is still naive, make it timezone aware
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback parsing if there's a format issue
            if end_date_str.endswith("Z"):
                end_date_str = end_date_str.replace("Z", "+00:00")
            end_date = datetime.fromisoformat(end_date_str)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

        # Only round end_date to minute precision
        end_date = end_date.replace(second=0, microsecond=0)

        print(f"Current time: {current_time}")
        print(f"End date (rounded): {end_date}")

        if current_time > end_date:
            print(
                f"Trigger end date {end_date} has passed. Deleting trigger {trigger_id}"
            )
            api_url = os.getenv("EDEN_API_URL")
            response = requests.post(
                f"{api_url}/triggers/stop",
                headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
                json={"id": trigger.id},
            )

            if not response.ok:
                raise Exception(
                    f"Failed to delete trigger: {response.status_code} - {response.text}"
                )


def trigger_fn_sync():
    asyncio.run(trigger_fn())
