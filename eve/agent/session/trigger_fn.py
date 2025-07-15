import os
from fastapi import BackgroundTasks
import requests
import asyncio
from datetime import datetime, timezone
import sentry_sdk

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
    async_title_session,
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


async def create_notification(
    user_id: str,
    notification_type: str,
    title: str,
    message: str,
    trigger_id: str = None,
    session_id: str = None,
    agent_id: str = None,
    priority: str = "normal",
    action_url: str = None,
    metadata: dict = None,
):
    """Create a notification via API call"""
    try:
        api_url = os.getenv("EDEN_API_URL")
        if not api_url:
            print("***debug*** EDEN_API_URL not set, skipping notification creation")
            return

        notification_data = {
            "user_id": user_id,
            "type": notification_type,
            "title": title,
            "message": message,
            "priority": priority,
        }

        if trigger_id:
            notification_data["trigger_id"] = trigger_id
        if session_id:
            notification_data["session_id"] = session_id
        if agent_id:
            notification_data["agent_id"] = agent_id
        if action_url:
            notification_data["action_url"] = action_url
        if metadata:
            notification_data["metadata"] = metadata

        response = requests.post(
            f"{api_url}/notifications/create",
            headers={
                "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                "Content-Type": "application/json",
            },
            json=notification_data,
        )

        if response.ok:
            print(f"***debug*** Created notification: {title}")
        else:
            print(
                f"***debug*** Failed to create notification: {response.status_code} - {response.text}"
            )

    except Exception as e:
        print(f"***debug*** Error creating notification: {str(e)}")
        # Don't raise - notification creation failure shouldn't stop the trigger


async def trigger_fn():
    background_tasks = BackgroundTasks()
    trigger_id = os.getenv("TRIGGER_ID")

    # Set up sentry context for trigger execution
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("trigger_execution", True)
        scope.set_tag("trigger_id", trigger_id)
        scope.set_context("trigger", {"trigger_id": trigger_id})

    try:
        # Stage 1: Trigger lookup
        try:
            trigger = Trigger.find_one(
                {"trigger_id": trigger_id, "deleted": {"$ne": True}}
            )
            if not trigger:
                raise Exception(f"Trigger {trigger_id} not found")

            # Update last_run_time at the start of execution
            trigger.last_run_time = datetime.now(timezone.utc)
            trigger.save()

        except Exception as e:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("trigger_failure", True)
                scope.set_tag("failure_stage", "lookup")
            sentry_sdk.capture_exception(e)
            raise

        # Stage 2: Session setup
        try:
            request = PromptSessionRequest(
                session_id=str(trigger.session) if trigger.session else None,
                user_id=str(trigger.user),
                actor_agent_id=str(trigger.agent),
                message={
                    "role": "system",
                    "content": trigger_message.format(instruction=trigger.instruction),
                },
                update_config=trigger.update_config,
            )
            if not trigger.session:
                request.creation_args = SessionCreationArgs(
                    owner_id=str(trigger.user),
                    agents=[str(trigger.agent)],
                    trigger=str(trigger.id),
                )

            from eve.api.handlers import setup_session

            session = setup_session(
                background_tasks, request.session_id, request.user_id, request
            )

            context = PromptSessionContext(
                session=session,
                initiating_user_id=request.user_id,
                actor_agent_id=request.actor_agent_id,
                message=request.message,
                update_config=request.update_config,
            )

            validate_prompt_session(session, context)
            actor = Agent.from_mongo(trigger.agent)
        except Exception as e:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("trigger_failure", True)
                scope.set_tag("failure_stage", "session_setup")
            sentry_sdk.capture_exception(e)
            raise

        # Stage 3: LLM context building and prompt session
        try:
            llm_context = await build_llm_context(session, actor, context)

            async for update in async_prompt_session(
                session, llm_context, actor, stream=True
            ):
                print(update)

            print(f"Completed trigger {trigger_id}")
        except Exception as e:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("trigger_failure", True)
                scope.set_tag("failure_stage", "prompt_session")
            sentry_sdk.capture_exception(e)
            raise

        # Stage 4: Title generation
        try:
            await async_title_session(session, trigger.instruction)
        except Exception as e:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("trigger_failure", True)
                scope.set_tag("failure_stage", "title_generation")
            sentry_sdk.capture_exception(e)
            # Don't raise - title generation failure shouldn't stop the trigger

        # Stage 5: Posting instructions
        posting_instructions = trigger.posting_instructions

        if not posting_instructions:
            print("No posting instructions")
        else:
            try:
                request = PromptSessionRequest(
                    session_id=str(session.id),
                    user_id=str(trigger.user),
                    actor_agent_id=str(trigger.agent),
                    message={
                        "role": "system",
                        "content": trigger_message_post.format(
                            platform=posting_instructions["post_to"],
                            channel_info=f"Post the following to {posting_instructions['post_to']}, channel {posting_instructions.get('channel_id', '')}"
                            if posting_instructions.get("channel_id")
                            else "",
                            post_instruction=posting_instructions[
                                "custom_instructions"
                            ],
                        ),
                    },
                    update_config=trigger.update_config,
                )

                custom_tools = None

                platform = posting_instructions.get("post_to")
                if platform == "discord":
                    channel_id = posting_instructions.get("channel_id", None)
                    custom_tools = {
                        "discord_post": Tool.from_raw_yaml(
                            {
                                "parent_tool": "discord_post",
                                **{
                                    "parameters": {
                                        "channel_id": {"default": channel_id}
                                    }
                                },
                            }
                        )
                    }
                elif platform == "telegram":
                    channel_id = posting_instructions.get("channel_id", None)
                    custom_tools = {
                        "telegram_post": {
                            "parameters": {"channel_id": {"default": channel_id}}
                        }
                    }
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
            except Exception as e:
                with sentry_sdk.configure_scope() as scope:
                    scope.set_tag("trigger_failure", True)
                    scope.set_tag("failure_stage", "posting")
                sentry_sdk.capture_exception(e)
                raise

        # Stage 5.5: Notify trigger completion
        try:
            await create_notification(
                user_id=str(trigger.user),
                notification_type="trigger_complete",
                title="Trigger Completed Successfully",
                message=f"Your scheduled trigger has completed successfully: {trigger.instruction[:100]}...",
                trigger_id=str(trigger.id),
                session_id=str(session.id),
                agent_id=str(trigger.agent),
                priority="normal",
                action_url=f"/sessions/{session.id}",
                metadata={
                    "trigger_id": trigger_id,
                    "completion_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            print(f"***debug*** Failed to create completion notification: {str(e)}")
            # Don't raise - notification failure shouldn't stop the trigger

        # Stage 6: End date checking and deletion
        try:
            print("end date?", trigger.schedule)

            if trigger.schedule.get("end_date"):
                # Get current time with full precision
                current_time = datetime.now(timezone.utc)
                end_date = trigger.schedule["end_date"]

                # Handle both datetime objects and string formats
                if isinstance(end_date, str):
                    # Parse the date string and ensure it's timezone aware
                    try:
                        # If using fromisoformat, the timezone info should be preserved
                        end_date = datetime.fromisoformat(end_date)

                        # If somehow end_date is still naive, make it timezone aware
                        if end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=timezone.utc)
                    except ValueError:
                        # Fallback parsing if there's a format issue
                        if end_date.endswith("Z"):
                            end_date = end_date.replace("Z", "+00:00")
                        end_date = datetime.fromisoformat(end_date)
                        if end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=timezone.utc)
                elif isinstance(end_date, datetime):
                    # Already a datetime object, just ensure it's timezone aware
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                else:
                    raise ValueError(f"Unsupported end_date type: {type(end_date)}")

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
                        headers={
                            "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"
                        },
                        json={"id": str(trigger.id)},
                    )

                    if not response.ok:
                        raise Exception(
                            f"Failed to delete trigger: {response.status_code} - {response.text}"
                        )
        except Exception as e:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("trigger_failure", True)
                scope.set_tag("failure_stage", "end_date_check")
            sentry_sdk.capture_exception(e)
            raise

    except Exception as e:
        # Top-level error handling - notify failure
        try:
            # Try to get trigger info for failure notification
            if "trigger" in locals():
                await create_notification(
                    user_id=str(trigger.user),
                    notification_type="trigger_failed",
                    title="Trigger Failed",
                    message=f"Your scheduled trigger failed: {str(e)[:200]}...",
                    trigger_id=str(trigger.id),
                    session_id=str(session.id) if "session" in locals() else None,
                    agent_id=str(trigger.agent),
                    priority="high",
                    action_url=f"/sessions/{session.id}"
                    if "session" in locals()
                    else None,
                    metadata={
                        "trigger_id": trigger_id,
                        "error": str(e),
                        "failure_time": datetime.now(timezone.utc).isoformat(),
                    },
                )
        except Exception as notification_error:
            print(
                f"***debug*** Failed to create failure notification: {str(notification_error)}"
            )

        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("trigger_failure", True)
            scope.set_tag("failure_stage", "general")
        sentry_sdk.capture_exception(e)
        raise


def trigger_fn_sync():
    try:
        asyncio.run(trigger_fn())
    except Exception as e:
        trigger_id = os.getenv("TRIGGER_ID")
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("trigger_failure", True)
            scope.set_tag("trigger_id", trigger_id)
            scope.set_tag("failure_stage", "sync_wrapper")
        sentry_sdk.capture_exception(e)
        raise
