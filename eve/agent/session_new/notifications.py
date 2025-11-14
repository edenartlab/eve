import os
import traceback
from typing import Optional, Dict
import uuid

from bson import ObjectId
from loguru import logger
from sentry_sdk import capture_exception

from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
    Session,
)
from eve.agent.session.session_llm import async_prompt


async def check_if_session_active(user_id: str, session_id: str) -> dict:
    """Check if user is actively viewing a session via the API"""
    import httpx

    try:
        api_url = os.getenv("EDEN_FASTIFY_API_URL")
        if not api_url:
            logger.warning(
                "[NOTIFICATION] EDEN_FASTIFY_API_URL not set - assuming user not active"
            )
            return {"is_active": False, "redis_available": False}

        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": f"{os.getenv('EDEN_FASTIFY_ADMIN_KEY')}",
                "Content-Type": "application/json",
            }

            response = await client.get(
                f"{api_url}/v2/sessions/is-active",
                params={"user_id": user_id, "session_id": session_id},
                headers=headers,
                timeout=2.0,  # Quick timeout to avoid blocking
            )

            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.warning(
                    f"[NOTIFICATION] API returned status {response.status_code} - assuming not active"
                )
                return {"is_active": False, "redis_available": False}

    except Exception as e:
        logger.error(f"[NOTIFICATION] Error checking session activity: {str(e)}")
        # Fail open - assume not active so notification gets created
        return {"is_active": False, "redis_available": False}


async def create_session_message_notification(
    user_id: str, session_id: str, agent_id: str
):
    """Create a notification for a new session message via the Fastify API"""
    import httpx

    try:
        api_url = os.getenv("EDEN_FASTIFY_API_URL")
        if not api_url:
            logger.warning(
                "[NOTIFICATION] EDEN_FASTIFY_API_URL not set - cannot create notification"
            )
            return

        notification_data = {
            "user_id": user_id,
            "type": "session_message",
            "title": "New message",
            "message": "You have a new message in your session",
            "priority": "normal",
            "session_id": session_id,
            "agent_id": agent_id,
            "action_url": f"/sessions/{session_id}",
        }

        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": os.getenv("EDEN_FASTIFY_ADMIN_KEY"),
                "Content-Type": "application/json",
            }
            response = await client.post(
                f"{api_url}/v2/notifications",
                headers=headers,
                json=notification_data,
                timeout=5.0,
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    f"[NOTIFICATION] ❌ Failed to create notification (status {response.status_code}): {error_text}"
                )

    except Exception as e:
        logger.error(
            f"[NOTIFICATION] ❌ Error creating session message notification: {str(e)}"
        )
        import traceback

        traceback.print_exc()
        # Don't re-raise - notification creation shouldn't block message delivery


async def _send_session_notification(
    notification_config, session: Session, success: bool = True, error: str = None
):
    """Send a notification about session completion"""
    import httpx
    from datetime import datetime, timezone

    try:
        api_url = os.getenv("EDEN_API_URL")
        if not api_url:
            return

        # Determine notification details based on success/failure
        if success:
            notification_type = notification_config.notification_type
            title = notification_config.success_title or notification_config.title
            message = notification_config.success_message or notification_config.message
            priority = notification_config.priority
        else:
            notification_type = "session_failed"
            title = notification_config.failure_title or "Session Failed"
            message = (
                notification_config.failure_message
                or f"Your session failed: {error[:200]}..."
            )
            priority = "high"

        notification_data = {
            "user_id": notification_config.user_id,
            "type": notification_type,
            "title": title,
            "message": message,
            "priority": priority,
            "session_id": str(session.id),
            "action_url": f"/sessions/{session.id}",
            "metadata": {
                "session_id": str(session.id),
                "completion_time": datetime.now(timezone.utc).isoformat(),
                **(notification_config.metadata or {}),
                **({"error": error} if error else {}),
            },
        }

        # Add optional fields
        if notification_config.trigger_id:
            notification_data["trigger_id"] = notification_config.trigger_id
        if notification_config.agent_id:
            notification_data["agent_id"] = notification_config.agent_id

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/v2/notifications",
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
                json=notification_data,
            )
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"Failed to create notification: {error_text}")

    except Exception as e:
        logger.error(f"Error creating session notification: {str(e)}")
        capture_exception(e)


async def async_title_session(
    session: Session, initial_message_content: str, metadata: Optional[Dict] = None
):
    """
    Generate a title for a session based on the initial message content
    """

    from pydantic import BaseModel, Field

    class TitleResponse(BaseModel):
        """A title for a session of chat messages. It must entice a user to click on the session when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat session. It should be concise and terse, and not include any special characters or punctuation."
        )

    try:
        if not initial_message_content:
            # If no message content, return without setting a title
            return

        # Add a system message and the initial user message for title generation
        system_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="system",
            content="You are an expert at creating concise titles for chat sessions.",
        )

        # Add the initial user message
        user_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender (placeholder)
            role="user",
            content=initial_message_content,
        )

        # Add request message for title generation
        request_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="user",
            content="Come up with a title for this session based on the user's message.",
        )

        # Build message list
        messages = [system_message, user_message, request_message]

        # Create LLM context
        llm_context = LLMContext(
            messages=messages,
            tools={},  # No tools needed for title generation
            config=LLMConfig(model="gpt-4o-mini", response_format=TitleResponse),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-{str(session.id)}",
                trace_name="FN_title_session",
                trace_id=str(uuid.uuid4()),
                generation_name="FN_title_session",
                trace_metadata=LLMTraceMetadata(
                    session_id=str(session.id),
                ),
            ),
            enable_tracing=False,
        )

        # Generate title using async_prompt
        result = await async_prompt(llm_context)

        # Parse the response
        if hasattr(result, "content") and result.content:
            try:
                # Try to parse as JSON if response_format was used
                import json

                title_data = json.loads(result.content)
                if isinstance(title_data, dict) and "title" in title_data:
                    # session.title = title_data["title"]
                    session.update(title=title_data["title"])
                else:
                    # Fallback to using content directly
                    # session.title = result.content[:30]  # Limit to 30 chars
                    session.update(title=result.content[:30])
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, use content directly
                # session.title = result.content[:30]  # Limit to 30 chars
                session.update(title=result.content[:30])

            # session.save()

    except Exception as e:
        capture_exception(e)
        traceback.print_exc()
        return
