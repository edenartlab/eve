import os
from typing import Optional

from sentry_sdk import capture_exception

from eve.agent import Agent
from eve.agent.session.models import NotificationConfig, Session


async def check_if_session_active(user_id: str, session_id: str) -> dict:
    """Check if user is actively viewing a session via the API"""
    import httpx

    try:
        api_url = os.getenv("EDEN_FASTIFY_API_URL")
        if not api_url:
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
                return response.json()
            else:
                return {"is_active": False, "redis_available": False}

    except Exception:
        return {"is_active": False, "redis_available": False}


def _format_session_message_title(
    agent_name: Optional[str], session_title: Optional[str]
) -> str:
    clean_agent_name = agent_name.strip() if agent_name else None
    clean_session_title = session_title.strip() if session_title else None
    if clean_agent_name and clean_session_title:
        return f"{clean_agent_name} ({clean_session_title})"
    if clean_agent_name:
        return clean_agent_name
    if clean_session_title:
        return clean_session_title
    return "New message"


def _format_session_message_body(message: Optional[str]) -> str:
    if not message:
        return "You have a new message"
    snippet = " ".join(message.split()).strip()
    if not snippet:
        return "You have a new message"
    return f"{snippet[:140]}…" if len(snippet) > 140 else snippet


async def create_session_message_notification(
    user_id: str,
    session_id: str,
    agent_id: Optional[str],
    message: Optional[str] = None,
):
    """Create a notification for a new session message via the Fastify API"""
    import httpx

    try:
        api_url = os.getenv("EDEN_FASTIFY_API_URL")
        if not api_url:
            return

        agent_name = None
        session_title = None
        if session_id:
            try:
                session = Session.from_mongo(session_id)
                session_title = session.title
            except Exception:
                session_title = None

        if agent_id:
            try:
                agent = Agent.from_mongo(agent_id)
                agent_name = agent.name or agent.username
            except Exception:
                agent_name = None

        notification_data = {
            "user_id": user_id,
            "type": "session_message",
            "title": _format_session_message_title(agent_name, session_title),
            "message": _format_session_message_body(message),
            "priority": "normal",
            "session_id": session_id,
            "action_url": f"/sessions/{session_id}",
            "channels": ["in_app", "push"],
        }
        if agent_id:
            notification_data["agent_id"] = agent_id

        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": os.getenv("EDEN_FASTIFY_ADMIN_KEY"),
                "Content-Type": "application/json",
            }
            await client.post(
                f"{api_url}/v2/notifications",
                headers=headers,
                json=notification_data,
                timeout=5.0,
            )

            # Ignore non-200 silently; Fastify handles auth/validation

    except Exception as e:
        capture_exception(e)


async def _send_session_notification(
    notification_config: NotificationConfig,
    session: Session,
    success: bool = True,
    error: str = None,
):
    """Send a notification about session completion"""
    from datetime import datetime, timezone

    import httpx

    try:
        api_url = os.getenv("EDEN_API_URL")
        if not api_url:
            return

        # Extract session_id from Session object
        session_id = str(session.id)

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
            "session_id": session_id,
            "action_url": f"/sessions/{session_id}",
            "metadata": {
                "session_id": session_id,
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
                pass

    except Exception as e:
        capture_exception(e)
