import os
from sentry_sdk import capture_exception

from eve.agent.session.models import NotificationConfig


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
                result = response.json()
                return result
            else:
                return {"is_active": False, "redis_available": False}

    except Exception:
        return {"is_active": False, "redis_available": False}


async def create_session_message_notification(
    user_id: str, session_id: str, agent_id: str
):
    """Create a notification for a new session message via the Fastify API"""
    import httpx

    try:
        api_url = os.getenv("EDEN_FASTIFY_API_URL")
        if not api_url:
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
                pass

    except Exception as e:
        capture_exception(e)


async def _send_session_notification(
    notification_config: NotificationConfig,
    session_id: str,
    success: bool = True,
    error: str = None,
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
