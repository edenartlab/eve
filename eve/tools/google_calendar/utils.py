"""
Shared utilities for Google Calendar tools.

Provides formatting helpers to keep LLM context efficient and consistent.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from eve.agent.agent import Agent
from eve.agent.deployments.google_calendar import get_calendar_service
from eve.agent.session.models import Deployment


async def get_calendar_deployment(agent_id: str) -> Deployment:
    """Get Google Calendar deployment for an agent."""
    agent = Agent.from_mongo(agent_id)
    deployment = Deployment.load(agent=agent.id, platform="google_calendar")
    if not deployment:
        raise Exception("No valid Google Calendar deployment found for this agent")
    return deployment


async def get_service_and_config(agent_id: str):
    """Get calendar service and config from deployment."""
    deployment = await get_calendar_deployment(agent_id)
    secrets = deployment.secrets.google_calendar
    config = deployment.config.google_calendar

    if not secrets:
        raise Exception("Google Calendar credentials not configured")
    if not config:
        raise Exception("Google Calendar settings not configured")

    service = await get_calendar_service(secrets)
    return service, config, deployment


def parse_datetime(dt_str: Optional[str], default_tz: str = "UTC") -> Optional[datetime]:
    """
    Parse a datetime string to a datetime object.
    Handles ISO format and common natural language relative times.
    """
    if not dt_str:
        return None

    # Handle relative time expressions
    now = datetime.now(timezone.utc)
    dt_lower = dt_str.lower().strip()

    if dt_lower == "now":
        return now
    elif dt_lower == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif dt_lower == "tomorrow":
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif dt_lower == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Try ISO format parsing
    try:
        # Handle 'Z' suffix for UTC
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        return datetime.fromisoformat(dt_str)
    except ValueError:
        pass

    # Try parsing without timezone (assume UTC)
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', ''))
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Unable to parse datetime: {dt_str}. Use ISO format (YYYY-MM-DDTHH:MM:SS) or relative terms like 'now', 'today', 'tomorrow'")


def format_datetime_for_display(dt_str: Optional[str], include_date: bool = True) -> str:
    """
    Format datetime string for concise LLM display.
    Examples: "Dec 15, 2:30 PM" or "2:30 PM"
    """
    if not dt_str:
        return "Unknown"

    try:
        # Handle Google's datetime format
        if 'T' in dt_str:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        else:
            # Date-only event
            dt = datetime.fromisoformat(dt_str)
            if include_date:
                return dt.strftime("%b %d")
            return "All day"

        if include_date:
            return dt.strftime("%b %d, %I:%M %p").replace(" 0", " ").strip()
        return dt.strftime("%I:%M %p").replace(" 0", " ").strip()
    except Exception:
        return dt_str


def format_event_compact(
    event: Dict[str, Any],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Format a Google Calendar event for compact LLM context.

    Default fields: id, title, start, end, location (if present)
    """
    default_fields = ["id", "title", "start", "end", "location", "description", "attendees", "status"]

    if include_fields:
        fields_to_include = include_fields
    elif exclude_fields:
        fields_to_include = [f for f in default_fields if f not in exclude_fields]
    else:
        # Default: minimal essential fields
        fields_to_include = ["id", "title", "start", "end"]

    formatted = {}

    # Always include ID for reference
    formatted["id"] = event.get("id", "")

    if "title" in fields_to_include or "summary" in fields_to_include:
        formatted["title"] = event.get("summary", "Untitled")

    if "start" in fields_to_include:
        start = event.get("start", {})
        start_str = start.get("dateTime") or start.get("date")
        formatted["start"] = format_datetime_for_display(start_str)

    if "end" in fields_to_include:
        end = event.get("end", {})
        end_str = end.get("dateTime") or end.get("date")
        formatted["end"] = format_datetime_for_display(end_str, include_date=False)

    if "location" in fields_to_include and event.get("location"):
        formatted["location"] = event.get("location")

    if "description" in fields_to_include and event.get("description"):
        desc = event.get("description", "")
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:197] + "..."
        formatted["description"] = desc

    if "attendees" in fields_to_include and event.get("attendees"):
        attendees = event.get("attendees", [])
        # Format as simple list of names/emails
        formatted["attendees"] = [
            a.get("displayName") or a.get("email", "Unknown")
            for a in attendees[:10]  # Limit to 10 attendees
        ]
        if len(attendees) > 10:
            formatted["attendees"].append(f"+{len(attendees) - 10} more")

    if "status" in fields_to_include and event.get("status"):
        status = event.get("status")
        if status != "confirmed":  # Only show if not the default
            formatted["status"] = status

    if "recurring" in fields_to_include and event.get("recurringEventId"):
        formatted["recurring"] = True

    if "url" in fields_to_include or "link" in fields_to_include:
        formatted["url"] = event.get("htmlLink", "")

    return formatted


def format_events_list(
    events: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    max_events: int = 20,
) -> List[Dict[str, Any]]:
    """Format a list of events for LLM context."""
    formatted = []
    for event in events[:max_events]:
        formatted.append(format_event_compact(event, include_fields, exclude_fields))

    return formatted


def format_duration(delta: timedelta) -> str:
    """Format a timedelta as a human-readable duration."""
    total_minutes = int(delta.total_seconds() / 60)

    if total_minutes < 60:
        return f"{total_minutes}min"

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}min"


def check_permissions(config, require_write: bool = False, require_delete: bool = False):
    """Check if the deployment has required permissions."""
    if require_write and not config.allow_write:
        raise Exception(
            "Write access not enabled for this Google Calendar integration. "
            "Contact the agent owner to enable write permissions."
        )
    if require_delete and not config.allow_delete:
        raise Exception(
            "Delete access not enabled for this Google Calendar integration. "
            "Contact the agent owner to enable delete permissions."
        )
