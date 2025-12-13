"""
Google Calendar List Events Handler

Retrieves events from a Google Calendar within a specified time range.
Recurring events are bundled together with their recurrence schedule.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from googleapiclient.errors import HttpError
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.deployments.google_calendar import get_calendar_service
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


def parse_recurrence_rule(recurrence: List[str]) -> Optional[str]:
    """Parse RRULE into human-readable schedule description."""
    if not recurrence:
        return None

    for rule in recurrence:
        if rule.startswith("RRULE:"):
            rrule = rule[6:]  # Remove "RRULE:" prefix
            parts = dict(p.split("=") for p in rrule.split(";") if "=" in p)

            freq = parts.get("FREQ", "").lower()
            interval = int(parts.get("INTERVAL", 1))
            byday = parts.get("BYDAY", "")

            # Build human-readable description
            if freq == "daily":
                if interval == 1:
                    return "Daily"
                return f"Every {interval} days"
            elif freq == "weekly":
                days_map = {
                    "MO": "Mon",
                    "TU": "Tue",
                    "WE": "Wed",
                    "TH": "Thu",
                    "FR": "Fri",
                    "SA": "Sat",
                    "SU": "Sun",
                }
                if byday:
                    days = [days_map.get(d, d) for d in byday.split(",")]
                    days_str = ", ".join(days)
                    if interval == 1:
                        return f"Weekly on {days_str}"
                    return f"Every {interval} weeks on {days_str}"
                if interval == 1:
                    return "Weekly"
                return f"Every {interval} weeks"
            elif freq == "monthly":
                if interval == 1:
                    return "Monthly"
                return f"Every {interval} months"
            elif freq == "yearly":
                if interval == 1:
                    return "Yearly"
                return f"Every {interval} years"

    return None


def format_event(
    event: Dict[str, Any], instances: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Format a Google Calendar event into a clean structure."""
    start = event.get("start", {})
    end = event.get("end", {})

    start_time = start.get("dateTime") or start.get("date")
    end_time = end.get("dateTime") or end.get("date")
    is_all_day = "date" in start and "dateTime" not in start

    result = {
        "id": event.get("id"),
        "title": event.get("summary", "(No title)"),
        "description": event.get("description"),
        "location": event.get("location"),
        "start": start_time,
        "end": end_time,
        "is_all_day": is_all_day,
        "time_zone": start.get("timeZone") or end.get("timeZone"),
        "creator": event.get("creator", {}).get("email"),
    }

    # Add recurrence info if this is a recurring event
    recurrence = event.get("recurrence")
    if recurrence:
        result["is_recurring"] = True
        result["schedule"] = parse_recurrence_rule(recurrence)

        # Add next instance if we have instances
        if instances:
            # Find the next upcoming instance
            now = datetime.now().isoformat()
            upcoming = [
                inst
                for inst in instances
                if (inst.get("start", {}).get("dateTime") or inst.get("start", {}).get("date", ""))
                >= now
            ]
            if upcoming:
                next_inst = upcoming[0]
                next_start = next_inst.get("start", {})
                result["next_occurrence"] = next_start.get("dateTime") or next_start.get("date")
    else:
        result["is_recurring"] = False

    return result


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    List events from Google Calendar within a specified time range.

    Returns events sorted by start time (oldest first).
    Recurring events are bundled with their schedule and next occurrence.
    """
    if not context.agent:
        raise Exception("Agent is required")

    agent_obj = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="google_calendar")

    if not deployment:
        raise Exception(
            "No Google Calendar deployment found. Please connect your Google Calendar first."
        )

    if not deployment.secrets or not deployment.secrets.google_calendar:
        raise Exception("Google Calendar credentials not found in deployment.")

    if not deployment.config or not deployment.config.google_calendar:
        raise Exception("Google Calendar configuration not found in deployment.")

    secrets = deployment.secrets.google_calendar
    config = deployment.config.google_calendar

    # Get the calendar service
    try:
        service = await get_calendar_service(secrets)
    except Exception as e:
        logger.error(f"Failed to get Google Calendar service: {e}")
        raise Exception(f"Failed to authenticate with Google Calendar: {e}")

    # Build the request parameters
    time_min = context.args.get("time_min")
    time_max = context.args.get("time_max")
    max_results = min(context.args.get("max_results", 50), 250)

    # Validate required parameters
    if not time_min or not time_max:
        raise Exception("Both time_min and time_max are required parameters.")

    # Build request kwargs - get expanded instances sorted by start time
    request_kwargs = {
        "calendarId": config.calendar_id,
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": max_results,
        "singleEvents": True,  # Expand recurring events into instances
        "orderBy": "startTime",  # Always oldest first
    }

    # Use calendar's timezone if available
    if config.time_zone:
        request_kwargs["timeZone"] = config.time_zone

    try:
        # Execute the request
        events_result = service.events().list(**request_kwargs).execute()
        events = events_result.get("items", [])

        # Group events by recurring event ID to bundle recurring events
        recurring_groups: Dict[str, List[Dict[str, Any]]] = {}
        standalone_events: List[Dict[str, Any]] = []

        for event in events:
            recurring_id = event.get("recurringEventId")
            if recurring_id:
                if recurring_id not in recurring_groups:
                    recurring_groups[recurring_id] = []
                recurring_groups[recurring_id].append(event)
            else:
                standalone_events.append(event)

        # Format standalone events
        formatted_events = [format_event(event) for event in standalone_events]

        # Format recurring event groups - use first instance as base, include schedule
        for recurring_id, instances in recurring_groups.items():
            if instances:
                # Sort instances by start time
                instances.sort(
                    key=lambda x: x.get("start", {}).get("dateTime")
                    or x.get("start", {}).get("date", "")
                )

                # Get the master event to access recurrence rules
                try:
                    master_event = (
                        service.events()
                        .get(calendarId=config.calendar_id, eventId=recurring_id)
                        .execute()
                    )
                    # Use master event info but with instances for next occurrence
                    formatted = format_event(master_event, instances)
                    # Use the first instance's times as the "template" times
                    first_inst = instances[0]
                    formatted["start"] = first_inst.get("start", {}).get(
                        "dateTime"
                    ) or first_inst.get("start", {}).get("date")
                    formatted["end"] = first_inst.get("end", {}).get(
                        "dateTime"
                    ) or first_inst.get("end", {}).get("date")
                    formatted["next_occurrence"] = formatted["start"]
                    formatted["occurrences_in_range"] = len(instances)
                    formatted_events.append(formatted)
                except HttpError:
                    # If we can't get master, just use first instance
                    formatted = format_event(instances[0])
                    formatted["is_recurring"] = True
                    formatted["occurrences_in_range"] = len(instances)
                    formatted_events.append(formatted)

        # Sort all events by start time
        formatted_events.sort(key=lambda x: x.get("start", ""))

        # Build response
        response = {
            "output": {
                "calendar_name": config.calendar_name,
                "time_range": {"start": time_min, "end": time_max},
                "total_events": len(formatted_events),
                "events": formatted_events,
            }
        }

        logger.info(
            f"Listed {len(formatted_events)} events from Google Calendar "
            f"for agent {agent_obj.id}"
        )

        return response

    except HttpError as e:
        error_msg = f"Google Calendar API error: {e.reason}"
        logger.error(f"{error_msg} - Status: {e.resp.status}")

        if e.resp.status == 401:
            raise Exception(
                "Google Calendar authentication expired. Please reconnect your account."
            )
        elif e.resp.status == 404:
            raise Exception(
                f"Calendar not found: {config.calendar_id}. "
                "Please check your calendar settings."
            )
        else:
            raise Exception(error_msg)

    except Exception as e:
        logger.error(f"Failed to list Google Calendar events: {e}")
        raise Exception(f"Failed to retrieve calendar events: {e}")
