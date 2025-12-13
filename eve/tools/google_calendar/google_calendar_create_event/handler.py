"""
Google Calendar Create Event Handler

Creates a new event on Google Calendar.
"""

from typing import Any, Dict, Optional

from googleapiclient.errors import HttpError
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.deployments.google_calendar import get_calendar_service
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


def is_all_day_format(time_str: str) -> bool:
    """Check if a time string is in all-day format (YYYY-MM-DD) vs datetime format."""
    return len(time_str) == 10 and "T" not in time_str


def build_event_body(
    title: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    time_zone: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the event body for the Google Calendar API."""
    event = {
        "summary": title,
    }

    # Handle all-day vs timed events
    is_all_day = is_all_day_format(start_time)

    if is_all_day:
        event["start"] = {"date": start_time}
        event["end"] = {"date": end_time}
    else:
        start_obj = {"dateTime": start_time}
        end_obj = {"dateTime": end_time}

        if time_zone:
            start_obj["timeZone"] = time_zone
            end_obj["timeZone"] = time_zone

        event["start"] = start_obj
        event["end"] = end_obj

    if description:
        event["description"] = description

    if location:
        event["location"] = location

    return event


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    Create a new event on Google Calendar.

    Requires the deployment to have allow_write=True in its configuration.
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

    # Check write permission
    if not config.allow_write:
        raise Exception(
            "Write access is not enabled for this Google Calendar deployment. "
            "Please enable 'Allow Write' in your calendar settings to create events."
        )

    # Get required parameters
    title = context.args.get("title")
    start_time = context.args.get("start_time")
    end_time = context.args.get("end_time")

    if not title:
        raise Exception("Event title is required.")
    if not start_time:
        raise Exception("Event start_time is required.")
    if not end_time:
        raise Exception("Event end_time is required.")

    # Get optional parameters
    description = context.args.get("description")
    location = context.args.get("location")
    time_zone = context.args.get("time_zone") or config.time_zone

    # Get the calendar service
    try:
        service = await get_calendar_service(secrets)
    except Exception as e:
        logger.error(f"Failed to get Google Calendar service: {e}")
        raise Exception(f"Failed to authenticate with Google Calendar: {e}")

    # Build the event body
    event_body = build_event_body(
        title=title,
        start_time=start_time,
        end_time=end_time,
        description=description,
        location=location,
        time_zone=time_zone,
    )

    try:
        # Create the event (no notifications sent)
        created_event = (
            service.events()
            .insert(
                calendarId=config.calendar_id,
                body=event_body,
                sendUpdates="none",
            )
            .execute()
        )

        # Build response
        response = {
            "output": {
                "id": created_event.get("id"),
                "title": created_event.get("summary"),
                "start": created_event.get("start", {}).get("dateTime")
                or created_event.get("start", {}).get("date"),
                "end": created_event.get("end", {}).get("dateTime")
                or created_event.get("end", {}).get("date"),
                "calendar_name": config.calendar_name,
            }
        }

        if location:
            response["output"]["location"] = location

        if description:
            response["output"]["description"] = description

        logger.info(
            f"Created Google Calendar event '{title}' (ID: {created_event.get('id')}) "
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
        elif e.resp.status == 403:
            raise Exception(
                "Permission denied. You may not have write access to this calendar."
            )
        elif e.resp.status == 404:
            raise Exception(
                f"Calendar not found: {config.calendar_id}. "
                "Please check your calendar settings."
            )
        else:
            raise Exception(error_msg)

    except Exception as e:
        logger.error(f"Failed to create Google Calendar event: {e}")
        raise Exception(f"Failed to create calendar event: {e}")
