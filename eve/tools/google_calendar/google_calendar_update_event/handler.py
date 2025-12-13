"""
Google Calendar Update Event Handler

Updates an existing event on Google Calendar using PATCH semantics.
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


def build_patch_body(
    title: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    time_zone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the patch body for updating an event.

    Only includes fields that are explicitly provided (not None).
    This allows for true PATCH semantics where only specified fields are updated.
    """
    patch = {}

    if title is not None:
        patch["summary"] = title

    if description is not None:
        patch["description"] = description

    if location is not None:
        patch["location"] = location

    # Handle time updates
    if start_time is not None:
        is_all_day = is_all_day_format(start_time)
        if is_all_day:
            patch["start"] = {"date": start_time}
        else:
            start_obj = {"dateTime": start_time}
            if time_zone:
                start_obj["timeZone"] = time_zone
            patch["start"] = start_obj

    if end_time is not None:
        is_all_day = is_all_day_format(end_time)
        if is_all_day:
            patch["end"] = {"date": end_time}
        else:
            end_obj = {"dateTime": end_time}
            if time_zone:
                end_obj["timeZone"] = time_zone
            patch["end"] = end_obj

    return patch


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    Update an existing event on Google Calendar.

    Uses PATCH semantics - only fields that are explicitly provided will be updated.
    Other fields remain unchanged.

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
            "Please enable 'Allow Write' in your calendar settings to update events."
        )

    # Get required parameters
    event_id = context.args.get("event_id")
    if not event_id:
        raise Exception("event_id is required to identify the event to update.")

    # Get optional parameters (all can be None for PATCH semantics)
    title = context.args.get("title")
    start_time = context.args.get("start_time")
    end_time = context.args.get("end_time")
    description = context.args.get("description")
    location = context.args.get("location")
    time_zone = context.args.get("time_zone") or config.time_zone

    # Build the patch body
    patch_body = build_patch_body(
        title=title,
        start_time=start_time,
        end_time=end_time,
        description=description,
        location=location,
        time_zone=time_zone,
    )

    if not patch_body:
        raise Exception(
            "No fields provided to update. Please specify at least one field to change."
        )

    # Get the calendar service
    try:
        service = await get_calendar_service(secrets)
    except Exception as e:
        logger.error(f"Failed to get Google Calendar service: {e}")
        raise Exception(f"Failed to authenticate with Google Calendar: {e}")

    try:
        # Update the event using PATCH (no notifications sent)
        updated_event = (
            service.events()
            .patch(
                calendarId=config.calendar_id,
                eventId=event_id,
                body=patch_body,
                sendUpdates="none",
            )
            .execute()
        )

        # Build response
        response = {
            "output": {
                "id": updated_event.get("id"),
                "title": updated_event.get("summary"),
                "start": updated_event.get("start", {}).get("dateTime")
                or updated_event.get("start", {}).get("date"),
                "end": updated_event.get("end", {}).get("dateTime")
                or updated_event.get("end", {}).get("date"),
                "calendar_name": config.calendar_name,
                "fields_updated": list(patch_body.keys()),
            }
        }

        if updated_event.get("location"):
            response["output"]["location"] = updated_event.get("location")

        if updated_event.get("description"):
            response["output"]["description"] = updated_event.get("description")

        logger.info(
            f"Updated Google Calendar event (ID: {event_id}) "
            f"fields: {list(patch_body.keys())} for agent {agent_obj.id}"
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
                "Permission denied. You may not have write access to this calendar or event."
            )
        elif e.resp.status == 404:
            raise Exception(
                f"Event not found: {event_id}. "
                "The event may have been deleted or you may not have access to it."
            )
        elif e.resp.status == 410:
            raise Exception(
                f"Event has been deleted: {event_id}. Cannot update a deleted event."
            )
        else:
            raise Exception(error_msg)

    except Exception as e:
        logger.error(f"Failed to update Google Calendar event: {e}")
        raise Exception(f"Failed to update calendar event: {e}")
