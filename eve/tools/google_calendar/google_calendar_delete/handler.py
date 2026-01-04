"""
Handler for google_calendar_delete tool.

Deletes/cancels an event from Google Calendar.
"""

import asyncio
from typing import Any, Dict

from googleapiclient.errors import HttpError
from loguru import logger

from eve.tool import ToolContext
from eve.tools.google_calendar.utils import (
    check_permissions,
    format_datetime_for_display,
    get_service_and_config,
)


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    Delete an event from Google Calendar.
    """
    if not context.agent:
        raise Exception("Agent is required")

    async def _execute(request):
        return await asyncio.to_thread(request.execute)

    # Get service and config
    service, config, deployment = await get_service_and_config(context.agent)

    # Check delete permission
    check_permissions(config, require_delete=True)

    calendar_id = config.calendar_id

    # Extract parameters
    event_id = context.args.get("event_id")
    if not event_id:
        raise Exception("event_id is required")

    send_notifications = context.args.get("send_notifications", True)
    confirm_title = context.args.get("confirm_title")

    # First, fetch the event to confirm what we're deleting
    try:
        event = await _execute(
            service.events().get(calendarId=calendar_id, eventId=event_id)
        )
    except HttpError as e:
        if e.resp.status == 404:
            raise Exception(
                f"Event not found: {event_id}. It may have already been deleted."
            )
        raise Exception(f"Failed to fetch event: {str(e)}")

    event_title = event.get("summary", "Untitled")
    start = event.get("start", {})
    start_str = start.get("dateTime") or start.get("date")

    # Safety check if confirm_title is provided
    if confirm_title:
        if confirm_title.lower() not in event_title.lower():
            return {
                "output": {
                    "status": "aborted",
                    "message": f"Safety check failed. Expected title containing '{confirm_title}' but found '{event_title}'.",
                    "event": {
                        "id": event_id,
                        "title": event_title,
                        "start": format_datetime_for_display(start_str),
                    },
                    "suggestion": "Verify the event_id is correct or remove confirm_title parameter.",
                }
            }

    # Delete the event
    try:
        await _execute(
            service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendNotifications=send_notifications,
            )
        )

        logger.info(f"Deleted event: {event_title} (ID: {event_id})")

    except HttpError as e:
        if e.resp.status == 401:
            raise Exception(
                "Google Calendar authentication expired. Please reconnect your account."
            )
        elif e.resp.status == 403:
            raise Exception(
                "Permission denied. You may not have permission to delete this event."
            )
        elif e.resp.status == 410:
            # Event was already deleted
            return {
                "output": {
                    "status": "already_deleted",
                    "message": "This event was already deleted.",
                    "event_id": event_id,
                }
            }
        else:
            raise Exception(f"Failed to delete event: {str(e)}")

    # Format response with deleted event info
    response = {
        "status": "deleted",
        "message": f"Successfully cancelled '{event_title}'",
        "deleted_event": {
            "id": event_id,
            "title": event_title,
            "start": format_datetime_for_display(start_str),
        },
    }

    if event.get("attendees"):
        attendee_count = len(event.get("attendees", []))
        response["attendees_notified"] = attendee_count if send_notifications else 0

    return {"output": response}
