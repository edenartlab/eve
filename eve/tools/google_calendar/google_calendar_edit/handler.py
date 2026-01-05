"""
Handler for google_calendar_edit tool.

Consolidated write operations: create and update events.
"""

import asyncio
from typing import Any, Dict, List, Optional

from googleapiclient.errors import HttpError
from loguru import logger

from eve.tool import ToolContext
from eve.tools.google_calendar.utils import (
    check_permissions,
    format_datetime_for_display,
    get_service_and_config,
    parse_datetime,
)


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    Create or update Google Calendar events.
    """
    if not context.agent:
        raise Exception("Agent is required")

    action = context.args.get("action")
    if not action:
        raise Exception("action is required (create or update)")

    # Get service and config
    service, config, deployment = await get_service_and_config(context.agent)

    # Check write permission
    check_permissions(config, require_write=True)

    calendar_id = config.calendar_id

    if action == "create":
        return await _create_event(context, service, calendar_id, config)
    elif action == "update":
        return await _update_event(context, service, calendar_id, config)
    else:
        raise Exception(f"Unknown action: {action}. Use create or update.")


async def _execute(request):
    return await asyncio.to_thread(request.execute)


async def _create_event(
    context: ToolContext,
    service,
    calendar_id: str,
    config,
) -> Dict[str, Any]:
    """Create a new event."""
    title = context.args.get("title")
    start_time_str = context.args.get("start_time")
    end_time_str = context.args.get("end_time")
    description = context.args.get("description")
    location = context.args.get("location")
    attendees = context.args.get("attendees", [])
    send_notifications = context.args.get("send_notifications", True)
    reminder_minutes = context.args.get("reminder_minutes", 30)
    all_day = context.args.get("all_day", False)
    check_conflicts = context.args.get("check_conflicts", True)

    # Validate required
    if not title:
        raise Exception("title is required for action=create")
    if not start_time_str:
        raise Exception("start_time is required for action=create")
    if not end_time_str:
        raise Exception("end_time is required for action=create")

    # Build event body
    event_body = {"summary": title}

    if all_day:
        event_body["start"] = {"date": start_time_str[:10]}
        event_body["end"] = {"date": end_time_str[:10]}
    else:
        start_dt = parse_datetime(start_time_str)
        end_dt = parse_datetime(end_time_str)

        if not start_dt or not end_dt:
            raise Exception(
                "Invalid datetime format. Use ISO format: 2024-12-15T14:00:00"
            )
        if end_dt <= start_dt:
            raise Exception("end_time must be after start_time")

        tz = config.time_zone or "UTC"
        event_body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": tz}
        event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": tz}

    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location
    if attendees:
        event_body["attendees"] = [{"email": email} for email in attendees]

    if reminder_minutes > 0:
        event_body["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": reminder_minutes}],
        }
    else:
        event_body["reminders"] = {"useDefault": False, "overrides": []}

    # Check conflicts
    if check_conflicts and not all_day:
        conflicts = await _check_time_conflicts(
            service,
            calendar_id,
            event_body["start"]["dateTime"],
            event_body["end"]["dateTime"],
            exclude_event_id=None,
        )
        if conflicts:
            return {
                "output": {
                    "status": "conflict",
                    "message": f"Found {len(conflicts)} conflicting event(s) in this time slot.",
                    "conflicts": conflicts,
                    "suggestion": "Use find_free_slots to find available time, or set check_conflicts=false.",
                }
            }

    # Create event
    try:
        created = await _execute(
            service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                sendNotifications=send_notifications,
            )
        )
        logger.info(
            f"Created event: {created.get('summary')} (ID: {created.get('id')})"
        )
    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        elif e.resp.status == 403:
            raise Exception(
                "Permission denied. Calendar integration may not have write access."
            )
        else:
            raise Exception(f"Failed to create event: {str(e)}")

    start = created.get("start", {})
    end = created.get("end", {})
    start_str = start.get("dateTime") or start.get("date")
    end_str = end.get("dateTime") or end.get("date")

    response = {
        "status": "created",
        "event": {
            "id": created.get("id"),
            "title": created.get("summary"),
            "start": format_datetime_for_display(start_str),
            "end": format_datetime_for_display(end_str),
            "url": created.get("htmlLink"),
        },
    }

    if location:
        response["event"]["location"] = location
    if attendees:
        response["event"]["attendees_invited"] = len(attendees)

    return {"output": response}


async def _update_event(
    context: ToolContext,
    service,
    calendar_id: str,
    config,
) -> Dict[str, Any]:
    """Update an existing event."""
    event_id = context.args.get("event_id")
    if not event_id:
        raise Exception("event_id is required for action=update")

    title = context.args.get("title")
    start_time_str = context.args.get("start_time")
    end_time_str = context.args.get("end_time")
    description = context.args.get("description")
    location = context.args.get("location")
    attendees = context.args.get("attendees")
    add_attendees = context.args.get("add_attendees", [])
    remove_attendees = context.args.get("remove_attendees", [])
    send_notifications = context.args.get("send_notifications", True)
    check_conflicts = context.args.get("check_conflicts", True)

    # Fetch existing event
    try:
        existing = await _execute(
            service.events().get(calendarId=calendar_id, eventId=event_id)
        )
    except HttpError as e:
        if e.resp.status == 404:
            raise Exception(f"Event not found: {event_id}")
        raise Exception(f"Failed to fetch event: {str(e)}")

    changes = []
    original_start = existing.get("start", {})
    original_end = existing.get("end", {})
    is_all_day = "date" in original_start and "dateTime" not in original_start

    # Handle title
    if title is not None:
        existing["summary"] = title
        changes.append(f"title → '{title}'")

    # Handle time changes
    time_changed = False
    if start_time_str:
        new_start_dt = parse_datetime(start_time_str)
        if not new_start_dt:
            raise Exception(
                "Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )
        tz = original_start.get("timeZone") or config.time_zone or "UTC"
        existing["start"] = {"dateTime": new_start_dt.isoformat(), "timeZone": tz}
        changes.append(
            f"start → {format_datetime_for_display(new_start_dt.isoformat())}"
        )
        time_changed = True

    if end_time_str:
        new_end_dt = parse_datetime(end_time_str)
        if not new_end_dt:
            raise Exception(
                "Invalid end_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )
        tz = original_end.get("timeZone") or config.time_zone or "UTC"
        existing["end"] = {"dateTime": new_end_dt.isoformat(), "timeZone": tz}
        changes.append(f"end → {format_datetime_for_display(new_end_dt.isoformat())}")
        time_changed = True

    # Check conflicts if time changed
    if time_changed and check_conflicts and not is_all_day:
        conflicts = await _check_time_conflicts(
            service,
            calendar_id,
            existing["start"]["dateTime"],
            existing["end"]["dateTime"],
            exclude_event_id=event_id,
        )
        if conflicts:
            return {
                "output": {
                    "status": "conflict",
                    "message": f"Cannot move event - {len(conflicts)} conflicting event(s).",
                    "conflicts": conflicts,
                    "suggestion": "Choose different time or set check_conflicts=false.",
                }
            }

    # Handle description
    if description is not None:
        if description == "":
            existing.pop("description", None)
            changes.append("description cleared")
        else:
            existing["description"] = description
            changes.append("description updated")

    # Handle location
    if location is not None:
        if location == "":
            existing.pop("location", None)
            changes.append("location cleared")
        else:
            existing["location"] = location
            changes.append(f"location → '{location}'")

    # Handle attendees
    current_attendees = existing.get("attendees", [])
    current_emails = {a.get("email", "").lower() for a in current_attendees}

    if attendees is not None:
        existing["attendees"] = [{"email": email} for email in attendees]
        changes.append(f"attendees set ({len(attendees)})")
    else:
        if add_attendees:
            for email in add_attendees:
                if email.lower() not in current_emails:
                    current_attendees.append({"email": email})
                    changes.append(f"added {email}")
            existing["attendees"] = current_attendees

        if remove_attendees:
            remove_set = {e.lower() for e in remove_attendees}
            existing["attendees"] = [
                a
                for a in current_attendees
                if a.get("email", "").lower() not in remove_set
            ]
            changes.append(f"removed {len(remove_attendees)} attendee(s)")

    if not changes:
        return {
            "output": {
                "status": "no_changes",
                "message": "No changes specified.",
                "event_id": event_id,
            }
        }

    # Update
    try:
        updated = await _execute(
            service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=existing,
                sendNotifications=send_notifications,
            )
        )
        logger.info(f"Updated event: {updated.get('summary')} (ID: {event_id})")
    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        elif e.resp.status == 403:
            raise Exception("Permission denied to edit this event.")
        else:
            raise Exception(f"Failed to update event: {str(e)}")

    start = updated.get("start", {})
    end = updated.get("end", {})
    start_str = start.get("dateTime") or start.get("date")
    end_str = end.get("dateTime") or end.get("date")

    return {
        "output": {
            "status": "updated",
            "changes": changes,
            "event": {
                "id": updated.get("id"),
                "title": updated.get("summary"),
                "start": format_datetime_for_display(start_str),
                "end": format_datetime_for_display(end_str),
            },
        }
    }


async def _check_time_conflicts(
    service,
    calendar_id: str,
    start_time: str,
    end_time: str,
    exclude_event_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Check for conflicting events in a time range."""
    try:
        result = await _execute(
            service.events().list(
                calendarId=calendar_id,
                timeMin=start_time,
                timeMax=end_time,
                singleEvents=True,
                orderBy="startTime",
            )
        )

        events = result.get("items", [])
        if exclude_event_id:
            events = [e for e in events if e.get("id") != exclude_event_id]

        if not events:
            return []

        return [
            {
                "id": e.get("id"),
                "title": e.get("summary", "Untitled"),
                "start": format_datetime_for_display(
                    e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
                ),
            }
            for e in events[:5]
        ]
    except HttpError:
        return []  # Fail open - let the create/update proceed
