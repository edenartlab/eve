"""
Handler for google_calendar_query tool.

Consolidated read-only operations: list events, get event details, find free slots.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from googleapiclient.errors import HttpError
from loguru import logger

from eve.tool import ToolContext
from eve.tools.google_calendar.utils import (
    format_datetime_for_display,
    format_duration,
    format_event_compact,
    format_events_list,
    get_service_and_config,
    parse_datetime,
)


async def handler(context: ToolContext) -> Dict[str, Any]:
    """
    Query Google Calendar - list, get, or find_free_slots.
    """
    if not context.agent:
        raise Exception("Agent is required")

    action = context.args.get("action")
    if not action:
        raise Exception("action is required (list, get, or find_free_slots)")

    # Get service and config
    service, config, deployment = await get_service_and_config(context.agent)
    calendar_id = config.calendar_id

    if action == "list":
        return await _list_events(context, service, calendar_id, config)
    elif action == "get":
        return await _get_event(context, service, calendar_id)
    elif action == "find_free_slots":
        return await _find_free_slots(context, service, calendar_id, config)
    else:
        raise Exception(f"Unknown action: {action}. Use list, get, or find_free_slots.")


async def _list_events(
    context: ToolContext,
    service,
    calendar_id: str,
    config,
) -> Dict[str, Any]:
    """List events in a time range."""
    time_min_str = context.args.get("time_min")
    time_max_str = context.args.get("time_max")
    days_ahead = context.args.get("days_ahead", 7)
    query = context.args.get("query")
    max_results = min(context.args.get("max_results", 10), 50)
    include_fields = context.args.get("include_fields")

    # Calculate time range
    now = datetime.now(timezone.utc)
    time_min = parse_datetime(time_min_str) if time_min_str else now

    if time_max_str:
        time_max = parse_datetime(time_max_str)
    else:
        time_max = time_min + timedelta(days=days_ahead)

    try:
        request_params = {
            "calendarId": calendar_id,
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
            "showDeleted": False,
        }

        if query:
            request_params["q"] = query

        events_result = service.events().list(**request_params).execute()
        events = events_result.get("items", [])

        logger.info(f"Retrieved {len(events)} events from Google Calendar")

    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        elif e.resp.status == 404:
            raise Exception(f"Calendar not found: {calendar_id}")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

    formatted_events = format_events_list(events, include_fields=include_fields, max_events=max_results)

    if not formatted_events:
        return {
            "output": {
                "message": "No events found in the specified time range.",
                "time_range": f"{time_min.strftime('%b %d')} - {time_max.strftime('%b %d, %Y')}",
                "events": [],
            }
        }

    return {
        "output": {
            "count": len(formatted_events),
            "time_range": f"{time_min.strftime('%b %d')} - {time_max.strftime('%b %d, %Y')}",
            "events": formatted_events,
        }
    }


async def _get_event(
    context: ToolContext,
    service,
    calendar_id: str,
) -> Dict[str, Any]:
    """Get detailed information about a specific event."""
    event_id = context.args.get("event_id")
    if not event_id:
        raise Exception("event_id is required for action=get")

    include_fields = context.args.get("include_fields")

    try:
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        logger.info(f"Retrieved event: {event.get('summary', 'Untitled')}")
    except HttpError as e:
        if e.resp.status == 404:
            raise Exception(f"Event not found: {event_id}")
        elif e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

    # Format with all details by default for get
    default_fields = ["id", "title", "start", "end", "location", "description", "attendees", "status", "url", "conference_link"]
    fields = include_fields if include_fields else default_fields

    formatted = {"id": event.get("id", "")}

    if "title" in fields:
        formatted["title"] = event.get("summary", "Untitled")

    if "start" in fields:
        start = event.get("start", {})
        start_str = start.get("dateTime") or start.get("date")
        formatted["start"] = format_datetime_for_display(start_str)
        if start.get("dateTime"):
            formatted["start_iso"] = start.get("dateTime")

    if "end" in fields:
        end = event.get("end", {})
        end_str = end.get("dateTime") or end.get("date")
        formatted["end"] = format_datetime_for_display(end_str)

    if "location" in fields and event.get("location"):
        formatted["location"] = event.get("location")

    if "description" in fields and event.get("description"):
        formatted["description"] = event.get("description")

    if "attendees" in fields and event.get("attendees"):
        formatted["attendees"] = [
            {
                "name": a.get("displayName") or a.get("email", "Unknown"),
                "response": a.get("responseStatus", "needsAction"),
            }
            for a in event.get("attendees", [])[:15]
        ]

    if "status" in fields:
        formatted["status"] = event.get("status", "confirmed")

    if "url" in fields:
        formatted["url"] = event.get("htmlLink", "")

    if "conference_link" in fields:
        conference = event.get("conferenceData")
        if conference:
            for ep in conference.get("entryPoints", []):
                if ep.get("entryPointType") == "video":
                    formatted["conference_link"] = ep.get("uri")
                    break

    return {"output": formatted}


async def _find_free_slots(
    context: ToolContext,
    service,
    calendar_id: str,
    config,
) -> Dict[str, Any]:
    """Find available time slots."""
    duration_minutes = context.args.get("duration_minutes")
    if not duration_minutes:
        raise Exception("duration_minutes is required for action=find_free_slots")

    time_min_str = context.args.get("time_min")
    time_max_str = context.args.get("time_max")
    days_ahead = context.args.get("days_ahead", 7)
    earliest_hour = context.args.get("earliest_hour", 9)
    latest_hour = context.args.get("latest_hour", 17)
    working_days_only = context.args.get("working_days_only", True)
    max_slots = min(context.args.get("max_results", 5), 20)

    if earliest_hour > latest_hour:
        raise Exception(f"earliest_hour ({earliest_hour}) must be <= latest_hour ({latest_hour})")

    # Calculate time range
    now = datetime.now(timezone.utc)
    time_min = parse_datetime(time_min_str) if time_min_str else now
    if time_min < now:
        time_min = now

    time_max = parse_datetime(time_max_str) if time_max_str else time_min + timedelta(days=days_ahead)

    # Fetch events
    try:
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = events_result.get("items", [])
    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

    # Parse busy periods
    busy_periods = []
    for event in events:
        start = event.get("start", {})
        end = event.get("end", {})
        start_str = start.get("dateTime")
        if not start_str:
            # All-day event
            date_str = start.get("date")
            if date_str:
                start_dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                end_dt = start_dt + timedelta(days=1)
                busy_periods.append((start_dt, end_dt))
            continue
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end_str = end.get("dateTime")
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else start_dt + timedelta(hours=1)
        busy_periods.append((start_dt, end_dt))

    busy_periods.sort(key=lambda x: x[0])

    # Find free slots
    free_slots = []
    duration = timedelta(minutes=duration_minutes)
    min_gap = timedelta(minutes=15)
    current = time_min

    # Align to earliest_hour
    if current.hour < earliest_hour:
        current = current.replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
    elif current.hour > latest_hour:
        current = (current + timedelta(days=1)).replace(hour=earliest_hour, minute=0, second=0, microsecond=0)

    while current < time_max and len(free_slots) < max_slots:
        # Skip weekends
        if working_days_only and current.weekday() >= 5:
            current = (current + timedelta(days=1)).replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
            continue

        # Skip outside hours
        if current.hour < earliest_hour:
            current = current.replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
            continue
        if current.hour > latest_hour:
            current = (current + timedelta(days=1)).replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
            continue

        slot_end = current + duration
        if slot_end > time_max:
            break

        # Check conflicts
        is_free = True
        for busy_start, busy_end in busy_periods:
            if busy_start >= slot_end:
                break
            if busy_end <= current:
                continue
            is_free = False
            current = busy_end + min_gap
            break

        if is_free:
            free_slots.append({
                "day": current.strftime("%A"),
                "start": format_datetime_for_display(current.isoformat()),
                "end": format_datetime_for_display(slot_end.isoformat()),
                "start_iso": current.isoformat(),
                "end_iso": slot_end.isoformat(),
                "duration": format_duration(duration),
            })
            current = slot_end + min_gap

    if not free_slots:
        return {
            "output": {
                "message": "No available slots found matching criteria.",
                "search_window": f"{time_min.strftime('%b %d')} - {time_max.strftime('%b %d, %Y')}",
                "criteria": f"{duration_minutes}min, {earliest_hour}:00-{latest_hour}:00, {'weekdays' if working_days_only else 'all days'}",
                "suggestion": "Try expanding search window, relaxing hour constraints, or including weekends.",
                "slots": [],
            }
        }

    return {
        "output": {
            "found": len(free_slots),
            "duration": f"{duration_minutes} minutes",
            "slots": free_slots,
        }
    }
