"""
Handler for google_calendar_query tool.

Consolidated read-only operations: list events, get event details, find free slots.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

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
    parse_recurrence_rule,
)


async def handler(context: ToolContext) -> Dict[str, Any]:
    """Query Google Calendar - list, get, or find_free_slots."""
    if not context.agent:
        raise Exception("Agent is required")

    action = context.args.get("action")
    if not action:
        raise Exception("action is required (list, get, or find_free_slots)")

    service, config, deployment = await get_service_and_config(context.agent)
    calendar_id = config.calendar_id

    if action == "list":
        return await _list_events(context, service, calendar_id)
    elif action == "get":
        return await _get_event(context, service, calendar_id)
    elif action == "find_free_slots":
        return await _find_free_slots(context, service, calendar_id)
    else:
        raise Exception(f"Unknown action: {action}. Use list, get, or find_free_slots.")


async def _execute(request):
    return await asyncio.to_thread(request.execute)


async def _list_events(
    context: ToolContext,
    service,
    calendar_id: str,
) -> Dict[str, Any]:
    """List events in a time range."""
    start_time_str = context.args.get("start_time")
    end_time_str = context.args.get("end_time")
    max_results = min(context.args.get("max_results", 10), 50)
    include_description = context.args.get("include_description", False)

    now = datetime.now(timezone.utc)
    start_time = parse_datetime(start_time_str) if start_time_str else now
    end_time = (
        parse_datetime(end_time_str) if end_time_str else start_time + timedelta(days=7)
    )

    try:
        events_result = await _execute(
            service.events().list(
                calendarId=calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                showDeleted=False,
            )
        )
        events = events_result.get("items", [])
        logger.info(f"Retrieved {len(events)} events from Google Calendar")
    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        elif e.resp.status == 404:
            raise Exception(f"Calendar not found: {calendar_id}")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

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

    include_fields = ["id", "title", "start", "end"]
    if include_description:
        include_fields.extend(["description", "location", "attendees"])

    formatted_events = format_events_list(
        standalone_events, include_fields=include_fields, max_events=max_results
    )

    # Format recurring event groups - bundle into single entry with schedule info
    for recurring_id, instances in recurring_groups.items():
        if instances:
            instances.sort(
                key=lambda x: x.get("start", {}).get("dateTime")
                or x.get("start", {}).get("date", "")
            )
            try:
                master_event = await _execute(
                    service.events().get(calendarId=calendar_id, eventId=recurring_id)
                )
                formatted = format_event_compact(
                    master_event, include_fields=include_fields
                )
                recurrence = master_event.get("recurrence")
                if recurrence:
                    schedule = parse_recurrence_rule(recurrence)
                    if schedule:
                        formatted["schedule"] = schedule
                first_inst = instances[0]
                first_start = first_inst.get("start", {})
                formatted["next_occurrence"] = format_datetime_for_display(
                    first_start.get("dateTime") or first_start.get("date")
                )
                formatted["occurrences_in_range"] = len(instances)
                formatted["is_recurring"] = True
                formatted_events.append(formatted)
            except HttpError:
                formatted = format_event_compact(
                    instances[0], include_fields=include_fields
                )
                formatted["is_recurring"] = True
                formatted["occurrences_in_range"] = len(instances)
                formatted_events.append(formatted)

    formatted_events.sort(key=lambda x: x.get("start", ""))

    if not formatted_events:
        return {
            "output": {
                "message": "No events found in the specified time range.",
                "time_range": f"{start_time.strftime('%b %d')} - {end_time.strftime('%b %d, %Y')}",
                "events": [],
            }
        }

    return {
        "output": {
            "count": len(formatted_events),
            "time_range": f"{start_time.strftime('%b %d')} - {end_time.strftime('%b %d, %Y')}",
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

    try:
        event = await _execute(
            service.events().get(calendarId=calendar_id, eventId=event_id)
        )
        logger.info(f"Retrieved event: {event.get('summary', 'Untitled')}")
    except HttpError as e:
        if e.resp.status == 404:
            raise Exception(f"Event not found: {event_id}")
        elif e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

    formatted = {
        "id": event.get("id", ""),
        "title": event.get("summary", "Untitled"),
    }

    start = event.get("start", {})
    start_str = start.get("dateTime") or start.get("date")
    formatted["start"] = format_datetime_for_display(start_str)
    if start.get("dateTime"):
        formatted["start_iso"] = start.get("dateTime")

    end = event.get("end", {})
    end_str = end.get("dateTime") or end.get("date")
    formatted["end"] = format_datetime_for_display(end_str)

    if event.get("location"):
        formatted["location"] = event.get("location")

    if event.get("description"):
        formatted["description"] = event.get("description")

    if event.get("attendees"):
        formatted["attendees"] = [
            {
                "name": a.get("displayName") or a.get("email", "Unknown"),
                "response": a.get("responseStatus", "needsAction"),
            }
            for a in event.get("attendees", [])[:15]
        ]

    formatted["url"] = event.get("htmlLink", "")

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
) -> Dict[str, Any]:
    """Find available time slots."""
    duration_minutes = context.args.get("duration_minutes")
    if not duration_minutes:
        raise Exception("duration_minutes is required for action=find_free_slots")

    start_time_str = context.args.get("start_time")
    end_time_str = context.args.get("end_time")
    earliest_hour = context.args.get("earliest_hour", 9)
    latest_hour = context.args.get("latest_hour", 17)
    working_days_only = context.args.get("working_days_only", True)
    max_slots = min(context.args.get("max_results", 5), 20)

    if earliest_hour > latest_hour:
        raise Exception(
            f"earliest_hour ({earliest_hour}) must be <= latest_hour ({latest_hour})"
        )

    now = datetime.now(timezone.utc)
    start_time = parse_datetime(start_time_str) if start_time_str else now
    if start_time < now:
        start_time = now
    end_time = (
        parse_datetime(end_time_str) if end_time_str else start_time + timedelta(days=7)
    )

    try:
        events_result = await _execute(
            service.events().list(
                calendarId=calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
        )
        events = events_result.get("items", [])
    except HttpError as e:
        if e.resp.status == 401:
            raise Exception("Google Calendar authentication expired. Please reconnect.")
        else:
            raise Exception(f"Google Calendar API error: {str(e)}")

    busy_periods = []
    for event in events:
        start = event.get("start", {})
        end = event.get("end", {})
        start_str = start.get("dateTime")
        if not start_str:
            date_str = start.get("date")
            if date_str:
                start_dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                end_dt = start_dt + timedelta(days=1)
                busy_periods.append((start_dt, end_dt))
            continue
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end_str = end.get("dateTime")
        end_dt = (
            datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            if end_str
            else start_dt + timedelta(hours=1)
        )
        busy_periods.append((start_dt, end_dt))

    busy_periods.sort(key=lambda x: x[0])

    free_slots = []
    duration = timedelta(minutes=duration_minutes)
    min_gap = timedelta(minutes=15)
    current = start_time

    if current.hour < earliest_hour:
        current = current.replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
    elif current.hour > latest_hour:
        current = (current + timedelta(days=1)).replace(
            hour=earliest_hour, minute=0, second=0, microsecond=0
        )

    while current < end_time and len(free_slots) < max_slots:
        if working_days_only and current.weekday() >= 5:
            current = (current + timedelta(days=1)).replace(
                hour=earliest_hour, minute=0, second=0, microsecond=0
            )
            continue

        if current.hour < earliest_hour:
            current = current.replace(
                hour=earliest_hour, minute=0, second=0, microsecond=0
            )
            continue
        if current.hour > latest_hour:
            current = (current + timedelta(days=1)).replace(
                hour=earliest_hour, minute=0, second=0, microsecond=0
            )
            continue

        slot_end = current + duration
        if slot_end > end_time:
            break

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
            free_slots.append(
                {
                    "day": current.strftime("%A"),
                    "start": format_datetime_for_display(current.isoformat()),
                    "end": format_datetime_for_display(slot_end.isoformat()),
                    "start_iso": current.isoformat(),
                    "end_iso": slot_end.isoformat(),
                    "duration": format_duration(duration),
                }
            )
            current = slot_end + min_gap

    if not free_slots:
        return {
            "output": {
                "message": "No available slots found matching criteria.",
                "search_window": f"{start_time.strftime('%b %d')} - {end_time.strftime('%b %d, %Y')}",
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
