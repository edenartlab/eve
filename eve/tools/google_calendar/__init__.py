"""
Google Calendar Agent Tools

Tools for interacting with Google Calendar through the agent system.
Consolidated into 3 tools separated by permission level:

Available tools:
- google_calendar_query: List events, get event details, find free slots (no special permissions)
- google_calendar_edit: Create or update events (requires write permission)
- google_calendar_delete_event: Delete/cancel events (requires delete permission)

Each tool is defined in its own subdirectory with:
- api.yaml: Tool schema definition
- handler.py: Implementation
- test.json: Test parameters

Shared utilities are in utils.py.
"""

from eve.tools.google_calendar.utils import (
    check_permissions,
    format_datetime_for_display,
    format_duration,
    format_event_compact,
    format_events_list,
    get_calendar_deployment,
    get_service_and_config,
    parse_datetime,
    parse_recurrence_rule,
)

__all__ = [
    "get_calendar_deployment",
    "get_service_and_config",
    "parse_datetime",
    "format_datetime_for_display",
    "format_duration",
    "format_event_compact",
    "format_events_list",
    "check_permissions",
    "parse_recurrence_rule",
]
