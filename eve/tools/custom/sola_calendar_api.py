"""
Sola.day Calendar API Wrapper
Provides flexible access to calendar events with date-based filtering and indexing.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Literal
from dateutil import parser as date_parser
import pytz


class SolaCalendarAPI:
    """
    Flexible wrapper for the Sola.day calendar API.
    Supports event retrieval, filtering, and date-based indexing.
    """

    BASE_URL = "https://sola-event-scrape-MO58.replit.app"

    def __init__(self, default_group_id: Optional[int] = None,
                 default_timezone: str = "America/Argentina/Buenos_Aires"):
        """
        Initialize the API wrapper.

        Args:
            default_group_id: Default group ID for event searches
            default_timezone: Default timezone for event times
        """
        self.default_group_id = default_group_id
        self.default_timezone = default_timezone
        self.session = requests.Session()

    def get_events(self,
                   group_id: Optional[int] = None,
                   collection: Literal["upcoming", "past", "all"] = "upcoming",
                   search_title: Optional[str] = None,
                   timezone: Optional[str] = None,
                   page: int = 1,
                   limit: int = 25) -> Dict:
        """
        Search for events with flexible filtering.

        Args:
            group_id: Group ID to search within
            collection: Filter by "upcoming", "past", or "all" events
            search_title: Filter by title substring
            timezone: Timezone for event times
            page: Page number for pagination
            limit: Number of results per page (1-100)

        Returns:
            Dict containing events and metadata
        """
        group_id = group_id or self.default_group_id
        if group_id is None:
            raise ValueError("group_id is required (set default_group_id or pass as parameter)")

        timezone = timezone or self.default_timezone

        params = {
            "group_id": group_id,
            "collection": collection,
            "timezone": timezone,
            "page": page,
            "limit": min(limit, 100)
        }

        if search_title:
            params["search_title"] = search_title

        response = self.session.get(f"{self.BASE_URL}/events/search", params=params)
        response.raise_for_status()
        return response.json()

    def get_event_by_id(self, event_id: int, group_id: Optional[int] = None) -> Dict:
        """
        Get a specific event by ID.

        Args:
            event_id: The event ID
            group_id: Optional group ID

        Returns:
            Event object with full details
        """
        params = {}
        if group_id:
            params["group_id"] = group_id

        response = self.session.get(f"{self.BASE_URL}/events/{event_id}", params=params)
        response.raise_for_status()
        return response.json()

    def get_event_participants(self, event_id: int) -> Dict:
        """
        Get participants for a specific event.

        Args:
            event_id: The event ID

        Returns:
            Dict with participant count and list of attendees
        """
        response = self.session.get(f"{self.BASE_URL}/events/{event_id}/participants")
        response.raise_for_status()
        return response.json()

    def get_group_members(self, groupname: str) -> Dict:
        """
        Get members of a group.

        Args:
            groupname: The group name

        Returns:
            Dict with member count and list of usernames
        """
        response = self.session.get(f"{self.BASE_URL}/groups/{groupname}/members")
        response.raise_for_status()
        return response.json()

    def get_user_events(self,
                       username: str,
                       tab: Literal["attending", "hosting", "co-hosting"] = "attending") -> Dict:
        """
        Get events a user is attending, hosting, or co-hosting.

        Args:
            username: The username
            tab: Filter by relationship type

        Returns:
            Dict with user's events
        """
        params = {"tab": tab}
        response = self.session.get(f"{self.BASE_URL}/users/{username}/events", params=params)
        response.raise_for_status()
        return response.json()

    def get_events_by_date_range(self,
                                  start_date: Union[str, datetime],
                                  end_date: Union[str, datetime],
                                  group_id: Optional[int] = None,
                                  timezone: Optional[str] = None) -> List[Dict]:
        """
        Get events within a specific date range.

        Args:
            start_date: Start date (string or datetime object)
            end_date: End date (string or datetime object)
            group_id: Group ID to search within
            timezone: Timezone for comparison

        Returns:
            List of events within the date range
        """
        # Parse dates if strings
        if isinstance(start_date, str):
            start_date = date_parser.parse(start_date)
        if isinstance(end_date, str):
            end_date = date_parser.parse(end_date)

        # Make timezone-aware if needed
        tz = pytz.timezone(timezone or self.default_timezone)
        if start_date.tzinfo is None:
            start_date = tz.localize(start_date)
        if end_date.tzinfo is None:
            end_date = tz.localize(end_date)

        # Fetch all events (both upcoming and past)
        all_events = []
        page = 1

        while True:
            response = self.get_events(
                group_id=group_id,
                collection="all",
                timezone=timezone,
                page=page,
                limit=100
            )

            events = response.get("events", [])
            if not events:
                break

            all_events.extend(events)
            page += 1

        # Filter by date range
        filtered_events = []
        for event in all_events:
            event_start = date_parser.parse(event["start_time_utc"])
            if start_date <= event_start <= end_date:
                filtered_events.append(event)

        return filtered_events

    def get_events_on_date(self,
                          date: Union[str, datetime],
                          group_id: Optional[int] = None,
                          timezone: Optional[str] = None) -> List[Dict]:
        """
        Get events on a specific date.

        Args:
            date: The date (string or datetime object)
            group_id: Group ID to search within
            timezone: Timezone for comparison

        Returns:
            List of events on that date
        """
        # Parse date if string
        if isinstance(date, str):
            date = date_parser.parse(date)

        # Create date range for the entire day
        tz = pytz.timezone(timezone or self.default_timezone)
        if date.tzinfo is None:
            date = tz.localize(date.replace(hour=0, minute=0, second=0, microsecond=0))
        else:
            date = date.replace(hour=0, minute=0, second=0, microsecond=0)

        start_of_day = date
        end_of_day = date + timedelta(days=1) - timedelta(microseconds=1)

        return self.get_events_by_date_range(start_of_day, end_of_day, group_id, timezone)

    def index_events_by_date(self,
                            group_id: Optional[int] = None,
                            collection: Literal["upcoming", "past", "all"] = "upcoming",
                            timezone: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Create a date-indexed dictionary of events.

        Args:
            group_id: Group ID to search within
            collection: Filter by "upcoming", "past", or "all" events
            timezone: Timezone for date keys

        Returns:
            Dict with dates (YYYY-MM-DD) as keys and lists of events as values
        """
        events_by_date = {}
        page = 1

        while True:
            response = self.get_events(
                group_id=group_id,
                collection=collection,
                timezone=timezone,
                page=page,
                limit=100
            )

            events = response.get("events", [])
            if not events:
                break

            for event in events:
                # Parse the start time and extract date
                event_start = date_parser.parse(event["start_time_utc"])
                date_key = event_start.strftime("%Y-%m-%d")

                if date_key not in events_by_date:
                    events_by_date[date_key] = []

                events_by_date[date_key].append(event)

            page += 1

        return events_by_date

    def get_upcoming_events_next_n_days(self,
                                       n_days: int,
                                       group_id: Optional[int] = None,
                                       timezone: Optional[str] = None) -> List[Dict]:
        """
        Get upcoming events in the next N days.

        Args:
            n_days: Number of days to look ahead
            group_id: Group ID to search within
            timezone: Timezone for comparison

        Returns:
            List of events in the next N days
        """
        tz = pytz.timezone(timezone or self.default_timezone)
        now = datetime.now(tz)
        future_date = now + timedelta(days=n_days)

        return self.get_events_by_date_range(now, future_date, group_id, timezone)


# Convenience function for simple use cases
def get_events_by_date(date: Union[str, datetime],
                      group_id: int,
                      timezone: str = "America/Argentina/Buenos_Aires") -> List[Dict]:
    """
    Simple function to get events on a specific date.

    Args:
        date: The date (string or datetime object)
        group_id: Group ID to search within
        timezone: Timezone for comparison

    Returns:
        List of events on that date
    """
    api = SolaCalendarAPI(default_group_id=group_id, default_timezone=timezone)
    return api.get_events_on_date(date, timezone=timezone)


# Example usage
if __name__ == "__main__":
    # Initialize with a default group ID
    api = SolaCalendarAPI(default_group_id=1234)

    # Get events on a specific date
    events_today = api.get_events_on_date("2025-10-26")
    print(f"Events today: {len(events_today)}")

    # Get events in the next 7 days
    upcoming_week = api.get_upcoming_events_next_n_days(7)
    print(f"Events in next 7 days: {len(upcoming_week)}")

    # Index all events by date
    events_index = api.index_events_by_date(collection="all")
    print(f"Dates with events: {list(events_index.keys())}")

    # Get events in a date range
    events_in_range = api.get_events_by_date_range("2025-10-26", "2025-11-26")
    print(f"Events in range: {len(events_in_range)}")

    # Search with title filter
    search_results = api.get_events(search_title="Python", collection="all")
    print(f"Events matching 'Python': {len(search_results.get('events', []))}")
