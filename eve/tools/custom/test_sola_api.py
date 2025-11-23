"""
Test script for Sola.day Calendar API wrapper.
Demonstrates various API calls and outputs results as JSON.
"""

import json
from datetime import datetime, timedelta

from sola_calendar_api import SolaCalendarAPI


def print_json(data, title=""):
    """Pretty print JSON data with a title."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print("=" * 80)
    print(json.dumps(data, indent=2, default=str))


def test_sola_api(group_id: int, timezone: str = "America/Argentina/Buenos_Aires"):
    """
    Test the Sola Calendar API with various queries.

    Args:
        group_id: The group ID to query
        timezone: Timezone for event times
    """
    # Initialize API
    api = SolaCalendarAPI(default_group_id=group_id, default_timezone=timezone)

    results = {
        "test_timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "timezone": timezone,
        "tests": {},
    }

    # Test 1: Get upcoming events (first page)
    print_json({}, "TEST 1: Get Upcoming Events (first 10)")
    try:
        upcoming_response = api.get_events(collection="upcoming", limit=10)
        results["tests"]["upcoming_events"] = {
            "status": "success",
            "count": len(upcoming_response.get("events", [])),
            "data": upcoming_response,
        }
        print_json(upcoming_response)
    except Exception as e:
        results["tests"]["upcoming_events"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 2: Get events on today's date
    print_json({}, "TEST 2: Get Events on Today's Date")
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        events_today = api.get_events_on_date(today)
        results["tests"]["events_today"] = {
            "status": "success",
            "date": today,
            "count": len(events_today),
            "data": events_today,
        }
        print_json({"date": today, "count": len(events_today), "events": events_today})
    except Exception as e:
        results["tests"]["events_today"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 3: Get events in the next 7 days
    print_json({}, "TEST 3: Get Events in Next 7 Days")
    try:
        next_week = api.get_upcoming_events_next_n_days(7)
        results["tests"]["next_7_days"] = {
            "status": "success",
            "count": len(next_week),
            "data": next_week,
        }
        print_json({"count": len(next_week), "events": next_week})
    except Exception as e:
        results["tests"]["next_7_days"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 4: Get events in the next 30 days
    print_json({}, "TEST 4: Get Events in Next 30 Days")
    try:
        next_month = api.get_upcoming_events_next_n_days(30)
        results["tests"]["next_30_days"] = {
            "status": "success",
            "count": len(next_month),
            "data": next_month,
        }
        print_json({"count": len(next_month), "events": next_month})
    except Exception as e:
        results["tests"]["next_30_days"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 5: Index events by date (upcoming only)
    print_json({}, "TEST 5: Index Upcoming Events by Date")
    try:
        indexed_events = api.index_events_by_date(collection="upcoming")
        results["tests"]["indexed_events"] = {
            "status": "success",
            "dates_count": len(indexed_events),
            "dates": list(indexed_events.keys()),
            "data": indexed_events,
        }
        # Create summary
        summary = {
            "total_dates": len(indexed_events),
            "dates_with_counts": {
                date: len(events) for date, events in indexed_events.items()
            },
            "full_data": indexed_events,
        }
        print_json(summary)
    except Exception as e:
        results["tests"]["indexed_events"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 6: Get events in a specific date range
    print_json({}, "TEST 6: Get Events in Date Range (Today + 14 days)")
    try:
        today = datetime.now()
        two_weeks = today + timedelta(days=14)
        range_events = api.get_events_by_date_range(today, two_weeks)
        results["tests"]["date_range"] = {
            "status": "success",
            "start_date": today.strftime("%Y-%m-%d"),
            "end_date": two_weeks.strftime("%Y-%m-%d"),
            "count": len(range_events),
            "data": range_events,
        }
        print_json(
            {
                "start_date": today.strftime("%Y-%m-%d"),
                "end_date": two_weeks.strftime("%Y-%m-%d"),
                "count": len(range_events),
                "events": range_events,
            }
        )
    except Exception as e:
        results["tests"]["date_range"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 7: Search events by title
    print_json({}, "TEST 7: Search Events by Title (if available)")
    try:
        search_results = api.get_events(search_title="", collection="all", limit=5)
        results["tests"]["search_title"] = {
            "status": "success",
            "search_term": "",
            "count": len(search_results.get("events", [])),
            "data": search_results,
        }
        print_json(search_results)
    except Exception as e:
        results["tests"]["search_title"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Test 8: Get a specific event by ID (if we have one)
    print_json({}, "TEST 8: Get Specific Event by ID")
    try:
        # Try to get an event ID from previous results
        upcoming_response = api.get_events(collection="upcoming", limit=1)
        events = upcoming_response.get("events", [])

        if events:
            event_id = events[0].get("id")
            event_details = api.get_event_by_id(event_id)
            results["tests"]["event_by_id"] = {
                "status": "success",
                "event_id": event_id,
                "data": event_details,
            }
            print_json(event_details)

            # Test 8b: Get participants for this event
            print_json({}, "TEST 8b: Get Event Participants")
            try:
                participants = api.get_event_participants(event_id)
                results["tests"]["event_participants"] = {
                    "status": "success",
                    "event_id": event_id,
                    "data": participants,
                }
                print_json(participants)
            except Exception as e:
                results["tests"]["event_participants"] = {
                    "status": "error",
                    "error": str(e),
                }
                print_json({"error": str(e)})
        else:
            results["tests"]["event_by_id"] = {
                "status": "skipped",
                "reason": "No events found to test with",
            }
            print_json({"status": "skipped", "reason": "No events found"})
    except Exception as e:
        results["tests"]["event_by_id"] = {"status": "error", "error": str(e)}
        print_json({"error": str(e)})

    # Save complete results to file
    output_filename = (
        f"sola_api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_json({}, f"COMPLETE RESULTS SAVED TO: {output_filename}")
    print_json(
        {
            "output_file": output_filename,
            "total_tests": len(results["tests"]),
            "successful": sum(
                1 for t in results["tests"].values() if t.get("status") == "success"
            ),
            "failed": sum(
                1 for t in results["tests"].values() if t.get("status") == "error"
            ),
            "skipped": sum(
                1 for t in results["tests"].values() if t.get("status") == "skipped"
            ),
        }
    )

    return results


if __name__ == "__main__":
    import sys

    # Check if group_id is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python test_sola_api.py <group_id> [timezone]")
        print("\nExample: python test_sola_api.py 1234 'America/New_York'")
        print("\nPlease provide a group_id to test with.")
        sys.exit(1)

    group_id = int(sys.argv[1])
    timezone = sys.argv[2] if len(sys.argv) > 2 else "America/Argentina/Buenos_Aires"

    print(f"Testing Sola Calendar API with group_id={group_id}, timezone={timezone}")

    results = test_sola_api(group_id, timezone)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
