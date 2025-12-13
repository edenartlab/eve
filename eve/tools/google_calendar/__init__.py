"""
Google Calendar Agent Tools

These tools allow agents to interact with Google Calendar on behalf of users who have connected their Google account via OAuth.

Available tools:
- list_events: Retrieve events from a calendar within a time range
- create_event: Create a new calendar event
- update_event: Update an existing calendar event

Authentication:
The frontend handles the OAuth flow and stores the credentials encrypted. When these tools are invoked, they load the deployment's stored credentials and use them to authenticate with the Google Calendar API.

Permissions:
- Read operations (list_events) are always available when the deployment is active
- Write operations (create_event, update_event) require allow_write=True in deployment config
"""

from eve.agent.deployments.google_calendar import (
    SCOPES,
    create_oauth_flow,
    credentials_from_secrets,
    get_calendar_service,
    get_google_client_config,
    list_user_calendars,
    secrets_from_credentials,
)

__all__ = [
    "SCOPES",
    "get_google_client_config",
    "create_oauth_flow",
    "credentials_from_secrets",
    "secrets_from_credentials",
    "get_calendar_service",
    "list_user_calendars",
]
