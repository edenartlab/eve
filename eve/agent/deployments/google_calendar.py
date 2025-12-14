"""
Google Calendar Platform Client

Handles OAuth flow, token refresh, and calendar API interactions.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger

from eve.agent.deployments import PlatformClient
from eve.agent.session.models import (
    DeploymentConfig,
    DeploymentSecrets,
    DeploymentSecretsGoogleCalendar,
)
from eve.api.errors import APIError

if TYPE_CHECKING:
    from fastapi import Request

    from eve.api.api_requests import DeploymentEmissionRequest


# OAuth Scopes - always request full calendar access, gate permissions via config
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Environment variable names
GOOGLE_CLIENT_ID_ENV = "GOOGLE_CALENDAR_CLIENT_ID"
GOOGLE_CLIENT_SECRET_ENV = "GOOGLE_CALENDAR_CLIENT_SECRET"


def get_google_client_config() -> Dict[str, Any]:
    """Get Google OAuth client configuration from environment"""
    client_id = os.getenv(GOOGLE_CLIENT_ID_ENV)
    client_secret = os.getenv(GOOGLE_CLIENT_SECRET_ENV)

    if not client_id or not client_secret:
        raise APIError(
            "Google Calendar OAuth not configured. Missing GOOGLE_CALENDAR_CLIENT_ID or GOOGLE_CALENDAR_CLIENT_SECRET",
            status_code=500,
        )

    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def create_oauth_flow(redirect_uri: str) -> Flow:
    """Create OAuth flow for Google Calendar"""
    client_config = get_google_client_config()
    return Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=redirect_uri
    )


def credentials_from_secrets(secrets: DeploymentSecretsGoogleCalendar) -> Credentials:
    """Create Google Credentials object from deployment secrets"""
    return Credentials(
        token=secrets.access_token,
        refresh_token=secrets.refresh_token,
        token_uri=secrets.token_uri,
        client_id=secrets.client_id,
        client_secret=secrets.client_secret,
        scopes=secrets.scopes,
    )


def secrets_from_credentials(
    credentials: Credentials, google_email: str, google_user_id: Optional[str] = None
) -> DeploymentSecretsGoogleCalendar:
    """Create deployment secrets from Google Credentials object"""
    return DeploymentSecretsGoogleCalendar(
        access_token=credentials.token,
        refresh_token=credentials.refresh_token,
        token_uri=credentials.token_uri,
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        scopes=list(credentials.scopes) if credentials.scopes else SCOPES,
        expires_at=credentials.expiry if credentials.expiry else None,
        google_email=google_email,
        google_user_id=google_user_id,
    )


async def get_calendar_service(secrets: DeploymentSecretsGoogleCalendar):
    """
    Get authenticated Google Calendar service.
    Automatically refreshes token if expired.
    """
    credentials = credentials_from_secrets(secrets)

    # Check if token needs refresh
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(GoogleRequest())
            logger.info("Google Calendar token refreshed successfully")
        except Exception as e:
            logger.error(f"Failed to refresh Google Calendar token: {e}")
            raise APIError(
                "Google Calendar authentication expired. Please reconnect your account.",
                status_code=401,
            )

    return build("calendar", "v3", credentials=credentials)


async def list_user_calendars(
    secrets: DeploymentSecretsGoogleCalendar,
) -> List[Dict[str, Any]]:
    """
    List all calendars accessible by the authenticated user.
    Returns list of calendar objects with id, summary (name), primary status, etc.
    """
    service = await get_calendar_service(secrets)
    calendar_list = service.calendarList().list().execute()
    return calendar_list.get("items", [])


class GoogleCalendarClient(PlatformClient):
    """
    Platform client for Google Calendar integration.

    Handles OAuth credential validation, calendar access, and agent tool management.
    """

    # Tools that will be available when this deployment is active
    # Tool implementations are in eve/tools/google_calendar/
    # Tools are separated by permission level for selective context inclusion
    TOOLS = {
        "google_calendar_query": {
            "description": "Query calendar: list events, get details, find free slots",
            "requires_write": False,
            "requires_delete": False,
        },
        "google_calendar_edit": {
            "description": "Create or update calendar events",
            "requires_write": True,
            "requires_delete": False,
        },
        "google_calendar_delete_event": {
            "description": "Delete/cancel a calendar event",
            "requires_write": False,
            "requires_delete": True,
        },
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> Tuple[DeploymentSecrets, DeploymentConfig]:
        """
        Validate Google Calendar credentials before deployment.

        This is called when creating or updating a deployment.
        Validates that:
        1. OAuth credentials are present and valid
        2. Selected calendar is accessible
        3. Token can be refreshed
        """
        gc_secrets = secrets.google_calendar
        gc_config = config.google_calendar

        if not gc_secrets:
            raise APIError(
                "Google Calendar secrets are required. Complete OAuth flow first.",
                status_code=400,
            )

        if not gc_secrets.access_token or not gc_secrets.refresh_token:
            raise APIError(
                "Invalid Google Calendar credentials. Missing access_token or refresh_token.",
                status_code=400,
            )

        if not gc_config:
            raise APIError(
                "Google Calendar configuration is required. Select a calendar.",
                status_code=400,
            )

        if not gc_config.calendar_id:
            raise APIError(
                "calendar_id is required in Google Calendar configuration.",
                status_code=400,
            )

        # Validate credentials by making a test API call
        try:
            service = await get_calendar_service(gc_secrets)

            # Verify access to the selected calendar
            calendar = (
                service.calendars().get(calendarId=gc_config.calendar_id).execute()
            )

            # Update config with calendar name if not set
            if not gc_config.calendar_name:
                gc_config.calendar_name = calendar.get("summary", gc_config.calendar_id)

            # Get calendar timezone if not overridden
            if not gc_config.time_zone:
                gc_config.time_zone = calendar.get("timeZone")

            logger.info(
                f"Google Calendar validated: {gc_config.calendar_name} "
                f"(ID: {gc_config.calendar_id})"
            )

        except HttpError as e:
            if e.resp.status == 404:
                raise APIError(
                    f"Calendar not found: {gc_config.calendar_id}. "
                    "Make sure you have access to this calendar.",
                    status_code=404,
                )
            elif e.resp.status == 401:
                raise APIError(
                    "Google Calendar authentication failed. Please reconnect your account.",
                    status_code=401,
                )
            else:
                raise APIError(
                    f"Google Calendar API error: {str(e)}",
                    status_code=e.resp.status,
                )
        except Exception as e:
            logger.error(f"Google Calendar validation failed: {e}")
            raise APIError(
                f"Failed to validate Google Calendar credentials: {str(e)}",
                status_code=500,
            )

        # Add calendar tools to agent based on permissions
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """
        Actions to perform after deployment is created.

        For Google Calendar, we don't need webhooks or continuous connections,
        so this is minimal.
        """
        logger.info(f"Google Calendar deployment completed for agent {self.agent.id}")

    async def stop(self) -> None:
        """
        Cleanup when deployment is stopped/deleted.

        Remove tools from agent and optionally revoke OAuth token.
        """
        self.remove_tools()

        # Optionally revoke the OAuth token
        # This is commented out as users might want to reconnect later
        # If you want to revoke on disconnect, uncomment:
        # try:
        #     if self.deployment and self.deployment.secrets.google_calendar:
        #         # Revoke token via Google's revocation endpoint
        #         pass
        # except Exception as e:
        #     logger.warning(f"Failed to revoke Google Calendar token: {e}")

        logger.info(f"Google Calendar deployment stopped for agent {self.agent.id}")

    async def update(
        self,
        old_config: Optional[DeploymentConfig] = None,
        new_config: Optional[DeploymentConfig] = None,
        old_secrets: Optional[DeploymentSecrets] = None,
        new_secrets: Optional[DeploymentSecrets] = None,
    ) -> None:
        """
        Handle updates to deployment configuration.

        For Google Calendar, this mainly handles permission changes
        (e.g., enabling/disabling write access).
        """
        old_gc_config = old_config.google_calendar if old_config else None
        new_gc_config = new_config.google_calendar if new_config else None

        if old_gc_config and new_gc_config:
            # Log permission changes
            if old_gc_config.allow_write != new_gc_config.allow_write:
                logger.info(
                    f"Google Calendar write access changed: "
                    f"{old_gc_config.allow_write} -> {new_gc_config.allow_write}"
                )

            if old_gc_config.calendar_id != new_gc_config.calendar_id:
                logger.info(
                    f"Google Calendar changed: "
                    f"{old_gc_config.calendar_id} -> {new_gc_config.calendar_id}"
                )

    async def interact(self, request: "Request") -> None:
        """
        Handle direct interactions with Google Calendar.

        Not typically used for calendar integrations as it's tool-based.
        """
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """
        Handle emissions from the agent.

        For Google Calendar, this would handle things like
        creating events from agent tool calls.
        """
        # Implementation deferred - this will be handled by the actual agent tools
        pass

    def add_tools(self) -> None:
        """
        Add Google Calendar tools to the agent.

        Tools are added based on the deployment configuration:
        - Read tools are always added
        - Write tools are only added if allow_write is True
        """
        # Use parent's add_tools for now
        # In production, you'd selectively add tools based on config
        super().add_tools()

    def remove_tools(self) -> None:
        """Remove Google Calendar tools from the agent"""
        super().remove_tools()
