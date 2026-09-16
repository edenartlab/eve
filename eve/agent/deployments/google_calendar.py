"""Retired legacy Calendar adapter. Never refresh/revoke or expose saved tokens."""

from eve.agent.deployments import PlatformClient
from eve.api.errors import APIError

RETIRED_MESSAGE = (
    "Google Calendar is retired in old Eden. Use new Eden at https://dev.eden.art."
)


def calendar_retired():
    raise APIError(RETIRED_MESSAGE, status_code=410)


def get_google_client_config():
    calendar_retired()


def create_oauth_flow(redirect_uri):
    calendar_retired()


def credentials_from_secrets(secrets):
    calendar_retired()


def secrets_from_credentials(credentials, google_email, google_user_id=None):
    calendar_retired()


async def get_calendar_service(secrets):
    calendar_retired()


async def list_user_calendars(secrets):
    calendar_retired()


class GoogleCalendarClient(PlatformClient):
    # Retain names solely for the existing local remove_tools cleanup contract.
    TOOLS = {
        "google_calendar_query": {},
        "google_calendar_edit": {},
        "google_calendar_delete": {},
    }

    async def predeploy(self, secrets, config):
        calendar_retired()

    async def postdeploy(self):
        calendar_retired()

    async def update(
        self, old_config=None, new_config=None, old_secrets=None, new_secrets=None
    ):
        calendar_retired()

    async def interact(self, request):
        calendar_retired()

    async def handle_emission(self, emission):
        calendar_retired()

    def add_tools(self):
        calendar_retired()

    async def stop(self):
        # Existing local tool cleanup only; never revoke Google grants.
        self.remove_tools()

    def remove_tools(self):
        super().remove_tools()
