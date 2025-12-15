import os
from typing import Optional, Tuple

from fastapi import Request

from eve.agent.deployments import PlatformClient
from eve.agent.session.models import (
    DeploymentConfig,
    DeploymentSecrets,
    DeploymentSecretsInstagram,
    DeploymentSettingsInstagram,
)


class InstagramClient(PlatformClient):
    """Shared-token Instagram client (staging PoC)."""

    # No platform-specific tools added yet beyond stub tool

    def _get_page_token(self, secrets: DeploymentSecrets) -> str:
        env_token = (
            os.getenv("INSTAGRAM_PAGE_TOKEN")
            or os.getenv("IG_ACCESS_TOKEN")
            or os.getenv("IG_PAGE_TOKEN")
        )
        if env_token:
            return env_token
        if secrets and secrets.instagram and secrets.instagram.access_token:
            return secrets.instagram.access_token
        raise ValueError("Missing Instagram page token (set INSTAGRAM_PAGE_TOKEN)")

    def _resolve_ig_identity(
        self, page_token: str, existing_ig_user_id: Optional[str]
    ) -> tuple[str, Optional[str]]:
        ig_user_id = existing_ig_user_id
        username = None

        # If we don't have ig_user_id, try /me with page token (returns page with IG link)
        if not ig_user_id:
            resp = requests.get(
                "https://graph.facebook.com/v24.0/me",
                params={
                    "fields": "instagram_business_account",
                    "access_token": page_token,
                },
                timeout=15,
            )
            if resp.ok:
                ig_user_id = (
                    resp.json()
                    .get("instagram_business_account", {})
                    .get("id", ig_user_id)
                )

        # If still missing, optionally use INSTAGRAM_IG_USER_ID env
        if not ig_user_id:
            ig_user_id = os.getenv("INSTAGRAM_IG_USER_ID")

        if not ig_user_id:
            raise ValueError(
                "Could not resolve instagram_business_account id. Ensure the IG account is linked to the Page and the page token is valid."
            )

        # Fetch username for display
        resp = requests.get(
            f"https://graph.facebook.com/v24.0/{ig_user_id}",
            params={"fields": "username", "access_token": page_token},
            timeout=15,
        )
        if resp.ok:
            username = resp.json().get("username")

        return ig_user_id, username

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> Tuple[DeploymentSecrets, DeploymentConfig]:
        merged_secrets = secrets or DeploymentSecrets()
        merged_config = config or DeploymentConfig()

        page_token = self._get_page_token(merged_secrets)
        ig_user_id_env = (
            merged_config.instagram.ig_user_id if merged_config.instagram else None
        )
        ig_user_id, username = self._resolve_ig_identity(page_token, ig_user_id_env)

        merged_secrets.instagram = DeploymentSecretsInstagram(
            access_token=page_token,
            refresh_token=None,
            expires_at=None,
            username=username,
        )
        merged_config.instagram = DeploymentSettingsInstagram(
            username=username, ig_user_id=ig_user_id
        )

        return merged_secrets, merged_config

    async def postdeploy(self) -> None:
        # Nothing to start; we rely on shared token
        self.add_tools()

    async def stop(self) -> None:
        # No background workers to stop for shared-token flow
        return None

    async def update(
        self,
        old_config: Optional[DeploymentConfig] = None,
        new_config: Optional[DeploymentConfig] = None,
        old_secrets: Optional[DeploymentSecrets] = None,
        new_secrets: Optional[DeploymentSecrets] = None,
    ) -> None:
        # Nothing specific for now
        return None

    async def interact(self, request: Request) -> None:
        raise NotImplementedError("Interact not implemented for Instagram")

    async def handle_emission(self, emission) -> None:
        # Not used for Instagram yet
        return None


import requests
