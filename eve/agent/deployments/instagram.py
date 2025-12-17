import os
from typing import Optional, Tuple

import requests
from fastapi import Request

from eve.agent.deployments import PlatformClient
from eve.agent.session.models import (
    DeploymentConfig,
    DeploymentSecrets,
    DeploymentSecretsInstagram,
    DeploymentSettingsInstagram,
)


class InstagramClient(PlatformClient):
    """Instagram client using Instagram Login user access tokens."""

    # No platform-specific tools added yet beyond stub tool

    def _normalize_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        normalized = token.strip().strip('"').strip("'")
        if normalized.lower().startswith("bearer "):
            normalized = normalized.split(" ", 1)[1].strip()
        normalized = "".join(normalized.split())
        return normalized or None

    def _get_access_token(self, secrets: DeploymentSecrets) -> str:
        env_token = (
            os.getenv("INSTAGRAM_ACCESS_TOKEN")
            or os.getenv("INSTAGRAM_USER_TOKEN")
            # Back-compat with older env names used in staging.
            or os.getenv("IG_ACCESS_TOKEN")
            or os.getenv("INSTAGRAM_PAGE_TOKEN")
            or os.getenv("IG_PAGE_TOKEN")
        )
        normalized_env = self._normalize_token(env_token)
        if normalized_env:
            return normalized_env
        normalized_secret = self._normalize_token(
            secrets.instagram.access_token
            if secrets and secrets.instagram and secrets.instagram.access_token
            else None
        )
        if normalized_secret:
            return normalized_secret
        raise ValueError(
            "Missing Instagram access token (set INSTAGRAM_ACCESS_TOKEN or IG_ACCESS_TOKEN)"
        )

    def _ig_graph_base(self) -> str:
        host = os.getenv("INSTAGRAM_GRAPH_HOST", "https://graph.instagram.com").rstrip(
            "/"
        )
        version = os.getenv("INSTAGRAM_API_VERSION", "v24.0").strip().lstrip("/")
        return f"{host}/{version}" if version else host

    def _resolve_ig_identity(
        self, access_token: str, existing_ig_user_id: Optional[str]
    ) -> tuple[str, Optional[str]]:
        ig_user_id = existing_ig_user_id
        username = None

        # Instagram Graph endpoints accept tokens via query param; some also accept Bearer auth.
        headers = {"Authorization": f"Bearer {access_token}"}
        auth_params = {"access_token": access_token}

        # If we don't have ig_user_id, try /me with Instagram Login token.
        if not ig_user_id:
            base = self._ig_graph_base()
            resp = requests.get(
                f"{base}/me",
                params={"fields": "user_id,username,id", **auth_params},
                headers=headers,
                timeout=15,
            )
            if resp.ok:
                data = resp.json() or {}
                ig_user_id = data.get("user_id") or data.get("id") or ig_user_id
                username = data.get("username") or username

        # If still missing, optionally use INSTAGRAM_IG_USER_ID env
        if not ig_user_id:
            ig_user_id = os.getenv("INSTAGRAM_IG_USER_ID")

        if not ig_user_id:
            raise ValueError(
                "Could not resolve IG user id from token. Ensure the Instagram account is a professional (Creator/Business) account and the token is valid."
            )

        # Fetch username for display if we didn't get it from /me.
        if not username:
            base = self._ig_graph_base()
            resp = requests.get(
                f"{base}/{ig_user_id}",
                params={"fields": "username", **auth_params},
                headers=headers,
                timeout=15,
            )
            if resp.ok:
                username = (resp.json() or {}).get("username")

        return ig_user_id, username

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> Tuple[DeploymentSecrets, DeploymentConfig]:
        merged_secrets = secrets or DeploymentSecrets()
        merged_config = config or DeploymentConfig()

        access_token = self._get_access_token(merged_secrets)
        ig_user_id_env = (
            merged_config.instagram.ig_user_id if merged_config.instagram else None
        )
        ig_user_id, username = self._resolve_ig_identity(access_token, ig_user_id_env)

        merged_secrets.instagram = DeploymentSecretsInstagram(
            access_token=access_token,
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
