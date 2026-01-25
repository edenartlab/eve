"""Home Assistant REST API client."""

import os
from typing import Optional

import httpx

HA_URL = os.getenv("HOME_ASSISTANT_URL")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")


def get_headers() -> dict:
    """Get authorization headers for HA API."""
    if not HA_TOKEN:
        raise ValueError("HOME_ASSISTANT_TOKEN environment variable not set")
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def get_base_url() -> str:
    """Get the Home Assistant base URL."""
    if not HA_URL:
        raise ValueError("HOME_ASSISTANT_URL environment variable not set")
    return HA_URL.rstrip("/")


async def get_entity(entity_id: str) -> dict:
    """Get state of a specific entity."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{get_base_url()}/api/states/{entity_id}",
            headers=get_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def get_all_states() -> list:
    """Get all entity states."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{get_base_url()}/api/states",
            headers=get_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def call_service(
    domain: str, service: str, service_data: Optional[dict] = None
) -> list:
    """Call a Home Assistant service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{get_base_url()}/api/services/{domain}/{service}",
            headers=get_headers(),
            json=service_data or {},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def get_history(
    entity_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> list:
    """Get history for an entity."""
    params = {"filter_entity_id": entity_id}
    if end_time:
        params["end_time"] = end_time

    # Start time is part of the URL path
    url = f"{get_base_url()}/api/history/period"
    if start_time:
        url = f"{url}/{start_time}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers=get_headers(),
            params=params,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
