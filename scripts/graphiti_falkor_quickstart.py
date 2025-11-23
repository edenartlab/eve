"""Simple Falkor-backed Graphiti smoke test.

Run this script after exporting any overrides you need for the FalkorDB connection.
By default it targets the compose service (`localhost:6380`) with no authentication.

Optionally set ``FALKORDB_WEB_PORT`` when starting the container if the default
web UI port ``8380`` conflicts with an existing service.

The script seeds a few sample episodes, performs a basic search, and prints the
results to stdout. It mirrors the structure of the official Graphiti quickstart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_driver() -> FalkorDriver:
    """Instantiate a Falkor driver using environment variables."""
    load_dotenv()

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6380"))
    username: Optional[str] = os.getenv("FALKORDB_USERNAME")
    password: Optional[str] = os.getenv("FALKORDB_PASSWORD")

    logger.info("Connecting to FalkorDB host=%s port=%s", host, port)

    return FalkorDriver(
        host=host,
        port=port,
        username=username,
        password=password,
    )


async def _seed(graphiti: Graphiti) -> None:
    """Seed the graph with sample episodes."""
    episodes = [
        {
            "content": {
                "name": "Gavin Newsom",
                "position": "Governor",
                "term_start": "January 7, 2019",
                "term_end": "Present",
            },
            "type": EpisodeType.json,
            "description": "podcast metadata",
        },
    ]

    for index, episode in enumerate(episodes):
        body = (
            episode["content"]
            if isinstance(episode["content"], str)
            else json.dumps(episode["content"])
        )
        await graphiti.add_episode(
            name=f"Freakonomics Radio {index}",
            episode_body=body,
            source=episode["type"],
            source_description=episode["description"],
            reference_time=datetime.now(timezone.utc),
        )
        logger.info(
            "Added episode Freakonomics Radio %s (%s)", index, episode["type"].value
        )


async def _run_searches(graphiti: Graphiti) -> None:
    """Run basic edge and node searches and dump the results."""
    query = "Who was the California Attorney General?"
    logger.info("Running search: %s", query)

    results = await graphiti.search(query)
    if not results:
        logger.warning("No results returned for query")
        return

    logger.info("Search results:")
    for item in results:
        logger.info("UUID=%s fact=%s", item.uuid, item.fact)

    center_uuid = results[0].source_node_uuid
    logger.info("Reranking with center node %s", center_uuid)

    reranked = await graphiti.search(query, center_node_uuid=center_uuid)
    for item in reranked:
        logger.info("Reranked UUID=%s fact=%s", item.uuid, item.fact)

    logger.info("Running node search with NODE_HYBRID_SEARCH_RRF")
    config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 5
    nodes = await graphiti._search(query="California Governor", config=config)

    for node in nodes.nodes:
        summary = (
            node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
        )
        logger.info(
            "Node uuid=%s name=%s labels=%s summary=%s",
            node.uuid,
            node.name,
            node.labels,
            summary,
        )


async def main() -> None:
    driver = _build_driver()
    graphiti = Graphiti(graph_driver=driver)

    try:
        await _seed(graphiti)
        await _run_searches(graphiti)
    finally:
        await graphiti.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
