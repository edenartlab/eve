"""Graphiti memory bootstrap helpers.

Set the following environment variables before calling :func:`init_graphiti`:

* ``FALKORDB_HOST`` – hostname of the FalkorDB endpoint (defaults to ``localhost``).
* ``FALKORDB_PORT`` – port for the FalkorDB endpoint (defaults to ``6379``).
* ``FALKORDB_USERNAME`` – optional username for FalkorDB authentication.
* ``FALKORDB_PASSWORD`` – optional password for FalkorDB authentication.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
except ImportError as import_error:  # pragma: no cover - optional dependency
    Graphiti = None  # type: ignore[assignment]
    FalkorDriver = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

DEFAULT_FALKORDB_PORT = 6380


@dataclass(frozen=True)
class GraphitiConfig:
    """Container for the connection parameters used when creating Graphiti."""

    host: str
    port: int = DEFAULT_FALKORDB_PORT
    username: Optional[str] = None
    password: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Derive a config from the standard environment variables."""
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", DEFAULT_FALKORDB_PORT))
        username = os.getenv("FALKORDB_USERNAME")
        password = os.getenv("FALKORDB_PASSWORD")

        return cls(host=host, port=port, username=username, password=password)


def init_graphiti(config: Optional[GraphitiConfig] = None) -> "Graphiti":
    """Initialize Graphiti with a FalkorDB driver.

    Parameters
    ----------
    config:
        Optional configuration. When omitted the values are inferred from the
        environment (see :meth:`GraphitiConfig.from_env`).

    Returns
        -------
        Graphiti
            An instance wired to the configured FalkorDB backend.

    Raises
    ------
    ImportError
        If ``graphiti_core`` is not installed in the current environment.
    RuntimeError
        If driver initialization fails.
    """

    if Graphiti is None or FalkorDriver is None:
        raise ImportError(
            "graphiti_core is not available. Install it to enable Graphiti-backed memory."
        ) from _IMPORT_ERROR

    config = config or GraphitiConfig.from_env()

    logger.debug(
        "Initializing Graphiti; host=%s port=%s",
        config.host,
        config.port,
    )

    driver = FalkorDriver(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
    )

    return Graphiti(graph_driver=driver)
