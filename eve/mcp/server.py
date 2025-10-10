"""FastMCP server for Eden API using OpenAPI spec."""

import os
import json
import httpx
from pathlib import Path
from fastmcp import FastMCP

# Load the OpenAPI spec
OPENAPI_PATH = Path(__file__).parent / "openapi.json"


def load_openapi_spec():
    """Load OpenAPI spec from file."""
    with open(OPENAPI_PATH, "r") as f:
        return json.load(f)


# Determine API base URL from environment or default to staging
db = os.getenv("DB", "STAGE")
if db == "STAGE":
    base_url = "https://staging.api.eden.art"
elif db == "PROD":
    base_url = "https://api.eden.art"
else:
    base_url = os.getenv("EDEN_API_URL", "http://localhost:8080")

# Create HTTP client with authentication
api_client = httpx.AsyncClient(
    base_url=base_url,
    timeout=30.0,
)

# Load OpenAPI spec
openapi_spec = load_openapi_spec()

# Create the MCP server from OpenAPI spec
mcp = FastMCP.from_openapi(
    openapi_spec=openapi_spec,
    client=api_client,
    name="Eden API Server",
)

if __name__ == "__main__":
    mcp.run()
