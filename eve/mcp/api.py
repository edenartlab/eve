"""Modal deployment for Eden MCP server."""

import os
import modal
from pathlib import Path

app_name = "mcp-server"
db = os.getenv("DB", "STAGE").lower()

app = modal.App(
    name=f"{app_name}-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

root_dir = Path(__file__).parent.parent.parent

# Build image with fastmcp and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .pip_install("fastmcp>=2.12.4", "httpx>=0.27.2")
    .add_local_file(str(Path(__file__).parent / "openapi.json"), "/mcp/openapi.json")
    .add_local_file(str(Path(__file__).parent / "server.py"), "/mcp/server.py")
)


@app.function(
    image=image,
    min_containers=1,
    max_containers=5,
    timeout=600,
)
def run_mcp_server():
    """Run the MCP server."""
    import sys

    sys.path.insert(0, "/mcp")

    from server import mcp

    return mcp.run()


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    run_mcp_server.remote()
