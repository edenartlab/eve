"""
Modal app for running the MCP server in the cloud.
This provides a scalable way to expose Eve tools via MCP.
"""

import os
import modal
from pathlib import Path

from eve import db

# Modal app configuration
app_name = f"mcp-server-{db.lower()}"
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

# Get the root directory
root_dir = Path(__file__).parent.parent.parent

# Create the Modal image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE", "False")})
    .apt_install("git", "libmagic1")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("mcp[cli]")  # Install MCP SDK
    .add_local_file(str(root_dir / "pyproject.toml"), "/eve/pyproject.toml")
    .add_local_python_source("eve")
)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=10,
    container_idle_timeout=300,  # 5 minutes
    allow_concurrent_inputs=50,
    timeout=1800,  # 30 minutes
)
def run_mcp_server():
    """Run the MCP server on Modal"""
    from eve.mcp_server.server import mcp
    
    # Start the MCP server
    mcp.run()


@app.function(
    image=image,
    concurrency_limit=50,
    allow_concurrent_inputs=10,
    timeout=600,  # 10 minutes
)
async def execute_tool_call(tool_name: str, arguments: dict, api_key: str):
    """Execute a single tool call (for direct API access)"""
    from eve.mcp_server.server import execute_tool
    
    return await execute_tool(tool_name, arguments, api_key)


@app.function(
    image=image,
    concurrency_limit=10,
    allow_concurrent_inputs=50,
    timeout=60,
)
async def list_tools(api_key: str):
    """List available tools (for direct API access)"""
    from eve.mcp_server.server import list_available_tools
    
    return await list_available_tools(api_key)


@app.function(
    image=image,
    concurrency_limit=10,
    allow_concurrent_inputs=50,
    timeout=60,
)
async def get_schema(tool_name: str, api_key: str):
    """Get tool schema (for direct API access)"""
    from eve.mcp_server.server import get_tool_schema
    
    return await get_tool_schema(tool_name, api_key)


@app.function(
    image=image,
    concurrency_limit=10,
    allow_concurrent_inputs=50,
    timeout=60,
)
async def check_status(task_id: str, api_key: str):
    """Check task status (for direct API access)"""
    from eve.mcp_server.server import check_task_status
    
    return await check_task_status(task_id, api_key)


if __name__ == "__main__":
    # For local development
    with app.run():
        run_mcp_server.remote()