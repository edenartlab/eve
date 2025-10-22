import os
from typing import Dict, Any, Optional
from pydantic import Field
from urllib.parse import urlencode

from ..tool import Tool, tool_context
from ..task import Task


@tool_context("mcp")
class MCPTool(Tool):
    """Tool for connecting to MCP servers using streamable_http"""

    mcp_server_url: str = Field(..., description="URL of the MCP server")
    mcp_tool_name: str = Field(
        ..., description="Name of the tool to call on the MCP server"
    )
    mcp_env_params: Optional[Dict[str, str]] = Field(
        None, description="Environment variable mappings for query params"
    )
    mcp_timeout: int = Field(30, description="Timeout in seconds for MCP requests")

    def _build_url_with_auth(self) -> str:
        """Build URL with query parameters from environment variables"""
        url = self.mcp_server_url
        params = {}

        if self.mcp_env_params:
            for param_name, env_var in self.mcp_env_params.items():
                value = os.getenv(env_var)
                if value:
                    params[param_name] = value

        if params:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(params)}"

        return url

    async def _call_mcp_tool(self, args: Dict[str, Any]) -> str:
        """Call MCP tool using streamable_http"""
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = self._build_url_with_auth()
        formatted_args = self._format_args(args)

        async with streamablehttp_client(url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(self.mcp_tool_name, formatted_args)

                if hasattr(result, "content") and result.content:
                    content_parts = []
                    for i, item in enumerate(result.content):
                        if hasattr(item, "text"):
                            content_parts.append(item.text)
                        else:
                            content_parts.append(str(item))
                    final_result = (
                        "\n".join(content_parts) if content_parts else str(result)
                    )
                    return final_result

                return str(result)

    def _format_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Format arguments based on parameter definitions"""
        formatted = {}

        # Handle parameter name mapping for specific tools
        param_mappings = {
            "calculate": {"math_expression": "expression"}  # Calculator tool mapping
        }

        current_mapping = param_mappings.get(self.mcp_tool_name, {})

        for key, value in args.items():
            if value is None:
                continue

            # Apply parameter name mapping if exists
            target_key = current_mapping.get(key, key)

            param = self.parameters.get(key, {})
            param_type = param.get("type", "string")

            if param_type == "integer":
                formatted[target_key] = (
                    int(value) if not isinstance(value, int) else value
                )
            elif param_type == "number":
                formatted[target_key] = (
                    float(value) if not isinstance(value, (int, float)) else value
                )
            elif param_type == "boolean":
                formatted[target_key] = (
                    bool(value) if not isinstance(value, bool) else value
                )
            elif param_type == "array":
                formatted[target_key] = (
                    list(value) if not isinstance(value, list) else value
                )
            else:
                formatted[target_key] = (
                    str(value) if not isinstance(value, str) else value
                )

        return formatted

    @Tool.handle_run
    async def async_run(
        self,
        args: Dict,
        user_id: str = None,
        agent_id: str = None,
        session_id: str = None,
    ):
        """Execute the MCP tool and return result"""
        result = await self._call_mcp_tool(args)
        return {"output": result}

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        """Start an async task"""
        args = self.prepare_args(task.args)
        result = await self._call_mcp_tool(args)

        task.update(status="completed", result=[{"output": result}])

        return str(task.id)

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        """Wait for task completion"""
        task.reload()

        if task.status == "completed":
            return {"status": "completed", "result": task.result}
        elif task.status == "failed":
            return {"status": "failed", "error": task.error}
        else:
            return {"status": task.status}

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        """Cancel a task"""
        if task.status not in ["completed", "failed", "cancelled"]:
            task.update(status="cancelled")
