import json
import os
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from bson import ObjectId
from pydantic import Field

from ..mongo import get_collection
from ..task import Task
from ..tool import Tool, ToolContext, tool_context


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
    mcp_bearer_env: Optional[str] = Field(
        None, description="Environment variable name for bearer token"
    )
    mcp_use_user_api_key: bool = Field(
        False,
        description="Use the authenticated user's API key as the bearer token",
    )
    mcp_timeout: int = Field(30, description="Timeout in seconds for MCP requests")
    mcp_user_token_url: Optional[str] = Field(
        None, description="URL to mint a short-lived user MCP token"
    )
    mcp_user_token_api_key_env: Optional[str] = Field(
        None, description="Env var containing API key for token minting"
    )
    mcp_user_token_ttl_seconds: Optional[int] = Field(
        None, description="TTL seconds for minted user MCP tokens"
    )

    def _resolve_server_url(self) -> str:
        return os.path.expandvars(self.mcp_server_url)

    def _build_url_with_auth(self) -> str:
        """Build URL with query parameters from environment variables"""
        url = self._resolve_server_url()
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

    def _get_user_api_key(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        try:
            user_obj = ObjectId(user_id)
        except Exception:
            return None

        api_keys = get_collection("apikeys")
        api_key_doc = api_keys.find_one(
            {
                "user": user_obj,
                "character": {"$exists": False},
                "deleted": {"$ne": True},
            },
            sort=[("createdAt", -1)],
        )
        if not api_key_doc:
            api_key_doc = api_keys.find_one(
                {"user": user_obj, "deleted": {"$ne": True}},
                sort=[("createdAt", -1)],
            )
        if not api_key_doc:
            return None
        return api_key_doc.get("apiKey")

    async def _fetch_user_token(self, user_id: Optional[str]) -> Optional[str]:
        if not self.mcp_user_token_url or not user_id:
            return None

        api_key_env = self.mcp_user_token_api_key_env or "EDEN_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None

        url = os.path.expandvars(self.mcp_user_token_url)
        payload: Dict[str, Any] = {"userId": user_id}
        if self.mcp_user_token_ttl_seconds:
            payload["ttlSeconds"] = int(self.mcp_user_token_ttl_seconds)

        async with httpx.AsyncClient(timeout=self.mcp_timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"X-Api-Key": api_key},
            )
            response.raise_for_status()
            data = response.json()

        return data.get("token") or data.get("access_token")

    async def _resolve_bearer_token(self, user_id: Optional[str]) -> Optional[str]:
        if self.mcp_user_token_url:
            user_token = await self._fetch_user_token(user_id)
            if user_token:
                return user_token
            raise ValueError("Failed to mint MCP user token")

        if self.mcp_use_user_api_key and user_id:
            user_key = self._get_user_api_key(user_id)
            if user_key:
                return user_key

        if self.mcp_bearer_env:
            env_value = os.getenv(self.mcp_bearer_env)
            if env_value:
                return env_value

        return None

    async def _build_headers(self, user_id: Optional[str]) -> Optional[Dict[str, str]]:
        bearer = await self._resolve_bearer_token(user_id)
        if not bearer:
            return None
        return {"Authorization": f"Bearer {bearer}"}

    async def _call_mcp_tool(self, args: Dict[str, Any], user_id: Optional[str]) -> str:
        """Call MCP tool using streamable_http"""
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = self._build_url_with_auth()
        formatted_args = self._format_args(args)
        headers = await self._build_headers(user_id)

        async with streamablehttp_client(
            url, headers=headers, timeout=self.mcp_timeout
        ) as (
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
            elif param_type == "object":
                if isinstance(value, dict):
                    formatted[target_key] = value
                elif isinstance(value, str):
                    try:
                        formatted[target_key] = json.loads(value)
                    except Exception:
                        formatted[target_key] = value
                else:
                    formatted[target_key] = value
            else:
                formatted[target_key] = (
                    str(value) if not isinstance(value, str) else value
                )

        return formatted

    def _inject_session(
        self, args: Dict[str, Any], session_id: Optional[str]
    ) -> Dict[str, Any]:
        if not session_id or not isinstance(args, dict) or not self.parameters:
            return args

        if "session" in self.parameters and "session" not in args:
            args["session"] = session_id
        elif "sessionId" in self.parameters and "sessionId" not in args:
            args["sessionId"] = session_id
        elif "session_id" in self.parameters and "session_id" not in args:
            args["session_id"] = session_id

        return args

    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        """Execute the MCP tool and return result"""
        args = dict(context.args or {})
        args = self._inject_session(args, context.session)
        result = await self._call_mcp_tool(args, context.user)
        return {"output": result}

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        """Start an async task"""
        args = dict(task.args or {})
        args = self._inject_session(args, str(task.session) if task.session else None)
        args = self.prepare_args(args)
        result = await self._call_mcp_tool(args, str(task.user) if task.user else None)

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
