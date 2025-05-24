"""
MCP Server for exposing Eve tools to external clients.
This server provides a Model Context Protocol interface to access Eden's tools.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool, TextContent
from pydantic import BaseModel

from eve.tool import Tool, get_tools_from_api_files, get_tools_from_mongo
from eve.auth import verify_api_key
from eve.user import User
from eve.task import Task
from eve import db

logger = logging.getLogger(__name__)

# Create MCP server instance
mcp = FastMCP("Eden Tools MCP Server")

# Global tool cache
_tool_cache: Dict[str, Tool] = {}


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    api_key: str


def load_available_tools() -> Dict[str, Tool]:
    """Load all available tools from both YAML files and MongoDB"""
    global _tool_cache
    
    if _tool_cache:
        return _tool_cache
    
    try:
        # Load tools from YAML files (for development/staging)
        yaml_tools = get_tools_from_api_files(cache=True, include_inactive=False)
        
        # Load tools from MongoDB (for production)
        mongo_tools = get_tools_from_mongo(cache=True, include_inactive=False)
        
        # Merge tools, prioritizing MongoDB tools
        all_tools = {**yaml_tools, **mongo_tools}
        
        # Filter out hidden tools and tools that shouldn't be exposed via MCP
        exposed_tools = {}
        for key, tool in all_tools.items():
            if tool.visible and tool.active:
                exposed_tools[key] = tool
        
        _tool_cache = exposed_tools
        logger.info(f"Loaded {len(exposed_tools)} tools for MCP server")
        return exposed_tools
        
    except Exception as e:
        logger.error(f"Error loading tools: {e}")
        return {}


def validate_api_key_and_get_user(api_key: str) -> User:
    """Validate API key and return associated user"""
    try:
        user_data = verify_api_key(api_key)
        user = User.from_mongo(user_data.userId)
        if not user:
            raise ValueError("User not found")
        return user
    except Exception as e:
        raise ValueError(f"Invalid API key: {e}")


@mcp.tool()
async def execute_tool(tool_name: str, arguments: dict, api_key: str) -> dict:
    """Execute an Eden tool with the provided arguments"""
    
    # Validate API key and get user
    try:
        user = validate_api_key_and_get_user(api_key)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    
    # Load available tools
    tools = load_available_tools()
    
    if tool_name not in tools:
        return {
            "error": f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}",
            "status": "failed"
        }
    
    tool = tools[tool_name]
    
    try:
        # Start the task
        task = await tool.async_start_task(
            user_id=str(user.id),
            agent_id=None,
            args=arguments,
            public=False,
            mock=False,
            is_client_platform=True
        )
        
        # Wait for task completion
        result = await tool.async_wait(task)
        
        return {
            "status": result.get("status", "completed"),
            "result": result,
            "task_id": str(task.id),
            "cost": task.cost
        }
        
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def list_available_tools(api_key: str) -> dict:
    """List all available tools that can be executed via this MCP server"""
    
    # Validate API key
    try:
        validate_api_key_and_get_user(api_key)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    
    tools = load_available_tools()
    
    tool_list = []
    for key, tool in tools.items():
        tool_info = {
            "key": key,
            "name": tool.name,
            "description": tool.description,
            "tip": tool.tip,
            "output_type": tool.output_type,
            "cost_estimate": tool.cost_estimate,
            "parameters": {}
        }
        
        # Add parameter information
        if hasattr(tool, 'parameters') and tool.parameters:
            for param_name, param_info in tool.parameters.items():
                if not param_info.get('hide_from_agent', False):
                    tool_info["parameters"][param_name] = {
                        "type": param_info.get("type"),
                        "description": param_info.get("description"),
                        "required": param_info.get("required", False),
                        "default": param_info.get("default")
                    }
        
        tool_list.append(tool_info)
    
    return {
        "status": "success",
        "tools": tool_list,
        "count": len(tool_list)
    }


@mcp.tool()
async def get_tool_schema(tool_name: str, api_key: str) -> dict:
    """Get the full schema for a specific tool"""
    
    # Validate API key
    try:
        validate_api_key_and_get_user(api_key)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    
    tools = load_available_tools()
    
    if tool_name not in tools:
        return {
            "error": f"Tool '{tool_name}' not found",
            "status": "failed"
        }
    
    tool = tools[tool_name]
    
    try:
        # Get the Anthropic schema (works for OpenAI format too)
        schema = tool.anthropic_schema(exclude_hidden=True)
        
        return {
            "status": "success",
            "tool_name": tool_name,
            "schema": schema
        }
        
    except Exception as e:
        logger.error(f"Error getting schema for tool {tool_name}: {e}")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def check_task_status(task_id: str, api_key: str) -> dict:
    """Check the status of a previously submitted task"""
    
    # Validate API key
    try:
        user = validate_api_key_and_get_user(api_key)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    
    try:
        task = Task.from_mongo(task_id)
        
        # Verify user owns this task
        if str(task.user) != str(user.id):
            return {"error": "Unauthorized to access this task", "status": "failed"}
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_status": task.status,
            "result": task.result if task.status == "completed" else None,
            "error": task.error if task.status == "failed" else None,
            "cost": task.cost,
            "created_at": task.createdAt.isoformat() if task.createdAt else None,
            "updated_at": task.updatedAt.isoformat() if task.updatedAt else None
        }
        
    except Exception as e:
        logger.error(f"Error checking task status {task_id}: {e}")
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()