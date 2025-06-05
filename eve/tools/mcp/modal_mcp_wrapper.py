"""
Generic Modal-MCP integration wrapper.

This module provides a reusable foundation for running MCP servers as Modal apps
with HTTP transport. It handles the conversion from low-level MCP servers to
FastMCP format and provides deployment utilities.
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
from contextlib import asynccontextmanager

from modal import App, web_endpoint, asgi_app
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.types import Tool, Prompt, TextContent


class ModalMCPWrapper:
    """
    Generic wrapper for running MCP servers on Modal with HTTP transport.
    
    This class bridges between low-level MCP servers and Modal deployment,
    providing a clean interface for creating Modal apps from MCP tools.
    """
    
    def __init__(
        self,
        server_name: str,
        description: str = "MCP Server on Modal",
        mount_path: str = "/mcp",
        dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the Modal MCP wrapper.
        
        Args:
            server_name: Name of the MCP server
            description: Description for the Modal app
            mount_path: HTTP path to mount the MCP server (default: /mcp)
            dependencies: List of Python package dependencies
            **kwargs: Additional arguments passed to FastMCP
        """
        self.server_name = server_name
        self.description = description
        self.mount_path = mount_path
        self.dependencies = dependencies or []
        
        # Create FastMCP server for HTTP transport
        self.mcp_server = FastMCP(
            name=server_name,
            stateless_http=True,
            dependencies=self.dependencies,
            **kwargs
        )
        
        # Create Modal app
        self.modal_app = App(f"mcp-{server_name.lower().replace(' ', '-')}")
        
        self._tools_registered = False
        self._prompts_registered = False
    
    def tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Decorator to register a tool with the MCP server.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
        """
        return self.mcp_server.tool(name=name, description=description)
    
    def prompt(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Decorator to register a prompt with the MCP server.
        
        Args:
            name: Prompt name (defaults to function name)  
            description: Prompt description (defaults to function docstring)
        """
        return self.mcp_server.prompt(name=name, description=description)
    
    def resource(self, uri_template: str):
        """
        Decorator to register a resource with the MCP server.
        
        Args:
            uri_template: URI template for the resource
        """
        return self.mcp_server.resource(uri_template)
    
    def create_modal_endpoint(self):
        """
        Create the Modal endpoint function.
        This should be called at module level, not inside a class method.
        """
        @self.modal_app.function()
        @asgi_app()
        def mcp_server_app():
            """ASGI app serving the MCP server over HTTP."""
            return self.mcp_server.streamable_http_app()
        
        return mcp_server_app
    
    def get_modal_app(self) -> App:
        """
        Get the Modal app instance.
        
        Returns:
            Modal App instance
        """
        return self.modal_app


def create_simple_mcp_modal_app(
    server_name: str,
    tools: Optional[Dict[str, Callable]] = None,
    prompts: Optional[Dict[str, Callable]] = None,
    dependencies: Optional[List[str]] = None,
    **kwargs
) -> ModalMCPWrapper:
    """
    Factory function to quickly create a Modal MCP app with tools and prompts.
    
    Args:
        server_name: Name of the MCP server
        tools: Dictionary of tool name -> function mappings
        prompts: Dictionary of prompt name -> function mappings  
        dependencies: List of Python package dependencies
        **kwargs: Additional arguments for ModalMCPWrapper
        
    Returns:
        Configured ModalMCPWrapper instance
    """
    wrapper = ModalMCPWrapper(
        server_name=server_name,
        dependencies=dependencies,
        **kwargs
    )
    
    # Register tools
    if tools:
        for tool_name, tool_func in tools.items():
            wrapper.mcp_server.tool(name=tool_name)(tool_func)
    
    # Register prompts  
    if prompts:
        for prompt_name, prompt_func in prompts.items():
            wrapper.mcp_server.prompt(name=prompt_name)(prompt_func)
    
    return wrapper


# Utility functions for common Modal MCP patterns

def convert_legacy_tool_to_fastmcp(tool_handler: Callable, tool_schema: Tool) -> Callable:
    """
    Convert a legacy MCP tool handler to FastMCP format.
    
    Args:
        tool_handler: Original tool handler function
        tool_schema: Tool schema definition
        
    Returns:
        FastMCP-compatible tool function
    """
    async def fastmcp_tool(*args, **kwargs):
        # Convert arguments to legacy format if needed
        result = await tool_handler(tool_schema.name, kwargs)
        
        # Extract text content from result
        if isinstance(result, list) and result:
            if isinstance(result[0], TextContent):
                return result[0].text
        
        return str(result)
    
    # Copy metadata
    fastmcp_tool.__name__ = tool_schema.name
    fastmcp_tool.__doc__ = tool_schema.description
    
    return fastmcp_tool