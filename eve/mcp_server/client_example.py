"""
Example client for interacting with the Eden MCP server.
This demonstrates how external clients can use Eden tools via MCP.
"""

import asyncio
import json
from typing import Dict, Any

# Note: This is a simplified example. In practice, you would use the official MCP client
# or implement the MCP protocol properly.


class EdenMCPClient:
    """Simple client for interacting with Eden tools via MCP"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        from eve.mcp_server.server import list_available_tools
        return await list_available_tools(self.api_key)
    
    async def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for a specific tool"""
        from eve.mcp_server.server import get_tool_schema
        return await get_tool_schema(tool_name, self.api_key)
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        from eve.mcp_server.server import execute_tool
        return await execute_tool(tool_name, arguments, self.api_key)
    
    async def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """Check the status of a task"""
        from eve.mcp_server.server import check_task_status
        return await check_task_status(task_id, self.api_key)


async def demo_usage():
    """Demonstrate how to use the Eden MCP client"""
    
    # Initialize client with your API key
    api_key = "your-eden-api-key-here"
    client = EdenMCPClient(api_key)
    
    try:
        # List available tools
        print("=== Available Tools ===")
        tools_response = await client.list_tools()
        if tools_response.get("status") == "success":
            for tool in tools_response["tools"][:5]:  # Show first 5 tools
                print(f"- {tool['key']}: {tool['description']}")
        else:
            print(f"Error: {tools_response.get('error')}")
            return
        
        # Get schema for a specific tool
        tool_name = "websearch"  # Example tool
        print(f"\n=== Schema for {tool_name} ===")
        schema_response = await client.get_tool_schema(tool_name)
        if schema_response.get("status") == "success":
            schema = schema_response["schema"]
            print(json.dumps(schema, indent=2))
        else:
            print(f"Error: {schema_response.get('error')}")
            return
        
        # Execute a tool
        print(f"\n=== Executing {tool_name} ===")
        arguments = {
            "query": "latest AI news",
            "num_results": 3
        }
        
        result = await client.execute_tool(tool_name, arguments)
        if result.get("status") == "completed":
            print("Tool executed successfully!")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Cost: {result.get('cost')} manna")
            print("Result:", json.dumps(result.get("result"), indent=2)[:500] + "...")
        else:
            print(f"Error: {result.get('error')}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")


async def modal_client_example():
    """Example of using the Modal functions directly"""
    
    # This would be used when calling the Modal app functions directly
    # rather than through the MCP protocol
    
    api_key = "your-eden-api-key-here"
    
    # Import the Modal functions
    from eve.mcp_server.modal_app import execute_tool_call, list_tools
    
    try:
        # List tools
        tools = await list_tools.remote(api_key)
        print("Tools:", tools)
        
        # Execute a tool
        result = await execute_tool_call.remote(
            "websearch",
            {"query": "Eden AI platform", "num_results": 2},
            api_key
        )
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Eden MCP Client Demo ===")
    print("This demo shows how to interact with Eden tools via MCP")
    print("Make sure to replace 'your-eden-api-key-here' with a valid API key")
    print()
    
    # Run the demo
    asyncio.run(demo_usage())