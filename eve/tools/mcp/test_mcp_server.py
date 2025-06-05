#!/usr/bin/env python3
"""
Unified MCP Server Testing Tool.

Test any MCP server deployed on Modal by providing the server name.
When no test args provided: lists tools and capabilities.
When test args provided: runs the specific tool call.
"""

import argparse
import asyncio
import json
import sys
from typing import Dict, Any, Optional, List
import httpx
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


class MCPServerTester:
    """Unified testing class for MCP servers on Modal."""
    
    def __init__(self, server_name: str, base_url: Optional[str] = None):
        """
        Initialize the MCP server tester.
        
        Args:
            server_name: Name of the Modal app (e.g., "mcp-fetch")
            base_url: Optional custom base URL override
        """
        self.server_name = server_name
        if base_url:
            self.server_url = base_url
        else:
            # Construct Modal URL from server name
            self.server_url = f"https://edenartlab--{server_name}-mcp-server-app.modal.run/mcp"
    
    async def test_connectivity(self) -> bool:
        """Test basic HTTP connectivity to the server."""
        print(f"🔗 Testing connectivity to: {self.server_url}")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.server_url)
                print(f"   Status: {response.status_code}")
                
                if response.status_code in [200, 307, 406]:  # 307=redirect, 406=method not allowed (expected for GET on MCP)
                    print("✅ Server is accessible")
                    return True
                else:
                    print(f"❌ Unexpected status code: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            return False
    
    async def list_capabilities(self) -> bool:
        """List server capabilities without running tools."""
        print(f"\n🧪 Testing MCP Protocol: {self.server_url}")
        print("=" * 60)
        
        try:
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    
                    # Initialize
                    print("📡 Initializing connection...")
                    await session.initialize()
                    print("✅ Connection established!")
                    
                    # Discover capabilities
                    print("\n🔧 Discovering server capabilities...")
                    
                    # List tools
                    try:
                        print("   Listing tools...")
                        tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                        
                        # Handle different return types
                        if hasattr(tools_result, 'tools'):
                            tools = tools_result.tools
                        elif isinstance(tools_result, (list, tuple)):
                            tools = list(tools_result)
                        else:
                            tools = [tools_result]
                        
                        print(f"Available tools: {[tool.name for tool in tools]}")
                        for tool in tools:
                            description = getattr(tool, 'description', 'No description')
                            # Show input schema if available
                            input_schema = getattr(tool, 'inputSchema', None)
                            args_info = ""
                            if input_schema and hasattr(input_schema, 'properties'):
                                props = input_schema.properties
                                args_info = f" (args: {list(props.keys())})"
                            print(f"  - {tool.name}: {description}{args_info}")
                    except asyncio.TimeoutError:
                        print("   ❌ Tool listing timed out")
                        return False
                    except Exception as e:
                        print(f"   ❌ Tool listing failed: {e}")
                        return False
                    
                    # List prompts 
                    try:
                        prompts = await session.list_prompts()
                        if prompts:
                            print(f"Available prompts: {[prompt.name for prompt in prompts]}")
                        else:
                            print("   No prompts available")
                    except Exception:
                        print("   No prompts available")
                    
                    # List resources
                    try:
                        resources = await session.list_resources()
                        if resources:
                            print(f"Available resources: {[resource.uri for resource in resources]}")
                        else:
                            print("   No resources available")
                    except Exception:
                        print("   No resources available")
                    
                    print("\n✅ Server capabilities listed successfully!")
                    return True
                    
        except Exception as e:
            print(f"💥 MCP protocol test failed: {e}")
            return False
    
    async def run_specific_tool(self, test_args: Dict[str, Any]) -> bool:
        """Run a specific tool with provided arguments."""
        print(f"\n🛠️  Running specific tool...")
        
        try:
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    
                    # Initialize
                    await session.initialize()
                    
                    # Run the specific tool call
                    for tool_name, args in test_args.items():
                        print(f"\n   Calling {tool_name} with args: {args}")
                        try:
                            result = await session.call_tool(tool_name, args)
                            # Handle different result types and show full output
                            if hasattr(result, 'content'):
                                # CallToolResult object
                                content = result.content
                                if content and len(content) > 0:
                                    text_content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                                    print(f"   ✅ {tool_name} result:")
                                    print(text_content)
                                else:
                                    print(f"   ✅ {tool_name} result: {str(result)}")
                            else:
                                # Direct string result
                                print(f"   ✅ {tool_name} result: {str(result)}")
                        except Exception as e:
                            print(f"   ❌ {tool_name} failed: {e}")
                            return False
                    
                    return True
                    
        except Exception as e:
            print(f"💥 Tool execution failed: {e}")
            return False
    
    async def run_test(self, test_args: Optional[Dict[str, Any]] = None) -> bool:
        """Run test - either list capabilities or run specific tool."""
        print(f"🚀 Testing: {self.server_name}")
        print("=" * 40)
        
        # Test connectivity
        if not await self.test_connectivity():
            return False
        
        # If test_args provided, run specific tool; otherwise list capabilities
        if test_args:
            return await self.run_specific_tool(test_args)
        else:
            return await self.list_capabilities()
    




async def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Test MCP servers deployed on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List server capabilities
  python test_mcp_server.py mcp-fetch
  
  # Run specific tool
  python test_mcp_server.py mcp-fetch \\
    --test-args '{"fetch": {"url": "https://example.com", "max_length": 1000}}'
  
  # Test with custom URL
  python test_mcp_server.py my-server --url https://custom-server.com/mcp
        """
    )
    
    parser.add_argument(
        "server_name",
        help="Name of the Modal MCP server (e.g., 'mcp-fetch')"
    )
    
    
    parser.add_argument(
        "--url",
        help="Custom server URL (overrides auto-generated Modal URL)"
    )
    
    parser.add_argument(
        "--test-args",
        help="JSON string of test arguments for tools (e.g., '{\"fetch\": {\"url\": \"https://example.com\"}}')"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Parse test arguments
    test_args = None
    if args.test_args:
        try:
            test_args = json.loads(args.test_args)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --test-args: {e}")
            sys.exit(1)
    
    # Create tester
    tester = MCPServerTester(args.server_name, args.url)
    
    # Run test
    try:
        success = await tester.run_test(test_args)
        
        if success:
            if test_args:
                print(f"\n🎉 Tool execution completed for {args.server_name}!")
            else:
                print(f"\n🎉 Server capabilities listed for {args.server_name}!")
            sys.exit(0)
        else:
            print(f"\n💥 Test failed for {args.server_name}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())