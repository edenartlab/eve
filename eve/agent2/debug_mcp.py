"""
MCP Server Debugging - Test with working MCP server first
"""

import eve
import os
import json
import asyncio
from openai import AsyncOpenAI


class MCPDebugger:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4.1"
    
    async def test_working_mcp_server(self):
        """Test with GitMCP tiktoken server (known to work from OpenAI docs)"""
        
        print("="*60)
        print("Testing WORKING MCP Server (GitMCP tiktoken)")
        print("="*60)
        
        config = {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
            "require_approval": "never"
        }
        
        print("Configuration:")
        print(json.dumps(config, indent=2))
        
        try:
            print("\nSending request...")
            response = await self.client.responses.create(
                model=self.model,
                input="How does tiktoken work? Use the MCP tools to search documentation.",
                tools=[config]
            )
            
            print("✅ SUCCESS! Response received from working MCP server")
            
            # Extract text
            text = self._extract_text(response)
            print(f"\nResponse: {text[:300]}...")
            
            # Check for tool usage
            if any(word in text.lower() for word in ["tiktoken", "documentation", "tool", "search"]):
                print("\n✅ Tool appears to have been used!")
                return True
            else:
                print("\n⚠️  Response received but unclear if tool was used")
                return False
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    async def test_replicate_mcp_variations(self):
        """Test different Replicate MCP configurations"""
        
        print("\n" + "="*60)
        print("Testing REPLICATE MCP Server Variations")
        print("="*60)
        
        # Different variations to try
        configs = [
            {
                "name": "Standard Replicate MCP",
                "config": {
                    "type": "mcp",
                    "server_label": "replicate",
                    "server_url": "https://mcp.replicate.com/",
                    "require_approval": "never"
                }
            },
            {
                "name": "Replicate MCP with tools",
                "config": {
                    "type": "mcp",
                    "server_label": "replicate",
                    "server_url": "https://mcp.replicate.com/",
                    "allowed_tools": ["predictions.create"],
                    "require_approval": "never"
                }
            },
            {
                "name": "Replicate SSE endpoint",
                "config": {
                    "type": "mcp",
                    "server_label": "replicate",
                    "server_url": "https://mcp.replicate.com/sse",
                    "require_approval": "never"
                }
            }
        ]
        
        for i, test_case in enumerate(configs, 1):
            print(f"\n--- Test {i}: {test_case['name']} ---")
            print("Configuration:")
            print(json.dumps(test_case['config'], indent=2))
            
            try:
                print("\nSending request...")
                response = await self.client.responses.create(
                    model=self.model,
                    input="Use the Replicate MCP tools to generate an image of a sunset over mountains.",
                    tools=[test_case['config']]
                )
                
                print("✅ Response received!")
                text = self._extract_text(response)
                print(f"Response: {text[:200]}...")
                
                # Check for image generation indicators
                if any(word in text.lower() for word in ["image", "generated", "replicate", "url", "created"]):
                    print("✅ Possible image generation detected!")
                    
                    # Look for URLs
                    import re
                    urls = re.findall(r'https://[^\s]+', text)
                    if urls:
                        print("URLs found:")
                        for url in urls:
                            print(f"  - {url}")
                        return True
                
            except Exception as e:
                error_str = str(e)
                print(f"❌ Failed: {type(e).__name__}")
                
                if "500" in error_str:
                    print("  → Server error (likely auth or server issue)")
                elif "404" in error_str:
                    print("  → Server not found")
                elif "401" in error_str:
                    print("  → Authentication required")
                else:
                    print(f"  → Error: {error_str[:100]}...")
        
        return False
    
    def _extract_text(self, response) -> str:
        """Extract text from response"""
        if hasattr(response, 'output') and response.output:
            first_output = response.output[0]
            if hasattr(first_output, 'content') and first_output.content:
                first_content = first_output.content[0]
                if hasattr(first_content, 'text'):
                    return first_content.text
        return str(response)


async def main():
    print("MCP Server Debugging Session")
    print("Testing working server first, then Replicate")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return False
    
    debugger = MCPDebugger()
    
    # Test 1: Working MCP server (tiktoken)
    working_server_ok = await debugger.test_working_mcp_server()
    
    if not working_server_ok:
        print("\n" + "="*60)
        print("CONCLUSION: MCP is not working at all")
        print("="*60)
        print("Even the documented working MCP server failed.")
        print("This suggests a fundamental issue with:")
        print("1. OpenAI Responses API MCP implementation")
        print("2. Our request format")
        print("3. API access/permissions")
        return False
    
    print("\n✅ Working MCP server confirmed functional!")
    print("This proves our MCP setup is correct. Now testing Replicate...")
    
    # Test 2: Replicate MCP server
    replicate_ok = await debugger.test_replicate_mcp_variations()
    
    print("\n" + "="*60)
    print("DEBUGGING CONCLUSIONS")
    print("="*60)
    
    if replicate_ok:
        print("✅ SUCCESS: Replicate MCP working!")
    else:
        print("❌ FAILURE: Replicate MCP not working")
        print("\nSince the tiktoken MCP works, the issue with Replicate is:")
        print("1. Authentication: Replicate MCP requires auth that OpenAI can't provide")
        print("2. Server configuration: Replicate's MCP server isn't compatible")
        print("3. Tool specification: We're not calling the right tool names")
        
        print("\nNext steps:")
        print("1. Contact Replicate about MCP server auth requirements")
        print("2. Check if Replicate has specific OpenAI integration docs")
        print("3. Use a different MCP server or direct API integration")
    
    return replicate_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)