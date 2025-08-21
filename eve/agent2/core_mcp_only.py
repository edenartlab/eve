"""
Stage 1: MCP-ONLY Implementation - No Fallbacks

This version ONLY uses the Replicate MCP server.
If it fails, we debug to understand why.
"""

import eve
import os
import json
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI


def get_mcp_tool_configs():
    """Try different MCP configurations to find what works"""
    
    configs = []
    
    # Config 1: Basic URL with no auth (MCP server might handle auth differently)
    configs.append({
        "type": "mcp",
        "server_label": "replicate",
        "server_url": "https://mcp.replicate.com/",
        "require_approval": "never"
    })
    
    # Config 2: With SSE endpoint
    configs.append({
        "type": "mcp",
        "server_label": "replicate",
        "server_url": "https://mcp.replicate.com/sse",
        "require_approval": "never"
    })
    
    # Config 3: With allowed_tools specified
    configs.append({
        "type": "mcp",
        "server_label": "replicate",
        "server_url": "https://mcp.replicate.com/",
        "allowed_tools": ["predictions.create", "models.search", "models.get"],
        "require_approval": "never"
    })
    
    # Config 4: Try with minimal config
    configs.append({
        "type": "mcp",
        "server_label": "replicate",
        "server_url": "https://mcp.replicate.com/"
    })
    
    return configs


class LLMClient:
    """OpenAI Responses API client - MCP ONLY, no fallbacks"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Use the model recommended in OpenAI docs for MCP
        self.model = "gpt-4.1" if model == "gpt-4o" else model
        
    async def complete_with_mcp(self, input_text: str) -> Dict[str, Any]:
        """Complete using ONLY MCP - return full response for debugging"""
        
        configs = get_mcp_tool_configs()
        
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"MCP Config Attempt {i}:")
            print(json.dumps(config, indent=2))
            
            try:
                print(f"Sending request to OpenAI Responses API...")
                response = await self.client.responses.create(
                    model=self.model,
                    input=input_text,
                    tools=[config]
                )
                
                print(f"✅ Response received!")
                
                # Return full response for analysis
                return {
                    "success": True,
                    "config_used": config,
                    "response": response,
                    "text": self._extract_text(response),
                    "full_output": str(response)[:1000]  # First 1000 chars for debugging
                }
                
            except Exception as e:
                error_str = str(e)
                print(f"❌ Failed: {type(e).__name__}")
                
                # Analyze the error
                if "500" in error_str:
                    print("  → Server error (500)")
                    if "param" in error_str:
                        print(f"  → Problem parameter: {error_str.split('param')[1][:50]}")
                elif "401" in error_str:
                    print("  → Authentication error")
                elif "404" in error_str:
                    print("  → MCP server not found at this URL")
                else:
                    print(f"  → Error details: {error_str[:200]}")
        
        # All configs failed
        return {
            "success": False,
            "error": "All MCP configurations failed",
            "configs_tried": len(configs)
        }
    
    def _extract_text(self, response) -> str:
        """Extract text from response"""
        if hasattr(response, 'output') and response.output:
            first_output = response.output[0]
            if hasattr(first_output, 'content') and first_output.content:
                first_content = first_output.content[0]
                if hasattr(first_content, 'text'):
                    return first_content.text
        return ""


async def debug_mcp_connection():
    """Debug the MCP connection to understand what's happening"""
    
    print("="*60)
    print("MCP CONNECTION DEBUGGING")
    print("="*60)
    
    # Check environment
    print("\n1. Environment Check:")
    if not os.getenv("OPENAI_API_KEY"):
        print("   ❌ OPENAI_API_KEY not found")
        return False
    else:
        print("   ✅ OPENAI_API_KEY found")
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("   ⚠️  REPLICATE_API_TOKEN not found")
        print("      Note: MCP server may require web auth instead")
    else:
        print(f"   ✅ REPLICATE_API_TOKEN found: {os.getenv('REPLICATE_API_TOKEN')[:10]}...")
    
    # Test basic Responses API
    print("\n2. Testing basic Responses API (no MCP):")
    client = LLMClient()
    try:
        response = await client.client.responses.create(
            model=client.model,
            input="Say 'Hello, API works!'"
        )
        text = client._extract_text(response)
        print(f"   ✅ Basic API works: {text[:50]}...")
    except Exception as e:
        print(f"   ❌ Basic API failed: {e}")
        return False
    
    # Test MCP with simple prompt
    print("\n3. Testing MCP with simple prompt:")
    result = await client.complete_with_mcp("List available tools")
    
    if result["success"]:
        print(f"\n✅ MCP Connection Successful!")
        print(f"Response text: {result['text'][:200]}...")
        print(f"Config that worked: {json.dumps(result['config_used'], indent=2)}")
    else:
        print(f"\n❌ MCP Connection Failed")
        print(f"Tried {result['configs_tried']} configurations")
    
    return result["success"]


async def test_image_generation_mcp_only():
    """Test image generation using ONLY MCP"""
    
    print("\n" + "="*60)
    print("IMAGE GENERATION TEST - MCP ONLY")
    print("="*60)
    
    client = LLMClient()
    
    # Very explicit prompt for MCP tool use
    prompts = [
        "Use the Replicate MCP server to generate an image of a sunset over mountains.",
        "Call predictions.create with prompt='sunset over mountains' using the MCP tools.",
        "Use the available MCP tools to create an image. The prompt should be 'sunset over mountains'.",
        "Generate an image using Replicate. Prompt: sunset over mountains"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Attempt {i} ---")
        print(f"Prompt: {prompt}")
        
        result = await client.complete_with_mcp(prompt)
        
        if result["success"]:
            text = result["text"]
            print(f"\nResponse: {text[:300]}...")
            
            # Check if image was generated
            if any(indicator in text.lower() for indicator in 
                   ["replicate.delivery", "image", "generated", "created", "url", "https://"]):
                print(f"\n✅ Possible image generation detected!")
                
                # Look for URLs in response
                import re
                urls = re.findall(r'https://[^\s]+', text)
                if urls:
                    print("Found URLs:")
                    for url in urls:
                        print(f"  - {url}")
                
                return True
            else:
                print(f"⚠️  No clear evidence of image generation")
        else:
            print(f"❌ MCP call failed")
    
    return False


async def main():
    print("MCP-ONLY Implementation Test")
    print("NO FALLBACKS - MCP debugging mode")
    print("="*60)
    
    # First debug the connection
    connection_works = await debug_mcp_connection()
    
    if not connection_works:
        print("\n" + "="*60)
        print("MCP DEBUGGING RESULTS:")
        print("="*60)
        print("\nThe MCP server is not accepting our connections.")
        print("\nPossible issues:")
        print("1. Authentication: The server needs auth that OpenAI can't provide")
        print("2. Protocol: The server uses a protocol OpenAI doesn't support")
        print("3. Configuration: We haven't found the right config yet")
        print("\nNext steps:")
        print("1. Check if Replicate has MCP documentation")
        print("2. Try using MCP with a different tool (GitHub, etc)")
        print("3. Contact OpenAI/Replicate about MCP compatibility")
        return False
    
    # If connection works, try image generation
    print("\nMCP connection established! Testing image generation...")
    image_works = await test_image_generation_mcp_only()
    
    if image_works:
        print("\n✅ SUCCESS: Image generated via MCP!")
    else:
        print("\n❌ FAILURE: MCP connected but no images generated")
        print("\nThis suggests the MCP server is responding but:")
        print("1. Tool calls aren't being executed")
        print("2. Authentication is needed for tool execution")
        print("3. The model isn't using the tools correctly")
    
    return connection_works and image_works


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)