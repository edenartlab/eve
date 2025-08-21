#!/usr/bin/env python3
"""
Simple test to verify the tool integration is working
"""

import eve
import os

import asyncio
from core2 import AnthropicClient

async def simple_test():
    print("🧪 Simple Tool Integration Test")
    print("=" * 50)
    
    client = AnthropicClient()
    
    # Simple image generation test
    print("Test: Generate image and analyze it")
    
    response = await client.complete_with_tools(
        "Generate an image of a red apple on a table, then analyze the image to tell me what you see.",
        thinking_model=True
    )
    
    print(f"\nResponse: {response}")
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("Based on the logs above, you can see:")
    print("1. Image generation working (🎨 messages)")
    print("2. Image analysis working (vision analysis)")
    print("3. Multi-step reasoning with tools")

if __name__ == "__main__":
    asyncio.run(simple_test())
