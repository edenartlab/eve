#!/usr/bin/env python3
"""
Test visual reasoning with embedded images in conversation history
"""

import eve
import os

import asyncio
from core2 import AnthropicClient

async def test_embedded_images():
    print("🧪 Testing Visual Reasoning with Embedded Images")
    print("=" * 60)
    
    client = AnthropicClient()
    
    # Test iterative design with visual feedback
    prompt = """
    Generate an image of a simple logo design for a coffee shop called "Bean There".
    
    After you generate the image, look at it carefully and tell me:
    1. What visual elements do you see in the logo?
    2. How well does it convey "coffee shop" branding?
    3. What improvements would you make?
    
    Then generate an improved version based on your analysis.
    """
    
    print("Prompt:", prompt)
    print("\n" + "=" * 60)
    
    response = await client.complete_with_tools(prompt, thinking_model=True)
    
    print(f"\nFinal Response:\n{response}")
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("Look for evidence that Claude:")
    print("1. Generated the first image")
    print("2. Analyzed what it could see in the image")
    print("3. Made specific observations about the design")
    print("4. Generated an improved version")

if __name__ == "__main__":
    asyncio.run(test_embedded_images())
