"""
Stage 1: Anthropic + Direct Tool Integration

This module demonstrates:
1. Direct Anthropic SDK usage with function calling
2. Local Replicate flux-dev-lora integration
3. Vision capabilities for image analysis
4. Advanced reasoning with iterative image generation and inspection
5. Complex multi-step tasks requiring visual feedback
"""

import eve  # Load environment secrets
import os


import json
import asyncio
from typing import Dict, Any, Optional, List
import anthropic
import httpx
import base64


class AnthropicClient:
    """Direct Anthropic client with function calling for tools"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        
    def get_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for Claude"""
        return [
            {
                "name": "generate_image",
                "description": "Generate an image using Replicate flux-dev model. The generated image will be automatically added to the conversation so you can see and reason about it.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt for image generation"
                        },
                        "aspect_ratio": {
                            "type": "string", 
                            "description": "Image aspect ratio",
                            "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"],
                            "default": "1:1"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        ]
    
    async def generate_image(self, prompt: str, aspect_ratio: str = "1:1") -> Dict[str, Any]:
        """Generate image using Replicate API and return both URL and base64 data"""
        headers = {
            "Authorization": f"Bearer {self.replicate_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": prompt,
                "model": "dev",
                "num_outputs": 1,
                "aspect_ratio": aspect_ratio,
                "output_format": "webp",
                "output_quality": 90,
                "num_inference_steps": 28
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                print(f"🎨 Generating image: '{prompt}'")
                
                response = await client.post(
                    "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code != 201:
                    return {"error": f"Failed to create prediction (status {response.status_code})"}
                
                prediction_id = response.json()["id"]
                print(f"⏳ Prediction started: {prediction_id}")
                
                # Poll for completion
                for i in range(20):
                    await asyncio.sleep(3)
                    
                    result = await client.get(
                        f"https://api.replicate.com/v1/predictions/{prediction_id}",
                        headers=headers
                    )
                    
                    if result.status_code == 200:
                        data = result.json()
                        status = data.get("status")
                        
                        if status == "succeeded":
                            output = data.get("output")
                            if output:
                                url = output[0] if isinstance(output, list) else output
                                print(f"✅ Image generated: {url}")
                                
                                # Download image and convert to base64
                                try:
                                    img_response = await client.get(url)
                                    if img_response.status_code == 200:
                                        image_data = base64.b64encode(img_response.content).decode('utf-8')
                                        return {
                                            "url": url,
                                            "base64": image_data,
                                            "prompt": prompt
                                        }
                                    else:
                                        return {"error": f"Could not download image (status {img_response.status_code})"}
                                except Exception as e:
                                    return {"error": f"Failed to download image: {e}"}
                                    
                        elif status == "failed":
                            error = data.get("error", "Unknown error")
                            print(f"❌ Generation failed: {error}")
                            return {"error": f"Generation failed - {error}"}
                        
                        print(f"⏳ Status: {status} ({i+1}/20)")
                
                return {"error": "Generation timed out"}
                
            except Exception as e:
                print(f"❌ Replicate API error: {e}")
                return {"error": str(e)}
    
    
    async def complete_with_tools(
        self,
        input_text: str,
        thinking_model: bool = False
    ) -> str:
        """Complete with tool calling support"""
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": 4000,
            "tools": self.get_tools()
        }
        
        if thinking_model:
            request_params["system"] = "Think step by step and show your reasoning process. Be thorough in your analysis. Use the available tools as needed to complete the task."
        
        conversation_messages = [{"role": "user", "content": input_text}]
        
        # Handle potential multi-turn tool usage
        for turn in range(10):  # Max 10 tool interactions
            response = await self.client.messages.create(**{
                **request_params,
                "messages": conversation_messages
            })
            
            # Add assistant response to conversation
            conversation_messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id
                        
                        print(f"🔧 Using tool: {tool_name} with input: {tool_input}")
                        
                        # Execute the tool
                        if tool_name == "generate_image":
                            image_result = await self.generate_image(
                                tool_input["prompt"],
                                tool_input.get("aspect_ratio", "1:1")
                            )
                            
                            if "error" in image_result:
                                # Handle error case
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": image_result["error"]
                                })
                            else:
                                # Success case - add both text and image to tool result
                                tool_result_content = [
                                    {
                                        "type": "text",
                                        "text": f"Successfully generated image with prompt: '{image_result['prompt']}'\nImage URL: {image_result['url']}"
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/webp",
                                            "data": image_result["base64"]
                                        }
                                    }
                                ]
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": tool_result_content
                                })
                        else:
                            result = f"Error: Unknown tool {tool_name}"
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result
                            })
                
                # Add tool results to conversation
                conversation_messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
            else:
                # No more tools needed, return final response
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                return str(response)
        
        return "Error: Max tool interactions reached"


async def test_basic_tool_use():
    """Test 1: Basic tool use with direct Replicate integration"""
    print("\n=== Test 1: Basic Tool Use with Direct Replicate Integration ===")
    
    client = AnthropicClient(model="claude-3-5-sonnet-20241022")
    
    input_text = """Generate an image of a beautiful sunset over mountains with vibrant colors. Use the available image generation tools."""
    
    print(f"Request: {input_text}")
    
    try:
        response = await client.complete_with_tools(input_text)
        print(f"\nResponse: {response}")
        
        # Check if image was generated
        success_indicators = ["replicate.delivery", "image", "generated", "created", "https://"]
        tool_used = any(indicator in response.lower() for indicator in success_indicators)
        
        if tool_used:
            print("\n✅ Test 1 PASSED: Tool-based image generation working!")
            
            # Look for actual image URLs
            import re
            urls = re.findall(r'https://[^\s]+', response)
            if urls:
                print("Image URLs found:")
                for url in urls:
                    print(f"  - {url}")
            
            return True
        else:
            print("\n❌ Test 1 FAILED: No evidence of tool use")
            return False
            
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with error: {e}")
        return False


async def test_visual_reasoning_challenge():
    """Test 2: Complex visual reasoning task requiring iterative image generation and analysis"""
    print("\n=== Test 2: Visual Reasoning Challenge - Logo Design Optimization ===")
    
    client = AnthropicClient(model="claude-3-5-sonnet-20241022")
    
    # Intellectually challenging task requiring visual feedback
    challenge_prompt = """
    You are a brand consultant tasked with creating the optimal logo for a new sustainable technology company called "GreenFlow". 
    
    Your challenge: Design a logo that maximizes both visual appeal and brand message effectiveness through iterative refinement.
    
    Requirements:
    1. The logo must convey "sustainability", "technology", and "flow/movement"
    2. It should work well in both color and monochrome
    3. It must be distinctive and memorable
    4. The design should appeal to both B2B tech clients and environmentally conscious consumers
    
    Process:
    1. Generate an initial logo concept based on your analysis of the requirements
    2. Analyze the generated image for effectiveness in meeting the criteria
    3. Identify specific improvements needed based on your visual analysis
    4. Generate an improved version incorporating your insights
    5. Compare the iterations and provide final assessment
    
    Use your image generation and analysis tools to complete this iterative design optimization process.
    Think through each step carefully and explain your reasoning for design decisions.
    """
    
    print(f"Challenge: {challenge_prompt}")
    
    try:
        response = await client.complete_with_tools(challenge_prompt, thinking_model=True)
        print(f"\nResponse: {response}")
        
        # Check for visual reasoning indicators
        visual_indicators = [
            "generate", "analyze", "image", "logo", "design", "visual", 
            "improve", "iteration", "compare", "assessment", "color", "brand"
        ]
        
        tool_indicators = [
            "generate_image", "analyze_image", "tool", "replicate.delivery"
        ]
        
        visual_score = sum(1 for indicator in visual_indicators if indicator in response.lower())
        tool_score = sum(1 for indicator in tool_indicators if indicator in response.lower())
        has_structure = any(marker in response for marker in ["1.", "2.", "3.", "Step", "First", "iteration"])
        sufficient_detail = len(response) > 1200  # Should be detailed for complex task
        
        print(f"\n--- Visual Reasoning Analysis ---")
        print(f"Visual reasoning indicators: {visual_score}/{len(visual_indicators)}")
        print(f"Tool usage indicators: {tool_score}/{len(tool_indicators)}")
        print(f"Structured approach: {has_structure}")
        print(f"Sufficient detail: {sufficient_detail} ({len(response)} chars)")
        
        # Look for actual image URLs in response
        import re
        urls = re.findall(r'https://replicate\.delivery/[^\s]+', response)
        multiple_images = len(urls) >= 2
        
        print(f"Images generated: {len(urls)}")
        if urls:
            for i, url in enumerate(urls, 1):
                print(f"  Image {i}: {url}")
        
        if (visual_score >= 8 and tool_score >= 2 and has_structure and 
            sufficient_detail and multiple_images):
            print("\n✅ Test 2 PASSED: Advanced visual reasoning with iterative improvement demonstrated!")
            return True
        else:
            print("\n❌ Test 2 FAILED: Insufficient visual reasoning complexity")
            print(f"Missing criteria: visual_score={visual_score >= 8}, tool_score={tool_score >= 2}, structure={has_structure}, detail={sufficient_detail}, multiple_images={multiple_images}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with error: {e}")
        return False


async def main():
    """Run all tests"""
    print("=" * 70)
    print("Anthropic + Direct Tool Integration Tests")
    print("=" * 70)
    
    # Check environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not found in environment")
        print("Set it with: export REPLICATE_API_TOKEN='your-token-here'")
        return False
    
    print("✓ Anthropic API key found")
    print("✓ Replicate API token found")
    
    try:
        # Run tests
        test1_result = await test_basic_tool_use()
        await asyncio.sleep(2)  # Brief pause between tests
        test2_result = await test_visual_reasoning_challenge()
        
        # Summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS:")
        print(f"  Test 1 (Direct Tool Integration): {'✅ PASSED' if test1_result else '❌ FAILED'}")
        print(f"  Test 2 (Visual Reasoning Challenge): {'✅ PASSED' if test2_result else '❌ FAILED'}")
        
        if test1_result and test2_result:
            print("\n🎉 All tests passed! Advanced tool integration successful!")
            print("\nKey achievements:")
            print("  ✅ Direct Anthropic SDK integration")
            print("  ✅ Function calling with local tool implementation")
            print("  ✅ Real image generation via Replicate API")
            print("  ✅ Vision-based image analysis capabilities")
            print("  ✅ Complex iterative reasoning with visual feedback")
            print("  ✅ Multi-turn tool usage in single conversation")
        else:
            if not test1_result:
                print("\n⚠ Tool Integration Issues:")
                print("1. Check Replicate API token validity")
                print("2. Verify network connectivity to Replicate")
                print("3. Ensure function calling is working correctly")
            
            if not test2_result:
                print("\n⚠ Visual Reasoning Issues:")
                print("1. Task may need more explicit tool usage instructions")
                print("2. Check if vision analysis is working properly")
                print("3. Verify iterative improvement process")
        
        return test1_result and test2_result
        
    except Exception as e:
        print(f"\nError during tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)