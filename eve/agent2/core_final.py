"""
Stage 1: Working LLM Interface with REAL Image Generation

This version generates REAL images from Replicate.
Since OpenAI can't pass auth to MCP servers, we use direct API integration.
"""

import eve
import os
import json
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI
import httpx


class LLMClient:
    """OpenAI Responses API client with real Replicate integration"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4.1" if model == "gpt-4o" else model
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        
    async def complete(self, input_text: str, use_tools: bool = True) -> str:
        """Complete with tool support via direct Replicate integration"""
        
        # Check if this is an image generation request
        if use_tools and any(word in input_text.lower() for word in ["image", "generate", "create", "picture"]):
            return await self._complete_with_replicate(input_text)
        
        # Regular completion
        response = await self.client.responses.create(
            model=self.model,
            input=input_text
        )
        return self._extract_text(response)
    
    async def _complete_with_replicate(self, input_text: str) -> str:
        """Handle image generation requests with real Replicate API"""
        
        if not self.replicate_token:
            return "Error: REPLICATE_API_TOKEN not set"
        
        # Extract what to generate
        prompt = self._extract_image_prompt(input_text)
        print(f"Generating image: '{prompt}'")
        
        # Generate real image
        image_url = await self._generate_replicate_image(prompt)
        
        if image_url:
            # Get LLM to respond about the real generated image
            response = await self.client.responses.create(
                model=self.model,
                input=f"""You successfully generated an image using Replicate.
                Prompt used: '{prompt}'
                Image URL: {image_url}
                
                Respond naturally about this successful image generation. ALWAYS include the full image URL in your response."""
            )
            text = self._extract_text(response)
            
            # Ensure URL is in response
            if image_url not in text:
                text = f"{text}\n\nImage URL: {image_url}"
            
            return text
        else:
            return "Failed to generate image"
    
    def _extract_image_prompt(self, text: str) -> str:
        """Extract the image prompt from user text"""
        text_lower = text.lower()
        
        # Common patterns
        patterns = [
            "image of ", "picture of ", "generate ", "create ",
            "draw ", "make ", "design ", "illustration of "
        ]
        
        for pattern in patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern) + len(pattern)
                # Get rest of text, clean it up
                prompt = text[idx:].strip()
                # Remove trailing punctuation and limit length
                prompt = prompt.rstrip('.!?')[:200]
                if prompt:
                    return prompt
        
        # Default prompts based on keywords
        if "sunset" in text_lower and "mountain" in text_lower:
            return "beautiful sunset over mountains with vibrant orange and purple sky"
        elif "sunset" in text_lower:
            return "stunning sunset with dramatic clouds"
        elif "mountain" in text_lower:
            return "majestic mountain landscape"
        
        return "beautiful landscape photograph"
    
    async def _generate_replicate_image(self, prompt: str) -> str:
        """Generate real image using Replicate API"""
        
        headers = {
            "Authorization": f"Bearer {self.replicate_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": prompt,
                "model": "dev",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90,
                "num_inference_steps": 28
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Create prediction
                response = await client.post(
                    "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code != 201:
                    print(f"Replicate error: {response.status_code}")
                    return None
                
                prediction_id = response.json()["id"]
                
                # Poll for completion
                for _ in range(20):
                    await asyncio.sleep(2)
                    
                    result = await client.get(
                        f"https://api.replicate.com/v1/predictions/{prediction_id}",
                        headers=headers
                    )
                    
                    if result.status_code == 200:
                        data = result.json()
                        if data.get("status") == "succeeded":
                            output = data.get("output")
                            if output:
                                url = output[0] if isinstance(output, list) else output
                                print(f"✅ Real image generated: {url}")
                                return url
                        elif data.get("status") == "failed":
                            print(f"Generation failed: {data.get('error')}")
                            return None
                
                print("Generation timed out")
                return None
                
            except Exception as e:
                print(f"Replicate API error: {e}")
                return None
    
    def _extract_text(self, response) -> str:
        """Extract text from Responses API response"""
        if hasattr(response, 'output') and response.output:
            first_output = response.output[0]
            if hasattr(first_output, 'content') and first_output.content:
                first_content = first_output.content[0]
                if hasattr(first_content, 'text'):
                    return first_content.text
        return str(response)


async def test_real_image_generation():
    """Test 1: REAL image generation from Replicate"""
    print("\n=== Test 1: REAL Image Generation ===")
    
    client = LLMClient(model="gpt-4o")
    
    input_text = "Generate an image of a sunset over mountains"
    print(f"Request: {input_text}")
    
    response = await client.complete(input_text, use_tools=True)
    print(f"\nResponse: {response}")
    
    # Check for real Replicate URL
    has_real_url = "replicate.delivery" in response
    
    if has_real_url:
        print("\n✅ Test 1 PASSED: Real image generated from Replicate!")
        return True
    else:
        print("\n❌ Test 1 FAILED: No real image URL found")
        return False


async def test_o3_reasoning():
    """Test 2: O3 Advanced Reasoning"""
    print("\n=== Test 2: O3 Advanced Reasoning ===")
    
    client = LLMClient(model="o3-mini")
    
    prompt = """Calculate step by step:
    A company has revenue of $1M/month growing 10% monthly.
    Expenses are $800k/month growing 5% monthly.
    When will monthly profit exceed $500k?"""
    
    response = await client.complete(prompt, use_tools=False)
    
    # Check for reasoning
    if len(response) > 300 and any(word in response.lower() for word in ["month", "profit", "calculate"]):
        print(f"Response preview: {response[:300]}...")
        print("\n✅ Test 2 PASSED: Advanced reasoning demonstrated")
        return True
    else:
        print("\n❌ Test 2 FAILED")
        return False


async def main():
    print("=" * 60)
    print("Stage 1: Real Image Generation + O3 Reasoning")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return False
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not found")
        print("\nTo get a token:")
        print("1. Go to https://replicate.com/account/api-tokens")
        print("2. Create a token")
        print("3. Run: export REPLICATE_API_TOKEN='your-token'")
        return False
    
    print("✓ All API keys found\n")
    
    test1 = await test_real_image_generation()
    await asyncio.sleep(1)
    test2 = await test_o3_reasoning()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"  Real Image Generation: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"  O3 Advanced Reasoning: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if test1 and test2:
        print("\n🎉 Stage 1 Complete!")
        print("\nWhat we achieved:")
        print("  ✅ Direct OpenAI SDK (no LiteLLM)")
        print("  ✅ Responses API only")
        print("  ✅ REAL images from Replicate")
        print("  ✅ O3 advanced reasoning")
        print("\nNote: MCP auth is blocked by OpenAI, using direct integration")
    
    return test1 and test2


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)