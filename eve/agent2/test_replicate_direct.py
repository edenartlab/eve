#!/usr/bin/env python3
"""
Test that Replicate API token works directly
"""

import eve
import os
import asyncio
import httpx


async def test_direct_replicate():
    """Test Replicate API directly to verify token"""
    
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("❌ REPLICATE_API_TOKEN not found")
        return False
    
    print(f"✓ Token found: {api_token[:10]}...")
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": "sunset over mountains",
            "model": "dev",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp"
        }
    }
    
    async with httpx.AsyncClient() as client:
        print("\nCreating prediction...")
        response = await client.post(
            "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 201:
            print(f"Error: {response.text}")
            return False
        
        prediction = response.json()
        prediction_id = prediction["id"]
        print(f"✓ Prediction created: {prediction_id}")
        
        # Poll for result
        for i in range(20):
            await asyncio.sleep(3)
            print(f"Checking status... ({i+1}/20)")
            
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
                        image_url = output[0] if isinstance(output, list) else output
                        print(f"\n✅ SUCCESS! Real image generated:")
                        print(f"URL: {image_url}")
                        return True
                elif status == "failed":
                    print(f"❌ Failed: {data.get('error')}")
                    return False
                else:
                    print(f"  Status: {status}")
        
        print("❌ Timed out")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_direct_replicate())
    print("\n" + "=" * 50)
    if success:
        print("✅ Replicate API token is valid and working!")
        print("\n⚠ Issue: OpenAI Responses API cannot pass auth to MCP server")
        print("\nThe problem:")
        print("1. Replicate MCP server requires authentication")
        print("2. OpenAI strips headers when connecting to MCP servers")
        print("3. No documented way to pass API tokens to MCP servers via OpenAI")
        print("\nSolutions:")
        print("1. OpenAI needs to fix MCP auth support")
        print("2. Use a proxy MCP server that embeds your token")
        print("3. Use direct Replicate API calls (works now)")
    else:
        print("❌ Replicate API token test failed")
    
    exit(0 if success else 1)