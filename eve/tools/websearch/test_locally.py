import asyncio
from handler import handler
import json
from datetime import datetime
import os

async def test_website(url: str) -> None:
    """Test the handler with a single website and print results."""
    print(f"\n{'='*80}")
    print(f"Testing URL: {url}")
    print(f"{'='*80}")
    
    try:
        result = await handler({"url": url})
        print(result["output"])
    except Exception as e:
        print(f"Error testing {url}: {str(e)}")

async def run_tests():
    """Run tests on various types of websites."""
    results_dir = "websearch_results"
    os.makedirs(results_dir, exist_ok=True)
    
    test_urls = [
        "https://news.ycombinator.com",
        "https://huggingface.co/papers",
        "https://www.reddit.com/r/StableDiffusion/",
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    for url in test_urls:
        try:
            result = await handler({"url": url})
            results[url] = {
                "status": "success",
                "output": result["output"]
            }
        except Exception as e:
            results[url] = {
                "status": "error",
                "error": str(e)
            }
        
        output_file = os.path.join(results_dir, f"websearch_test_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        await test_website(url)
        print("\nWaiting 2 seconds before next test...")
        await asyncio.sleep(2)
    
    print(f"\nTest results saved to: {output_file}")

if __name__ == "__main__":
    print("Starting websearch tool tests...")
    asyncio.run(run_tests())