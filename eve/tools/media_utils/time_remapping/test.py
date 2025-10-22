import asyncio
import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.insert(0, project_root)


async def test_time_remapping():
    # Test parameters
    args = {
        "video": "https://edenartlab-lfs.s3.us-east-1.amazonaws.com/comfyui/models2/assets/fire-pulse.mp4",
        "duration": 15.0,
        "target_fps": 60,
        "blend_strength": 1.0,
        "method": "cosine",
        "loop_seamless": False,
        "preserve_pitch": False,
    }

    # Import and run the handler
    from eve.tools.media_utils.time_remapping.handler import handler
    from eve.tool import ToolContext

    context = ToolContext(args=args)
    await handler(context)


# Run the test
if __name__ == "__main__":
    asyncio.run(test_time_remapping())
