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
        "preserve_pitch": False
    }
    
    # Create dynamic description of what we're doing
    if "total_frames" in args:
        action_desc = f"Creating a {args['total_frames']}-frame video at {args['target_fps']}fps"
        duration_desc = f"({args['total_frames']/args['target_fps']:.1f} seconds)"
    elif "duration" in args:
        action_desc = f"Creating a {args['duration']:.1f}-second video at {args['target_fps']}fps"
        duration_desc = f"({int(args['duration'] * args['target_fps'])} frames)"
    else:
        action_desc = f"Preserving original duration at {args['target_fps']}fps"
        duration_desc = ""
    
    print(f"Starting time remapping test...")
    print(f"{action_desc} {duration_desc}")
    print(f"Using {args['method']} interpolation with blend strength {args['blend_strength']}")
    if args['loop_seamless']:
        print("Creating seamless loop")
    
    # Import and run the handler
    from eve.tools.media_utils.time_remapping.handler import handler
    result = await handler(args, db=None)
    
    print(f"\nOutput video saved to: {result['output']}")
    print(f"You can play it with: ffplay {result['output']}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_time_remapping())