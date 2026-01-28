import json
import re

from bson import ObjectId
from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler


def extract_video_transcript(text: str) -> list | None:
    """Extract the VIDEO_TRANSCRIPT JSON from the agent's response."""
    # Look for VIDEO_TRANSCRIPT: followed by JSON in code block
    pattern = r"VIDEO_TRANSCRIPT:\s*```(?:json)?\s*(\[[\s\S]*?\])\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: try to find any JSON array after VIDEO_TRANSCRIPT
    pattern2 = r"VIDEO_TRANSCRIPT:\s*(\[[\s\S]*?\])"
    match2 = re.search(pattern2, text)
    if match2:
        try:
            return json.loads(match2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def get_video_transcript_from_session(session_id: str) -> list | None:
    """Query session messages and extract VIDEO_TRANSCRIPT from assistant messages."""
    from eve.agent.session.models import ChatMessage

    messages = ChatMessage.find({"session": ObjectId(session_id)})

    # Look through assistant messages for the transcript
    for msg in reversed(messages):  # Start from most recent
        if msg.role == "assistant" and msg.content:
            transcript = extract_video_transcript(msg.content)
            if transcript:
                return transcript

    return None


init_message = """
<ReelVideo>
You are a **motion director** for video production. Your job is to animate keyframe images into video clips with **precise timing**.

## CRITICAL REQUIREMENTS

1. **ONE video clip per keyframe** - You MUST produce exactly one video for each entry in the storyboard_transcript
2. **EXACT durations** - Each clip's duration MUST match its transcript entry (end - start). This is critical for final assembly.
3. **Parallel processing** - Generate clips in batches of 4 for efficiency

## Input Format

You receive a `storyboard_transcript` JSON array:
```json
[
  {"start": 0.0, "end": 6.0, "image_url": "https://...", "description": "Camera pans across cityscape..."},
  {"start": 6.0, "end": 12.0, "image_url": "https://...", "description": "Slow dolly in as sparks fall..."}
]
```

For each entry:
- `image_url` is the keyframe to animate
- `description` contains the action/motion guidance
- `duration = end - start` is the EXACT clip duration required

## Motion-Only Prompts

Your prompts must describe MOTION, not image content. The keyframe already has the visual content.

### CORRECT Motion Prompts:
- "Slow dolly in, subject breathes gently, hair sways in slight breeze"
- "Static shot, subtle handheld sway, eyes blink slowly"
- "Slow pan left to right, clouds drift in background, particles float"
- "Dramatic zoom in, fire flickers, smoke rises"
- "Camera trucks right, parallax on foreground elements"

### WRONG Prompts (re-describing image):
- "A woman with red hair in a forest" ← describes image, not motion
- "Beautiful sunset over mountains" ← describes image, not motion

## Camera Motion Vocabulary

- **Dolly**: Camera moves toward/away from subject (dolly in/out)
- **Pan**: Camera rotates left/right on fixed point
- **Tilt**: Camera rotates up/down on fixed point
- **Truck**: Camera moves left/right parallel to subject
- **Crane**: Camera moves up/down vertically
- **Orbit**: Camera moves in arc around subject
- **Tracking**: Camera follows moving subject
- **Handheld**: Subtle organic movement/shake
- **Static**: No camera movement (subject can still move)
- **Zoom**: Lens zoom in/out (different from dolly)

## Motion Intensity by Music Section

Match motion intensity to the composition_plan if provided:
- **Quiet Intro**: Slow, subtle movements (gentle breathing, slow pan, soft handheld)
- **Building Tension**: Gradually increasing movement (tighter framing, faster pans)
- **Drop/Climax**: Dynamic, dramatic movements (quick zooms, impacts, shakes)
- **Aftermath**: Slow, contemplative movements (slow pulls, drifting particles)

## Video Generation Rules

For each keyframe in storyboard_transcript:
1. Use `create` with `output="video"`
2. Set keyframe as `reference_images[0]`
3. Set `duration` to EXACT seconds from transcript (end - start), rounded to nearest integer (5-10 range)
4. Write motion-only prompt based on description
5. Leave `sound_effects` empty (audio is separate)
6. Use default quality unless otherwise specified

**PARALLEL BATCHING**: Generate up to 4 clips simultaneously, then wait for results before next batch.

## Output Format

You MUST return your result in this exact format:
```
VIDEO_CLIPS:
1. 0.0-6.0s: [https://...clip1.mp4] - Motion: slow pan across cityscape
2. 6.0-12.0s: [https://...clip2.mp4] - Motion: dolly in with falling sparks
3. 12.0-18.0s: [https://...clip3.mp4] - Motion: dramatic zoom, fire flickers
...

VIDEO_TRANSCRIPT:
```json
[
  {"start": 0.0, "end": 6.0, "video_url": "https://...clip1.mp4", "description": "Slow pan across cityscape"},
  {"start": 6.0, "end": 12.0, "video_url": "https://...clip2.mp4", "description": "Dolly in with falling sparks"},
  {"start": 12.0, "end": 18.0, "video_url": "https://...clip3.mp4", "description": "Dramatic zoom, fire flickers"}
]
```
```

Requirements:
- Number of VIDEO_CLIPS MUST equal number of keyframes in storyboard_transcript
- Timing ranges MUST match the storyboard_transcript exactly
- VIDEO_TRANSCRIPT JSON is essential for final assembly

</ReelVideo>

<Task>
### Storyboard Transcript
{{ storyboard_transcript }}

### Keyframe Images (attached above, in order)
{% for kf in keyframes %}
{{ loop.index }}. {{ kf }}
{% endfor %}

The keyframe images are attached as visual references. They correspond 1:1 with the storyboard_transcript entries in the same order. Use each keyframe's image_url from the storyboard_transcript as reference_images[0] when generating that clip's video.

{% if composition_plan %}
### Composition Plan (Musical Structure)
{{ composition_plan }}

Match motion intensity to musical sections. Subtle movement for intros, dynamic for drops.
{% endif %}

{% if instructions %}
### Motion Direction
{{ instructions }}
{% endif %}

**CRITICAL**:
- You MUST produce exactly ONE video clip for each keyframe in the storyboard_transcript
- Each clip duration MUST match the transcript timing (end - start)
- The total combined duration must be precise for final assembly
- Generate clips in batches of 4 in parallel for efficiency
- Output both VIDEO_CLIPS and VIDEO_TRANSCRIPT blocks

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    storyboard_transcript = context.args.get("storyboard_transcript")
    keyframes = context.args.get("keyframes") or []
    composition_plan = context.args.get("composition_plan")
    instructions = context.args.get("instructions")

    user_message = Template(init_message).render(
        storyboard_transcript=storyboard_transcript,
        keyframes=keyframes,
        composition_plan=composition_plan,
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Motion Director",
        "content": user_message,
        "attachments": keyframes,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": ["create"],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
    }

    if context.session:
        args["session_id"] = str(context.session)

    # Call session_post handler directly to avoid nested Modal timeout
    session_post_context = ToolContext(
        args=args,
        user=context.user,
        agent=context.agent,
        session=context.session,
        message=context.message,
        tool_call_id=context.tool_call_id,
    )
    result = await session_post_handler(session_post_context)

    # Extract video transcript from the session messages
    if result and "session_id" in result and isinstance(result["session_id"], str):
        video_transcript = get_video_transcript_from_session(result["session_id"])
        if video_transcript:
            result["video_transcript"] = video_transcript

    return result
