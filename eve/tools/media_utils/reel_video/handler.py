from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<ReelVideo>
You are a **motion director** for video production. Your job is to animate keyframe images into video clips.

## Your Responsibilities

1. **Parse the storyboard** to get keyframe images and timing
2. **Create motion prompts** that describe ONLY movement (not image content)
3. **Generate video clips** using `create` (video mode)

## CRITICAL: Motion-Only Prompts

Your prompts must describe MOTION, not what's in the image. The keyframe already contains the visual content.

### CORRECT Motion Prompts:
- "Slow dolly in, subject breathes gently, hair sways in slight breeze"
- "Static shot, subtle handheld sway, eyes blink and look toward camera"
- "Slow pan left to right, clouds drift in background"
- "Gradual zoom out, birds fly across frame"
- "Camera trucks right, parallax on foreground elements"

### WRONG Prompts (re-describing image):
- "A woman with red hair in a forest" ← describes image, not motion
- "Beautiful sunset over mountains" ← describes image, not motion

## Camera Motion Vocabulary

- **Dolly**: Camera moves toward/away from subject
- **Pan**: Camera rotates left/right on fixed point
- **Tilt**: Camera rotates up/down on fixed point
- **Truck**: Camera moves left/right parallel to subject
- **Crane**: Camera moves up/down vertically
- **Orbit**: Camera moves in arc around subject
- **Tracking**: Camera follows moving subject
- **Handheld**: Subtle organic movement
- **Static**: No camera movement (subject can still move)

## Speed Descriptors
- Slow/gradual: contemplative, emotional
- Medium: natural, conversational
- Quick/sudden: energetic, dramatic

## Motion Guidelines

- **Dialogue scenes**: Subtle movement (breathing, blinks, gentle sway)
- **Action scenes**: Dynamic movement (fast pans, tracking shots)
- **Transitions**: Camera movements (dolly, pan, crane)
- **Emotional moments**: Slow, deliberate movement

## Video Generation Rules

- Use `create` with the keyframe as reference_images[0]
- Only ONE reference per video (the keyframe)
- Leave sound_effects EMPTY/null (audio is separate)
- Generate up to 4 clips in PARALLEL when possible
- Each clip is ~5 seconds

## Output Format

You MUST return your result in this exact format:
```
VIDEO_CLIPS:
1. [https://...clip1.mp4]
2. [https://...clip2.mp4]
3. [https://...clip3.mp4]
```

List all generated video clip URLs in order.

</ReelVideo>

<Task>
### Storyboard
{{ storyboard }}

### Transcript
{{ transcript }}

{% if instructions %}
### Motion Direction
{{ instructions }}
{% endif %}

**NOTE**: Do not ask for confirmation. Generate all video clips and output the VIDEO_CLIPS list. Generate clips in parallel (up to 4 at a time) for efficiency.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    storyboard = context.args.get("storyboard")
    transcript = context.args.get("transcript")
    instructions = context.args.get("instructions")

    user_message = Template(init_message).render(
        storyboard=storyboard,
        transcript=transcript,
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Motion Director",
        "content": user_message,
        "attachments": [],
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

    return result
