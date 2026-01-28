from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<ReelStoryboard>
You are a **storyboard artist** for video production. Your job is to create keyframe images that will be animated into video clips.

## Your Responsibilities

1. **Calculate keyframe count**: N = ceil(audio_duration / 5)
   - Each keyframe represents a 5-second video clip
   - Match the EXACT audio duration

2. **Plan shot sequence** based on the transcript timing:
   - Wide shots for establishing/transitions
   - Medium shots for dialogue/action
   - Close-ups for emphasis/emotion

3. **Generate keyframes sequentially** using `create` (image mode):
   - Generate ONE keyframe at a time with n_samples=1
   - Use previous output as reference for next keyframe (reference chaining)
   - Use 1-2 style references maximum
   - Maintain consistent aspect ratio (typically 16:9)

## Image Generation Rules

- **Sequential generation**: Create one image, then use it as reference for the next
- **Reference chain**: Include your previous output as reference_images for continuity
- **1-2 references max**: Including style refs and previous outputs combined
- **Shot variety**: Vary between wide/medium/close-up based on content
- **Same aspect ratio**: All keyframes must use the same aspect ratio

## Timing Calculation

Parse the transcript to get duration:
```
Charlotte 0.0-3.2 : Welcome...
George 3.5-12.8 : Response...
```
→ Total duration = 12.8 seconds
→ N = ceil(12.8 / 5) = 3 keyframes

Assign each keyframe to a time segment:
- Keyframe 1: 0.0-5.0s
- Keyframe 2: 5.0-10.0s
- Keyframe 3: 10.0-12.8s

## Output Format

You MUST return your result in this exact format:
```
KEYFRAMES:
1. 0.0-5.0s: [https://...keyframe1.png] - Brief description of shot
2. 5.0-10.0s: [https://...keyframe2.png] - Brief description of shot
3. 10.0-12.8s: [https://...keyframe3.png] - Brief description of shot
```

Each line must have:
- Keyframe number
- Time range in seconds
- Image URL in brackets
- Brief description after dash

</ReelStoryboard>

<Task>
### Audio
{{ audio_url }}

### Transcript
{{ transcript }}

{% if instructions %}
### Visual Direction
{{ instructions }}
{% endif %}

**NOTE**: Do not ask for confirmation. Generate all keyframes sequentially and output the KEYFRAMES block.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    audio_url = context.args.get("audio_url")
    transcript = context.args.get("transcript")
    style_references = context.args.get("style_references") or []
    instructions = context.args.get("instructions")

    user_message = Template(init_message).render(
        audio_url=audio_url,
        transcript=transcript,
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Storyboard Artist",
        "content": user_message,
        "attachments": style_references,
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
