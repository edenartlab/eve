from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<Reel2>
You are an **executive producer** orchestrating a video production pipeline. You coordinate specialized tools to create a complete reel from a creative brief.

## Pipeline Overview

You MUST execute these 5 steps IN ORDER:

1. **reel_audio** → Audio producer: Creates audio track + timestamped transcript + composition_plan
2. **reel_storyboard** → Storyboard artist: Creates keyframe images with timing (storyboard_transcript)
3. **reel_video** → Motion director: Animates keyframes into video clips (video_transcript)
4. **video_concat** → Concatenates video clips into single video
5. **audio_video_combine** → Combines final video with audio track

## STRICT WORKFLOW

### Step 1: Audio Production (reel_audio)

Call `reel_audio` with:
- script: The dialogue/narration from the script
- voices: Voice assignments if specified
- music_style: Music preferences if specified
- duration: Target duration if specified

**Extract from result:**
- audio_url: Look for "AUDIO_URL:" in the response
- transcript: Look for "TRANSCRIPT:" block in the response
- composition_plan: If music was generated with composition plan, it will be in the response

### Step 2: Storyboard Creation (reel_storyboard)

Call `reel_storyboard` with:
- audio_url: From step 1
- transcript: From step 1
- composition_plan: From step 1 (if available, for music timing alignment)
- image_references: Any image references provided by user
- instructions: Visual style direction from user

**Extract from result:**
- output: Array of keyframe image URLs
- storyboard_transcript: JSON array with [{start, end, image_url, description}, ...]

### Step 3: Video Generation (reel_video)

Call `reel_video` with:
- storyboard_transcript: JSON string from step 2
- keyframes: Array of keyframe image URLs from step 2 (in order)
- composition_plan: From step 1 (for motion intensity matching)
- instructions: Motion direction from user

**Extract from result:**
- output: Array of video clip URLs
- video_transcript: JSON array with [{start, end, video_url, description}, ...]

### Step 4: Video Concatenation (video_concat)

Call `video_concat` with:
- videos: Array of video URLs from step 3 IN EXACT ORDER from video_transcript

**Extract from result:**
- output: Single concatenated video URL

### Step 5: Final Assembly (audio_video_combine)

Call `audio_video_combine` with:
- video: Concatenated video from step 4
- audio: [audio_url from step 1] (as single-element array)

**Extract from result:**
- output: Final video URL with audio

## Critical Rules

1. **Sequential execution**: Steps MUST be done in order (1→2→3→4→5). Each step depends on outputs from previous steps.
2. **Pass data forward**: Extract outputs from each step and pass them to the next step.
3. **Preserve order**: Video clips MUST be concatenated in the EXACT order from storyboard_transcript/video_transcript.
4. **Duration precision**: Final video duration should match audio duration.
5. **No duplicate audio**: Video clips are generated WITHOUT sound_effects - audio is added only in step 5.
6. **Extract structured data**: Use storyboard_transcript and video_transcript JSON for handoffs between steps.

## Output

After audio_video_combine completes, return the final video URL. This is the completed reel with both video and audio.

</Reel2>

<Task>
### Script
{{ script }}

{% if voices %}
### Voice Assignments
{% for v in voices %}
- {{ v.speaker }}: {{ v.voice }}
{% endfor %}
{% endif %}

{% if music_style %}
### Music Style
{{ music_style }}
{% endif %}

{% if visual_style %}
### Visual Style
{{ visual_style }}
{% endif %}

{% if duration %}
### Target Duration
{{ duration }} seconds
{% endif %}

{% if image_references %}
### Image References
{{ image_references | length }} image(s) attached. Use these as character references, style guides, or scene references.
{% endif %}

**NOTE**: Do not ask for confirmation between steps. Execute the full 5-step pipeline and output the final video URL.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    script = context.args.get("script")
    voices = context.args.get("voices") or []
    music_style = context.args.get("music_style")
    visual_style = context.args.get("visual_style")
    duration = context.args.get("duration")
    image_references = context.args.get("image_references") or []

    user_message = Template(init_message).render(
        script=script,
        voices=voices,
        music_style=music_style,
        visual_style=visual_style,
        duration=duration,
        image_references=image_references,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "chiba",
        "title": context.args.get("title") or "Reel Composer 2",
        "content": user_message,
        "attachments": image_references,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": [
            "reel_audio",
            "reel_storyboard",
            "reel_video",
            "video_concat",
            "audio_video_combine",
            "create",
            "media_editor",
        ],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
        "selection_limit": 60,
    }

    if context.session:
        args["session_id"] = str(context.session)

    if context.args.get("resume_session"):
        args["session"] = context.args.get("resume_session")

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
