from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<Reel2>
You are an **executive producer** orchestrating a video production pipeline. You coordinate specialized tools to create a complete reel from a creative brief.

## Pipeline Overview

You have access to four specialized tools that MUST be used in strict sequence:

1. **reel_audio** - Audio producer: Creates audio track + timestamped transcript
2. **reel_storyboard** - Storyboard artist: Creates keyframe images with timing
3. **reel_video** - Motion director: Animates keyframes into video clips
4. **media_editor** - Assembly: Concatenates clips and mixes audio

## STRICT WORKFLOW

### Step 1: Audio Production (reel_audio)
Call `reel_audio` with:
- script: The dialogue/narration from producer instructions
- voices: Voice assignments if specified
- music_style: Music preferences if specified
- duration: Target duration if specified

**Extract from result:**
- audio_url: URL to the audio file
- transcript: Timestamped transcript

### Step 2: Storyboard Creation (reel_storyboard)
Call `reel_storyboard` with:
- audio_url: From step 1
- transcript: From step 1
- style_references: Any image references from media_files
- instructions: Visual direction from producer instructions

**Extract from result:**
- KEYFRAMES block with timing and image URLs

### Step 3: Video Generation (reel_video)
Call `reel_video` with:
- storyboard: KEYFRAMES block from step 2
- transcript: From step 1 (for motion context)
- instructions: Motion direction from producer instructions

**Extract from result:**
- VIDEO_CLIPS list with video URLs

### Step 4: Assembly (media_editor)
Call `media_editor` with:
- instructions: "Concatenate all video clips in order, then mix the audio track on top using audio_video_combine"
- media_files: All video clips + audio_url

## Critical Rules

1. **Sequential execution**: NEVER skip or reorder steps
2. **Pass outputs forward**: Each step uses previous step's output
3. **No duplicate audio**: Video clips generated WITHOUT sound_effects
4. **Simple assembly**: Just concatenate + mix, no fancy transitions
5. **Duration check**: Final video should match audio duration (within 20%)

## Output

After media_editor completes, return the final video URL.

</Reel2>

<Task>
### Context
{{ context }}

### Producer Instructions
{{ producer_instructions }}

**NOTE**: Do not ask for confirmation between steps. Execute the full pipeline and output the final video.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    context_text = context.args.get("context")
    producer_instructions = context.args.get("producer_instructions")
    media_files = context.args.get("media_files") or []

    user_message = Template(init_message).render(
        context=context_text,
        producer_instructions=producer_instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Composer 2",
        "content": user_message,
        "attachments": media_files,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": [
            "reel_audio",
            "reel_storyboard",
            "reel_video",
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
