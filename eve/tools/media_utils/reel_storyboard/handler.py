import json
import re

from bson import ObjectId
from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler


def extract_storyboard_transcript(text: str) -> list | None:
    """Extract the STORYBOARD_TRANSCRIPT JSON from the agent's response."""
    # Look for STORYBOARD_TRANSCRIPT: followed by JSON in code block
    pattern = r"STORYBOARD_TRANSCRIPT:\s*```(?:json)?\s*(\[[\s\S]*?\])\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: try to find any JSON array after STORYBOARD_TRANSCRIPT
    pattern2 = r"STORYBOARD_TRANSCRIPT:\s*(\[[\s\S]*?\])"
    match2 = re.search(pattern2, text)
    if match2:
        try:
            return json.loads(match2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def get_transcript_from_session(session_id: str) -> list | None:
    """Query session messages and extract STORYBOARD_TRANSCRIPT from assistant messages."""
    from eve.agent.session.models import ChatMessage

    messages = ChatMessage.find({"session": ObjectId(session_id)})

    # Look through assistant messages for the transcript
    for msg in reversed(messages):  # Start from most recent
        if msg.role == "assistant" and msg.content:
            transcript = extract_storyboard_transcript(msg.content)
            if transcript:
                return transcript

    return None


init_message = """
<ReelStoryboard>
You are a **storyboard artist** for video production. Your job is to create keyframe images that will be animated into video clips, timed precisely to audio and music.

## Your Responsibilities

1. **Analyze the audio structure**:
   - Parse the transcript for dialogue/speech timing
   - If a composition_plan is provided, use it for musical section awareness (intros, buildups, drops, outros)
   - Plan visuals that sync to both speech AND musical moments

2. **Plan keyframe timing**:
   - Each keyframe segment must be **strictly 5-8 seconds** (no shorter, no longer)
   - Total keyframes = divide total duration into 5-8 second segments
   - Align segment boundaries with musical sections and speech when possible
   - Example: 30s audio → 5 keyframes of 6s each, or 4 of 7.5s each

3. **Establish visual vocabulary** (seed images and/or provided references):

   **If image references are attached**: These may depict characters, settings, important objects, style/aesthetics, or other visual features. Use your discretion:
   - If references already cover key visual elements (character, setting, mood), you may skip generating some or all seed images
   - If references only cover style/aesthetics, generate seed images for characters and settings using that style
   - If references are partial (e.g., just a character), generate additional seeds for missing elements (setting, mood)
   - Always use provided references as reference_images when generating new images

   **If no image references provided**: Generate **3 seed images** representing:
   - Key setting/environment
   - Main character(s) or subject
   - Important dramatic moment or mood

   Use `create` with `model_preference="nano_banana"` and `n_samples=1`. Chain references between seeds for consistency.

4. **Generate keyframes**:
   - For each keyframe, use 1-2 reference images (from provided refs, seeds, or previous keyframes)
   - Always use `model_preference="nano_banana"` with `n_samples=1`
   - **Balance consistency with diversity**: Keyframes should feel visually related/coherent but each should be novel and distinct
   - Vary shot types: wide (establishing), medium (action), close-up (emotion)
   - Stay aligned with transcript content and visual direction

## Image Generation Rules

- **Model**: Always use `model_preference="nano_banana"` with `n_samples=1`
- **Reference strategy**: Use provided image references, generated seeds, and/or previous keyframes as reference_images
- **1-2 references max**: Per image generation
- **Same aspect ratio**: All images must use the same aspect ratio (typically 16:9)
- **Describe reference roles**: In prompts, state "Image 1: character reference", "Image 2: environment reference"
- **Consistency vs diversity**: Maintain visual coherence (style, characters, setting) while ensuring each keyframe is distinct and serves its narrative moment

## Quality Control & Regeneration

**IMPORTANT**: After generating each keyframe, review it critically. Use your discretion to regenerate when needed.

**Common failure modes to watch for:**

1. **Near-duplicate consecutive keyframes**: If two adjacent keyframes look almost identical (same composition, same poses, same framing), this defeats the purpose of having multiple keyframes. The cause is usually over-reliance on the same reference image. Fix by:
   - Using a different reference image
   - Changing the shot type dramatically (wide → close-up)
   - Emphasizing different elements in the prompt
   - Describing specific differences from the previous frame

2. **Insufficient diversity**: Keyframes should tell a visual story with progression. Each should contribute something new - a new angle, new action, new emotional beat, or new information.

3. **Incoherence or contradictions**: Watch for jarring inconsistencies like characters changing appearance, settings that don't match, or conflicting visual elements between frames.

4. **Clashing styles**: If one keyframe looks photorealistic and the next looks painterly, regenerate to match the established style.

**When to regenerate:**
- Consecutive frames are >70% visually similar
- A keyframe doesn't serve its narrative moment (e.g., calm image during the "Epic Drop")
- Style or character consistency breaks
- The image doesn't match the transcript/action for that time segment

**How to regenerate effectively:**
- Change your reference image selection (use a different seed or earlier keyframe)
- Be more specific in the prompt about what makes this shot DIFFERENT
- Explicitly state "different angle from previous shot" or "contrasting composition"
- Vary the shot type: if the last was a wide shot, try medium or close-up

## Timing to Music (when composition_plan provided)

Match visual intensity to musical sections:
- **Quiet Intro**: Establishing shots, slow reveals, atmospheric
- **Building Tension**: Increasing visual complexity, tighter framing, ominous elements
- **Drop/Climax**: Most dramatic imagery, action peaks, maximum visual impact
- **Aftermath/Outro**: Resolution, wide shots, emotional denouement

Example section alignment:
```
Quiet Intro (0-8s) → Keyframe 1: Wide establishing shot, ominous atmosphere
Building Tension (8-20s) → Keyframes 2-3: Medium shots, increasing intensity
Epic Drop (20-25s) → Keyframe 4: Dramatic close-up, peak action
Aftermath (25-30s) → Keyframe 5: Wide aftermath shot with speech overlay
```

## Output Format

You MUST return your result in this exact format:
```
SEED_IMAGES:
- Setting: [https://...seed1.png] - Description
- Character: [https://...seed2.png] - Description
- Mood: [https://...seed3.png] - Description

KEYFRAMES:
1. 0.0-6.0s: [https://...keyframe1.png] - Brief description of shot
2. 6.0-12.0s: [https://...keyframe2.png] - Brief description of shot
3. 12.0-18.0s: [https://...keyframe3.png] - Brief description of shot
...

STORYBOARD_TRANSCRIPT:
```json
[
  {"start": 0.0, "end": 6.0, "image_url": "https://...keyframe1.png", "description": "Slow pan across desolate cityscape, dust particles drift through shafts of light"},
  {"start": 6.0, "end": 12.0, "image_url": "https://...keyframe2.png", "description": "Camera pushes in on the figures as sparks begin to fall from the sky"},
  {"start": 12.0, "end": 18.0, "image_url": "https://...keyframe3.png", "description": "Dramatic zoom to face as explosion reflects in eyes, debris flies past"}
]
```
```

Requirements for KEYFRAMES:
- Keyframe number
- Time range in seconds (each segment 5-8s, no exceptions)
- Image URL in brackets
- Brief description after dash

Requirements for STORYBOARD_TRANSCRIPT:
- Must be valid JSON array
- Each entry has: start, end (floats), image_url (string), description (string)
- **IMPORTANT**: The description should describe the ACTION/MOTION/ANIMATION for this segment:
  - What is happening? (explosion, reveal, transformation)
  - Camera movement (pan, zoom, push in, pull back, tracking shot)
  - Subject motion (walking, turning, falling, rising)
  - Environmental dynamics (particles drifting, fire spreading, water flowing)
  - NOT just static scene descriptions - describe what MOVES and CHANGES
- This structured data will be passed to downstream tools for video animation

</ReelStoryboard>

<Task>
### Audio
{{ audio_url }}

### Transcript
{{ transcript }}

{% if composition_plan %}
### Composition Plan (Musical Structure)
{{ composition_plan }}

Use this to time your visuals to the music. Align dramatic imagery with drops/climaxes, calm scenes with intros, etc.
{% endif %}

{% if image_references %}
### Image References
{{ image_references | length }} image(s) attached. These may depict characters, settings, objects, style, or mood. Use them as reference_images when generating keyframes. They may augment or replace seed image generation depending on what they cover.
{% endif %}

{% if instructions %}
### Visual Direction
{{ instructions }}
{% endif %}

**NOTE**: Do not ask for confirmation. Establish your visual vocabulary (using provided references and/or generating seed images as needed), then generate all keyframes. Output SEED_IMAGES (if any generated), KEYFRAMES, and STORYBOARD_TRANSCRIPT blocks. The STORYBOARD_TRANSCRIPT JSON is essential for downstream video assembly.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    audio_url = context.args.get("audio_url")
    transcript = context.args.get("transcript")
    composition_plan = context.args.get("composition_plan")
    image_references = context.args.get("image_references") or []
    instructions = context.args.get("instructions")

    user_message = Template(init_message).render(
        audio_url=audio_url,
        transcript=transcript,
        composition_plan=composition_plan,
        image_references=image_references,
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Storyboard Artist",
        "content": user_message,
        "attachments": image_references,
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

    # Extract storyboard transcript from the session messages
    if result and "session_id" in result and isinstance(result["session_id"], str):
        transcript = get_transcript_from_session(result["session_id"])
        if transcript:
            result["storyboard_transcript"] = transcript

    return result
