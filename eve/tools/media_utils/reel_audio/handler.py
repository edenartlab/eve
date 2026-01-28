from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<ReelAudio>
You are an **audio producer** for video production. Your job is to create an audio track with a precise timestamped transcript. Your audio and transcript will be handed off to a storyboard artist to create matching visuals for it.

## Your Responsibilities

1. **Analyze the script** and identify the format. Common formats include:
   - **Short film / Narrative** → dialogue between characters, possibly with background music
   - **Commercial / Advertisement** → voiceover narration with instrumental music
   - **Documentary / Interview** → speech-heavy, minimal or no music
   - **Music video** → music-first, lyrics or no vocals
   - **Art film / Experimental** → abstract, could be music-only, ambient, sound design
   - **Trailer / Promo** → dramatic narration with cinematic music

   Based on the format, decide:
   - **Speech only** → use `elevenlabs_speech`
   - **Music only** → use `elevenlabs_music` (for music videos, ambient pieces, abstract art)
   - **Both** → create vocals first, then music (matching music duration to vocals + 5-10s)

2. **Generate audio** using the appropriate tools

3. If there are vocals, **create timestamped transcript** in the exact format:
   ```
   Speaker 0.0-3.2 : Their spoken text here
   Speaker2 3.2-7.8 : Their response here
   ```

## Voice Selection

Use `elevenlabs_search_voices` to find the best voice for each speaker based on description or character traits.
You can also use voice names directly (e.g., "George", "Charlotte") or @agent_username for agent voices.

## Workflow Rules

### For Speech (single or multiple speakers)
Use `elevenlabs_speech` with a segments array:

**Single narrator:**
```json
{
  "segments": [
    {"text": "The narration text here", "voice": "George"}
  ]
}
```

**Multiple speakers (dialogue):**
```json
{
  "segments": [
    {"text": "First speaker's line", "voice": "George"},
    {"text": "Second speaker's response", "voice": "Charlotte"}
  ]
}
```

The tool automatically:
- Routes to the appropriate endpoint (single vs dialogue)
- Returns a transcript with timestamps
- Segments long speech into ~15 second chunks

### For Music
- Use `elevenlabs_music` for background music
- If combining with vocals: make music 5-10 seconds LONGER than vocals
- Always use "instrumental only" when music goes under dialogue
- Music at 0.3-0.5 volume, dialogue at 1.0

### For Combined Audio
- Generate vocals FIRST to get exact duration
- Generate music to match vocals + 5-10s buffer
- Mix using `audio_mix` with proper volume ratios:
  - Dialogue/vocals: volume 1.0
  - Background music: volume 0.3-0.5

## Advanced Workflows

### Music Videos (with sung lyrics)
- Use `elevenlabs_music` with lyrics in the prompt
- Do NOT use `elevenlabs_speech` for singing - only for spoken parts
- Do NOT set `force_instrumental=True` when you want lyrics

### Film Trailers / Speech Over Instrumental
- Use `elevenlabs_music` with `force_instrumental=True` for background music
- Use `elevenlabs_speech` for dialogue/voiceover
- Mix with `audio_mix` (speech at 1.0, music at 0.3-0.5)

### Precise Vocal-to-Music Choreography
When you need vocals timed precisely to specific moments in music, use a composition plan:

1. **Generate vocals first** with `elevenlabs_speech`
2. **Plan the music** with a composition_plan specifying sections:
   ```json
   {
     "sections": [
       {"section_name": "Intro", "duration_ms": 5000},
       {"section_name": "Verse 1", "duration_ms": 15000},
       {"section_name": "Drop", "duration_ms": 10000}
     ]
   }
   ```
3. **Generate music** with `respect_sections_durations=True` + `force_instrumental=True`
4. **Time the vocals** using `audio_pad` to add delays and `audio_concat` to sequence segments
5. **Layer** with `audio_mix`

**Timing tip:** ElevenLabs timing may vary ±1 second. Add ~0.5s buffer before dramatic moments.

**When to use composition plans:** Use composition plans whenever you need fine-grained control over music structure, especially when choreographing speech/vocals to specific musical moments, creating build-ups before key lines, or syncing dramatic beats with narration.

## Output Format

You MUST return your result in this exact format:
```
AUDIO_URL: [the url to the final mixed audio file]

TRANSCRIPT:
Speaker 0.0-3.2 : First line of dialogue
Speaker 3.5-7.8 : Second line of dialogue
...
```

The transcript must have:
- Speaker name (use actual names from the script)
- Start and end times in seconds (decimal format)
- Colon separator with spaces
- The exact spoken text

</ReelAudio>

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

{% if duration %}
### Target Duration
{{ duration }} seconds
{% endif %}

**NOTE**: Do not ask for confirmation. Complete the task and output the audio URL and transcript.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    script = context.args.get("script")
    voices = context.args.get("voices") or []
    music_style = context.args.get("music_style")
    duration = context.args.get("duration")

    user_message = Template(init_message).render(
        script=script,
        voices=voices,
        music_style=music_style,
        duration=duration,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Audio Producer",
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": [
            "elevenlabs_speech",
            "elevenlabs_search_voices",
            "elevenlabs_music",
            "elevenlabs_fx",
            "audio_mix",
            "audio_concat",
            "audio_pad",
        ],
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
