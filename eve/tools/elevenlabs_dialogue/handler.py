import asyncio
import base64
import os
from tempfile import NamedTemporaryFile
from typing import Dict, List

import httpx
from elevenlabs.client import ElevenLabs

from eve import utils
from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

DEFAULT_VOICE = "Charlotte"


def resolve_voice_id(
    voice: str,
    voice_name_map: Dict[str, str],
    voice_ids: List[str],
) -> str:
    """
    Resolve a voice identifier to an ElevenLabs voice ID.

    Args:
        voice: Can be a voice ID, voice name, or @agent_username
        voice_name_map: Map of voice names to voice IDs
        voice_ids: List of valid voice IDs

    Returns:
        Resolved ElevenLabs voice ID
    """
    # Check if it's an agent username (starts with @)
    if voice.startswith("@"):
        username = voice[1:]  # Remove the @ prefix
        from eve.agent.agent import Agent

        agent = Agent.find_one({"username": username})  # type: ignore[attr-defined]
        if not agent:
            raise ValueError(f"Agent '{username}' not found")
        if not agent.voice:
            raise ValueError(f"Agent '{username}' does not have a voice configured")
        voice = agent.voice

    # Check if it's already a valid voice ID
    if voice in voice_ids:
        return voice

    # Check if it's an exact voice name match
    if voice in voice_name_map:
        return voice_name_map[voice]

    # Check if voice name starts with the given string (handles "George" matching "George - Warm...")
    voice_lower = voice.lower()
    for name, vid in voice_name_map.items():
        # Match if name starts with the voice string (case insensitive)
        if name.lower().startswith(voice_lower):
            return vid
        # Also match the part before " - " if present
        if " - " in name and name.split(" - ")[0].lower() == voice_lower:
            return vid

    raise ValueError(
        f"Voice '{voice}' not found. Must be a valid voice ID, voice name, or @agent_username"
    )


async def handler(context: ToolContext):
    args = context.args
    segments = args["segments"]
    stability = args.get("stability", 0.5)
    style = args.get("style", 0.0)
    speed = args.get("speed", 1.0)

    async def generate_dialogue():
        def _generate():
            # Get all available voices for resolution
            response = eleven.voices.get_all()
            voice_name_map = {
                v.name: v.voice_id for v in response.voices if v.name is not None
            }
            voice_ids = [v.voice_id for v in response.voices]

            # Build inputs array with resolved voice IDs
            inputs = []
            for segment in segments:
                voice_id = resolve_voice_id(
                    segment["voice"],
                    voice_name_map,
                    voice_ids,
                )
                inputs.append(
                    {
                        "text": segment["text"],
                        "voice_id": voice_id,
                    }
                )

            # Call ElevenLabs text_to_dialogue with timestamps via HTTP API
            # The SDK doesn't have convert_with_timestamps for dialogue yet
            if not ELEVEN_API_KEY:
                raise ValueError("ELEVEN_API_KEY environment variable is not set")

            api_response = httpx.post(
                "https://api.elevenlabs.io/v1/text-to-dialogue/with-timestamps",
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": inputs,
                    "model_id": "eleven_v3",
                    "settings": {
                        "stability": stability,
                        "style": style,
                        "speed": speed,
                    },
                    "output_format": "mp3_44100_128",
                },
                timeout=120.0,
            )
            api_response.raise_for_status()
            return api_response.json(), inputs

        return await asyncio.to_thread(_generate)

    response = await utils.async_exponential_backoff(
        generate_dialogue,
        max_attempts=3,
        initial_delay=1,
    )
    if response is None:
        raise ValueError("Failed to generate dialogue after multiple attempts")
    result, _ = response

    # Decode audio from base64
    audio_bytes = base64.b64decode(result["audio_base64"])

    # Save audio to file
    audio_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_file.write(audio_bytes)
    audio_file.close()

    # Build transcript with timestamps for storyboarding
    # Segment-level timestamps from voice_segments
    transcript_segments = []
    voice_segments = result.get("voice_segments", [])
    for voice_segment in voice_segments:
        segment_idx = voice_segment.get("dialogue_input_index", 0)
        segment_data = {
            "index": segment_idx,
            "text": segments[segment_idx]["text"],
            "voice": segments[segment_idx]["voice"],
            "voice_id": voice_segment.get("voice_id"),
            "start": voice_segment.get("start_time_seconds", 0),
            "end": voice_segment.get("end_time_seconds", 0),
        }
        transcript_segments.append(segment_data)

    # Word-level timestamps from alignment (for precise sync)
    # The alignment has: characters (list of chars), character_start_times_seconds, character_end_times_seconds
    words = []
    alignment = result.get("alignment")
    if alignment:
        characters = alignment.get("characters", [])
        start_times = alignment.get("character_start_times_seconds", [])
        end_times = alignment.get("character_end_times_seconds", [])

        current_word = ""
        word_start = None
        last_end = 0

        for i, char in enumerate(characters):
            start = start_times[i] if i < len(start_times) else 0
            end = end_times[i] if i < len(end_times) else 0
            last_end = end

            if char.isspace() or char in ".,!?;:'\"-":
                if current_word:
                    words.append(
                        {
                            "word": current_word,
                            "start": word_start,
                            "end": end,
                        }
                    )
                    current_word = ""
                    word_start = None
            else:
                if word_start is None:
                    word_start = start
                current_word += char

        # Don't forget the last word
        if current_word:
            words.append(
                {
                    "word": current_word,
                    "start": word_start,
                    "end": last_end,
                }
            )

    # Split long segments into ~15 second chunks using word-level timing
    MAX_SEGMENT_DURATION = 15.0

    # Build a map of time -> word for lookup
    def get_words_in_range(start_time, end_time):
        return [w for w in words if w["start"] >= start_time and w["end"] <= end_time]

    # Split segments that are too long
    final_segments = []
    for seg in transcript_segments:
        seg_duration = seg["end"] - seg["start"]
        if seg_duration <= MAX_SEGMENT_DURATION or not words:
            final_segments.append(seg)
        else:
            # Get words for this segment
            seg_words = get_words_in_range(seg["start"], seg["end"])
            if not seg_words:
                final_segments.append(seg)
                continue

            # Group words into sub-segments of ~15 seconds
            sub_segment_words = []
            sub_start = seg["start"]

            for word in seg_words:
                if not sub_segment_words:
                    sub_segment_words = [word]
                    sub_start = word["start"]
                elif word["end"] - sub_start > MAX_SEGMENT_DURATION:
                    # Close current sub-segment
                    final_segments.append(
                        {
                            "voice": seg["voice"],
                            "start": sub_start,
                            "end": sub_segment_words[-1]["end"],
                            "text": " ".join(w["word"] for w in sub_segment_words),
                        }
                    )
                    sub_segment_words = [word]
                    sub_start = word["start"]
                else:
                    sub_segment_words.append(word)

            # Add final sub-segment
            if sub_segment_words:
                final_segments.append(
                    {
                        "voice": seg["voice"],
                        "start": sub_start,
                        "end": sub_segment_words[-1]["end"],
                        "text": " ".join(w["word"] for w in sub_segment_words),
                    }
                )

    # Compact text format for context efficiency
    transcript_lines = []
    for seg in final_segments:
        start = round(seg["start"], 1)
        end = round(seg["end"], 1)
        transcript_lines.append(f"{seg['voice']} {start}-{end} : {seg['text']}")
    transcript = "\n".join(transcript_lines)

    return {
        "output": audio_file.name,
        "transcript": transcript,
    }
