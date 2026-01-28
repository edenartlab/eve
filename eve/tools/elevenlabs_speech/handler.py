import asyncio
import base64
import os
from tempfile import NamedTemporaryFile
from typing import Dict, List, Literal

import httpx
import instructor
from elevenlabs.client import ElevenLabs
from openai import OpenAI

from eve import utils
from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

DEFAULT_VOICE = "Charlotte"
MAX_SEGMENT_DURATION = 15.0  # Target segment length in seconds for transcripts


def search_voice_by_description(
    description: str,
    voices: List,
) -> str:
    """Use LLM to find the best matching voice based on description."""
    client = instructor.from_openai(OpenAI())
    voice_names = [v.name for v in voices if v.name]

    def format_voice(v):
        parts = [v.name]
        if v.labels:
            label_str = ", ".join(f"{k}={val}" for k, val in v.labels.items() if val)
            if label_str:
                parts.append(f"[{label_str}]")
        if v.description:
            parts.append(f'"{v.description}"')
        return " ".join(parts)

    voice_descriptions = "\n".join([format_voice(v) for v in voices if v.name])

    prompt = f"""You are given the following list of voices and their descriptions.

---
{voice_descriptions}
---

You are given the following description of a desired character:

---
{description}
---

Select the voice that best matches the description of the character."""

    selected_voice = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Literal[tuple(voice_names)],  # type: ignore
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at selecting the right voice for a character.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return selected_voice


def resolve_voice_id(
    voice: str,
    voice_name_map: Dict[str, str],
    voice_ids: List[str],
    voices: List = None,
) -> str:
    """
    Resolve a voice identifier to an ElevenLabs voice ID.

    Args:
        voice: Can be a voice ID, voice name, description, or @agent_username
        voice_name_map: Map of voice names to voice IDs
        voice_ids: List of valid voice IDs
        voices: List of voice objects for LLM search fallback

    Returns:
        Resolved ElevenLabs voice ID
    """
    # Check if it's an agent username (starts with @)
    if voice.startswith("@"):
        username = voice[1:]  # Remove the @ prefix
        from eve.agent.agent import Agent

        agent = Agent.find_one({"username": username})
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

    # Case-insensitive matching
    voice_lower = voice.lower()
    for name, vid in voice_name_map.items():
        if name.lower() == voice_lower:
            return vid
        # Match prefix (handles "George" matching "George - Warm Storyteller")
        if name.lower().startswith(voice_lower):
            return vid
        # Match the part before " - "
        if " - " in name and name.split(" - ")[0].lower() == voice_lower:
            return vid

    # Fallback: use LLM to search for best matching voice
    if voices:
        matched_name = search_voice_by_description(voice, voices)
        if matched_name in voice_name_map:
            return voice_name_map[matched_name]

    raise ValueError(
        f"Voice '{voice}' not found. Use elevenlabs_search_voices to find valid voice names."
    )


def get_voice_name(voice_id: str, voice_name_map: Dict[str, str]) -> str:
    """Get the short voice name for a voice ID."""
    for name, vid in voice_name_map.items():
        if vid == voice_id:
            return name.split(" - ")[0]  # Get first part before " - "
    return "Narrator"


def build_words_from_alignment(alignment: dict) -> List[dict]:
    """Parse character-level alignment into word-level timing."""
    if not alignment:
        return []

    characters = alignment.get("characters", [])
    start_times = alignment.get("character_start_times_seconds", [])
    end_times = alignment.get("character_end_times_seconds", [])

    words = []
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
            # Include punctuation as part of the previous word
            if char in ".,!?;:" and words:
                words[-1]["word"] += char
        else:
            if word_start is None:
                word_start = start
            current_word += char

    # Don't forget the last word
    if current_word and word_start is not None:
        words.append(
            {
                "word": current_word,
                "start": word_start,
                "end": last_end,
            }
        )

    return words


def segment_by_duration(
    words: List[dict],
    voice_name: str,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> List[dict]:
    """Group words into segments of approximately max_duration seconds."""
    if not words:
        return []

    segments = []
    segment_words = []
    segment_start = None

    for word in words:
        if segment_start is None:
            segment_start = word["start"]
            segment_words = [word]
        elif word["end"] - segment_start > max_duration:
            # Close current segment and start new one
            segments.append(
                {
                    "voice": voice_name,
                    "start": segment_start,
                    "end": segment_words[-1]["end"],
                    "text": " ".join(w["word"] for w in segment_words),
                }
            )
            segment_start = word["start"]
            segment_words = [word]
        else:
            segment_words.append(word)

    # Add final segment
    if segment_words:
        segments.append(
            {
                "voice": voice_name,
                "start": segment_start,
                "end": segment_words[-1]["end"],
                "text": " ".join(w["word"] for w in segment_words),
            }
        )

    return segments


def format_transcript(segments: List[dict]) -> str:
    """Format segments into transcript lines."""
    lines = []
    for seg in segments:
        start = round(seg["start"], 1)
        end = round(seg["end"], 1)
        lines.append(f"{seg['voice']} {start}-{end} : {seg['text']}")
    return "\n".join(lines)


async def handler(context: ToolContext):
    args = context.args
    stability = args.get("stability", 0.5)
    style = args.get("style", 0.0)
    speed = args.get("speed", 1.0)

    segments_input = args.get("segments")
    if not segments_input:
        raise ValueError("Must provide 'segments' array")

    is_dialogue = len(segments_input) > 1

    async def generate_speech():
        def _generate():
            if not ELEVEN_API_KEY:
                raise ValueError("ELEVEN_API_KEY environment variable is not set")

            # Get all available voices for resolution
            response = eleven.voices.get_all()
            voices = list(response.voices)
            voice_name_map = {v.name: v.voice_id for v in voices if v.name is not None}
            voice_ids = [v.voice_id for v in voices]

            if is_dialogue:
                # Multi-speaker: use dialogue endpoint
                inputs = []
                for segment in segments_input:
                    voice_id = resolve_voice_id(
                        segment["voice"],
                        voice_name_map,
                        voice_ids,
                        voices,
                    )
                    inputs.append(
                        {
                            "text": segment["text"],
                            "voice_id": voice_id,
                        }
                    )

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
                return api_response.json(), segments_input, voice_name_map
            else:
                # Single speaker: use TTS endpoint
                segment = segments_input[0]
                voice_id = resolve_voice_id(
                    segment["voice"],
                    voice_name_map,
                    voice_ids,
                    voices,
                )
                voice_name = get_voice_name(voice_id, voice_name_map)

                api_response = httpx.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                    headers={
                        "xi-api-key": ELEVEN_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": segment["text"],
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": stability,
                            "style": style,
                            "speed": speed,
                            "use_speaker_boost": True,
                        },
                        "output_format": "mp3_44100_128",
                    },
                    timeout=120.0,
                )
                api_response.raise_for_status()
                return api_response.json(), voice_name, voice_name_map

        return await asyncio.to_thread(_generate)

    response = await utils.async_exponential_backoff(
        generate_speech,
        max_attempts=3,
        initial_delay=1,
    )

    if response is None:
        raise ValueError("Failed to generate speech after multiple attempts")

    result, extra_data, _ = response

    # Decode audio from base64
    audio_bytes = base64.b64decode(result["audio_base64"])

    # Save audio to file
    audio_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_file.write(audio_bytes)
    audio_file.close()

    # Build word-level timing from alignment
    alignment = result.get("alignment", {})
    words = build_words_from_alignment(alignment)

    if is_dialogue:
        # Multi-speaker: build segments from voice_segments, then split long ones
        segments_input_data = extra_data
        voice_segments = result.get("voice_segments", [])

        transcript_segments = []
        for voice_segment in voice_segments:
            segment_idx = voice_segment.get("dialogue_input_index", 0)
            transcript_segments.append(
                {
                    "voice": segments_input_data[segment_idx]["voice"],
                    "start": voice_segment.get("start_time_seconds", 0),
                    "end": voice_segment.get("end_time_seconds", 0),
                    "text": segments_input_data[segment_idx]["text"],
                }
            )

        # Split long segments using word timing
        def get_words_in_range(start_time, end_time):
            return [
                w for w in words if w["start"] >= start_time and w["end"] <= end_time
            ]

        final_segments = []
        for seg in transcript_segments:
            seg_duration = seg["end"] - seg["start"]
            if seg_duration <= MAX_SEGMENT_DURATION or not words:
                final_segments.append(seg)
            else:
                seg_words = get_words_in_range(seg["start"], seg["end"])
                if not seg_words:
                    final_segments.append(seg)
                    continue

                sub_segments = segment_by_duration(seg_words, seg["voice"])
                final_segments.extend(sub_segments)

        transcript = format_transcript(final_segments)
    else:
        # Single speaker: segment by duration
        voice_name = extra_data
        if words:
            final_segments = segment_by_duration(words, voice_name)
            transcript = format_transcript(final_segments)
        else:
            # Fallback
            word_count = len(segments_input[0]["text"].split())
            duration = (word_count / 150) * 60
            transcript = (
                f"{voice_name} 0.0-{round(duration, 1)} : {segments_input[0]['text']}"
            )

    return {
        "output": audio_file.name,
        "transcript": transcript,
    }
