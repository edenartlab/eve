import json
import subprocess
import tempfile

from eve.tool import ToolContext


def get_audio_loudness(audio_file: str) -> float:
    """
    Get the integrated loudness (LUFS) of an audio file using ffmpeg's loudnorm filter.
    This provides a perceptually accurate measure of audio volume.
    """
    cmd = [
        "ffmpeg",
        "-i",
        audio_file,
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the loudnorm output from stderr
    stderr = result.stderr

    # Find the JSON block in the output
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        try:
            loudness_data = json.loads(stderr[json_start:json_end])
            return float(loudness_data.get("input_i", -23.0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: return a default LUFS value
    return -23.0


async def handler(context: ToolContext):
    from .... import utils

    audio_tracks = context.args.get("audio_tracks", [])

    if not audio_tracks:
        raise ValueError("At least one audio track is required")

    # Download all audio files and get their properties
    track_data = []
    max_duration = 0.0

    for track in audio_tracks:
        audio_url = track.get("audio_url")
        volume = track.get("volume", 1.0)

        # Clamp volume to valid range
        volume = max(0.0, min(1.0, float(volume)))

        audio_file = utils.get_file_handler(".mp3", audio_url)
        duration = utils.get_media_duration(audio_file)
        loudness = get_audio_loudness(audio_file)

        track_data.append(
            {
                "file": audio_file,
                "volume": volume,
                "duration": duration,
                "loudness": loudness,
            }
        )

        max_duration = max(max_duration, duration)

    # Calculate target loudness (use the loudest track as reference after volume weighting)
    # We normalize all tracks to a common loudness level, then apply volume ratios
    target_loudness = -16.0  # Standard broadcast loudness

    output_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    # Build ffmpeg command with filter complex
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
    ]

    # Add all audio inputs
    for track in track_data:
        cmd.extend(["-i", track["file"]])

    # Build filter complex
    filter_parts = []

    for i, track in enumerate(track_data):
        # Calculate the loudness adjustment needed to normalize to target
        loudness_adjustment = target_loudness - track["loudness"]

        # Apply volume ratio on top of normalization
        # Convert volume ratio (0-1) to dB adjustment
        # volume=1.0 means no change, volume=0.5 means -6dB, etc.
        if track["volume"] > 0:
            volume_db = 20 * (track["volume"] ** 0.5 - 1)  # sqrt for perceptual scaling
        else:
            volume_db = -96  # effectively silent

        total_adjustment = loudness_adjustment + volume_db

        # Pad shorter tracks to match max duration
        # Apply volume adjustment
        filter_parts.append(
            f"[{i}:a]volume={total_adjustment}dB,apad=whole_dur={max_duration}[a{i}]"
        )

    # Mix all tracks together with normalization to prevent clipping
    num_tracks = len(track_data)
    mix_inputs = "".join(f"[a{i}]" for i in range(num_tracks))

    # Use amix with normalize=1 to prevent clipping
    # dropout_transition=0 keeps volume consistent when tracks end
    filter_parts.append(
        f"{mix_inputs}amix=inputs={num_tracks}:duration=longest:dropout_transition=0:normalize=1[mixed]"
    )

    # Apply final loudnorm to ensure output doesn't clip and has consistent loudness
    filter_parts.append("[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]")

    filter_complex = ";".join(filter_parts)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            output_file.name,
        ]
    )

    subprocess.run(cmd)

    return {"output": output_file.name}
