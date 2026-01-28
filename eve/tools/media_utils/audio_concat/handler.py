import asyncio
import subprocess
import tempfile

from eve.tool import ToolContext


async def handler(context: ToolContext):
    from .... import utils

    audio_files = context.args.get("audio_files", [])

    if len(audio_files) < 2:
        raise ValueError("At least 2 audio files required")

    # Download all files
    local_files = [utils.get_file_handler(".mp3", url) for url in audio_files]

    # Create concat file list
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for f in local_files:
        list_file.write(f"file '{f}'\n")
    list_file.close()

    output = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file.name,
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        output.name,
    ]

    await asyncio.to_thread(subprocess.run, cmd)

    return {"output": output.name}
