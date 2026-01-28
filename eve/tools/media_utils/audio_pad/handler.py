import asyncio
import subprocess
import tempfile

from eve.tool import ToolContext


async def handler(context: ToolContext):
    from .... import utils

    audio_url = context.args.get("audio_url")
    pad_start = context.args.get("pad_start", 0) or 0
    pad_end = context.args.get("pad_end", 0) or 0

    if pad_start == 0 and pad_end == 0:
        raise ValueError("At least one of pad_start or pad_end required")

    audio_file = utils.get_file_handler(".mp3", audio_url)
    output = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    filters = []
    if pad_start > 0:
        # adelay takes milliseconds, delays all channels
        filters.append(f"adelay={int(pad_start * 1000)}|{int(pad_start * 1000)}")
    if pad_end > 0:
        # apad adds silence at the end
        filters.append(f"apad=pad_dur={pad_end}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-i",
        audio_file,
        "-af",
        ",".join(filters),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        output.name,
    ]

    await asyncio.to_thread(subprocess.run, cmd)

    return {"output": output.name}
