"""Run a generated ffmpeg command in a throwaway Modal Sandbox.

The command is written by a model from user-supplied instructions and runs on
user-supplied media, so it must not execute in a container that holds service
credentials. The sandbox created here:

- mounts no secrets and no volumes, and gets no environment from the caller;
- has all networking blocked;
- contains only ffmpeg, fonts and the job's input files;
- is terminated when the job ends.

The caller downloads the inputs, copies them in, copies the single output
file back out, and does the upload itself. There is deliberately no fallback
to running the command in the calling process.
"""

import asyncio
import os
from typing import Iterable, List, Optional, Tuple

SANDBOX_WORKDIR = "/work"
SANDBOX_APP_NAME = "eve-ffmpeg-sandbox"
STDERR_TAIL_CHARS = 4000
# Time allowed on top of the command attempts for copying files in and out.
SANDBOX_OVERHEAD_SECONDS = 600
# Exit status reported for a command stopped at its time limit (as `timeout`).
TIMEOUT_EXIT_CODE = 124
# The caller reads the output into memory to upload it.
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
COPY_CONCURRENCY = 8

# Font families installed in the sandbox image, for the command generator.
SANDBOX_FONT_FAMILIES = ("DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono")
SANDBOX_FONT_DIR = "/usr/share/fonts/truetype/dejavu"


class OutputTooLargeError(Exception):
    """The command's output file is larger than the caller will upload."""


def _sandbox_image():
    import modal

    # A named Debian release rather than `debian_slim()`: that base follows
    # the workspace's image-builder version, and an end-of-life release there
    # can no longer install packages. Nothing but ffmpeg and fonts goes in.
    return modal.Image.from_registry(
        "debian:trixie-slim", add_python="3.11"
    ).apt_install("ffmpeg", "fontconfig", "fonts-dejavu-core", "fonts-liberation")


async def _sandbox_app():
    import modal

    # A dedicated app, so sandboxes are not part of the API app. Works the
    # same from a developer machine and from inside a Modal container.
    try:
        return await modal.App.lookup.aio(SANDBOX_APP_NAME, create_if_missing=True)
    except Exception:
        if modal.is_local():
            raise
        # Inside a container Modal can attach the sandbox to the running app.
        return None


class FFmpegSandbox:
    """One isolated sandbox per tool call, reused across retries."""

    def __init__(self, command_timeout: int, attempts: int = 1):
        self.command_timeout = command_timeout
        # The sandbox must outlive every retry of the command.
        self.lifetime = command_timeout * max(1, attempts) + SANDBOX_OVERHEAD_SECONDS
        self._sb = None

    async def start(self, input_paths: Iterable[str]) -> None:
        """Create the sandbox and copy the job's input files into it.

        `input_paths` are paths relative to the caller's working directory;
        they are mirrored under SANDBOX_WORKDIR so the generated command can
        use the same relative paths.
        """
        import modal

        self._sb = await modal.Sandbox.create.aio(
            app=await _sandbox_app(),
            image=_sandbox_image(),
            secrets=[],
            block_network=True,
            timeout=self.lifetime,
            cpu=4,
        )
        await self._sb.filesystem.make_directory.aio(SANDBOX_WORKDIR)

        limit = asyncio.Semaphore(COPY_CONCURRENCY)

        async def copy_in(rel_path: str) -> None:
            async with limit:
                await self._sb.filesystem.copy_from_local.aio(
                    rel_path, f"{SANDBOX_WORKDIR}/{rel_path}"
                )

        await asyncio.gather(*(copy_in(rel_path) for rel_path in input_paths))

    async def run(self, argv: List[str]) -> Tuple[int, Optional[str]]:
        """Execute argv (no shell) in the sandbox; return (exit code, stderr tail)."""
        import modal
        from modal.stream_type import StreamType

        # Standard output is discarded: the result is a file, and an unread
        # pipe would stall a command that writes to it.
        process = await self._sb.exec.aio(
            *argv,
            workdir=SANDBOX_WORKDIR,
            timeout=self.command_timeout,
            stdout=StreamType.DEVNULL,
        )
        try:
            stderr = await process.stderr.read.aio()
            returncode = await process.wait.aio()
        except modal.exception.ExecTimeoutError:
            return TIMEOUT_EXIT_CODE, None
        if returncode == -1:
            # How Modal reports a process it stopped at the time limit.
            returncode = TIMEOUT_EXIT_CODE
        return returncode, (stderr[-STDERR_TAIL_CHARS:] if stderr else None)

    async def fetch(self, rel_path: str, local_path: str) -> bool:
        """Copy one file out of the sandbox. Returns False if it is not there."""
        remote_path = f"{SANDBOX_WORKDIR}/{rel_path}"
        try:
            info = await self._sb.filesystem.stat.aio(remote_path)
        except Exception:
            return False
        if not info.is_file() or info.size == 0:
            return False
        if info.size > MAX_OUTPUT_BYTES:
            raise OutputTooLargeError(
                f"Output file is {info.size} bytes; the limit is {MAX_OUTPUT_BYTES}"
            )
        try:
            await self._sb.filesystem.copy_to_local.aio(remote_path, local_path)
        except Exception:
            if os.path.exists(local_path):
                os.remove(local_path)
            return False
        return os.path.exists(local_path)

    async def close(self) -> None:
        if self._sb is not None:
            try:
                await self._sb.terminate.aio()
            except Exception:
                pass
            self._sb = None
