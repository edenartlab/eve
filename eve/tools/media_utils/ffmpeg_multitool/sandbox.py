"""Run a generated ffmpeg command in a throwaway Modal Sandbox.

The command is written by a model from user-supplied instructions and runs on
user-supplied media, so it must not execute in a container that holds service
credentials. The sandbox created here:

- mounts no secrets and no volumes, and gets no environment from the caller;
- has all networking blocked;
- contains only ffmpeg and the job's input files;
- is terminated when the job ends.

The caller downloads the inputs, copies them in, copies the single output
file back out, and does the upload itself. There is deliberately no fallback
to running the command in the calling process.
"""

import os
from typing import Iterable, List, Optional, Tuple

SANDBOX_WORKDIR = "/work"
SANDBOX_APP_NAME = "eve-ffmpeg-sandbox"
STDERR_TAIL_CHARS = 4000
# Time allowed on top of the command attempts for copying files in and out.
SANDBOX_OVERHEAD_SECONDS = 600


def _sandbox_image():
    import modal

    # Same base and ffmpeg package as the API image, and nothing else.
    return modal.Image.debian_slim(python_version="3.11").apt_install(
        "ffmpeg", "fonts-dejavu-core"
    )


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

        kwargs = {}
        if modal.is_local():
            # Outside a Modal container there is no ambient app to attach to.
            kwargs["app"] = await modal.App.lookup.aio(
                SANDBOX_APP_NAME, create_if_missing=True
            )

        self._sb = await modal.Sandbox.create.aio(
            image=_sandbox_image(),
            secrets=[],
            block_network=True,
            timeout=self.lifetime,
            cpu=2,
            **kwargs,
        )
        await self._sb.filesystem.make_directory.aio(SANDBOX_WORKDIR)
        for rel_path in input_paths:
            await self._sb.filesystem.copy_from_local.aio(
                rel_path, f"{SANDBOX_WORKDIR}/{rel_path}"
            )

    async def run(self, argv: List[str]) -> Tuple[int, Optional[str]]:
        """Execute argv (no shell) in the sandbox; return (exit code, stderr tail)."""
        process = await self._sb.exec.aio(
            *argv,
            workdir=SANDBOX_WORKDIR,
            timeout=self.command_timeout,
        )
        stderr = await process.stderr.read.aio()
        returncode = await process.wait.aio()
        return returncode, (stderr[-STDERR_TAIL_CHARS:] if stderr else None)

    async def fetch(self, rel_path: str, local_path: str) -> bool:
        """Copy one file out of the sandbox. Returns False if it is not there."""
        try:
            await self._sb.filesystem.copy_to_local.aio(
                f"{SANDBOX_WORKDIR}/{rel_path}", local_path
            )
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
