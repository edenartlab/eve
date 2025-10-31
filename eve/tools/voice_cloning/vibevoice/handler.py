import tempfile
import modal
from eve.tool import ToolContext
from loguru import logger


async def handler(context: ToolContext):
    """
    Handler for VibeVoice voice cloning and TTS.

    Calls the Modal app to generate audio remotely.
    Supports single-speaker and multi-speaker generation with optional voice cloning.
    """
    args = context.args

    # Required parameters
    text = args["text"]

    # Optional parameters with defaults
    cfg_scale = args.get("cfg_scale", 1.33)
    diffusion_steps = args.get("diffusion_steps", 25)
    seed = args.get("seed", 42)
    voice_audio = args.get("voice_audio")  # List of audio URLs or None

    logger.info(f"Calling VibeVoice Modal app...")
    logger.info(f"Text: {text[:100]}...")
    logger.info(f"CFG Scale: {cfg_scale}, Diffusion Steps: {diffusion_steps}, Seed: {seed}")
    if voice_audio:
        logger.info(f"Voice audio files: {len(voice_audio)}")

    try:
        # Get the Modal function
        VibeVoiceContainer = modal.Cls.from_name("VibeVoice-audio-app", "VibeVoiceContainer")

        # Call the generate method remotely
        audio_bytes = VibeVoiceContainer().generate.remote(
            text=text,
            voice_audio=voice_audio,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
            seed=seed,
        )

        # Save the audio bytes to a local temporary file
        output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_file.write(audio_bytes)
        output_file.close()

        logger.info(f"Audio generated successfully: {output_file.name}")

        return {
            "output": output_file.name,
        }

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise
