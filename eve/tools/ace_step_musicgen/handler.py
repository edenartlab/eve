import tempfile
import asyncio

async def handler(args: dict, user: str = None, agent: str = None):
    """
    Handler for the ACE-Step Musicgen tool.
    Takes parameters, runs the ACEStepPipeline, and returns the S3 URL of the generated audio.
    """

    # ACEStepPipeline constructor arguments from args
    # Default checkpoint_path assumes a location within the Modal container
    checkpoint_path = "/app/checkpoints/acestep_model"
    bf16 = args.get("bf16", True)

    # ACEStepPipeline call arguments from args
    audio_duration = args.get("audio_duration")
    prompt = args.get("prompt")
    lyrics = args.get("lyrics", "") # Default to empty string if not provided
    infer_step = args.get("infer_step", 27)
    guidance_scale = args.get("guidance_scale", 7.0)
    scheduler_type = args.get("scheduler_type", "dpmpp_2m_sde")
    cfg_type = args.get("cfg_type", "text")
    omega_scale = args.get("omega_scale", 1.0)
    manual_seeds_str = args.get("manual_seeds", None)
    guidance_scale_text = args.get("guidance_scale_text", 0.0)
    guidance_scale_lyric = args.get("guidance_scale_lyric", 0.0)

    # Validate required arguments
    if not audio_duration or not prompt:
        raise ValueError("Missing required arguments: audio_duration and prompt are required.")

    # Temporary file for the output audio
    # ACEStepPipeline expects a save_path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_temp_path = tmpfile.name

    def run_pipeline_sync():
        # Import ACEStepPipeline here to ensure it's attempted within the Modal environment
        from acestep.pipeline_ace_step import ACEStepPipeline

        model = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )
        
        model(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds_str,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=output_temp_path
        )

    try:
        await asyncio.to_thread(run_pipeline_sync)
        return {
            "output": output_temp_path
        }

    except Exception as e:
        print(f"Error in ACE-Step Musicgen handler: {e}")
        raise
    finally:
        pass 