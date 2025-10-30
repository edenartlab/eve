"""
Simple Modal deployment with GPU and ComfyUI dependencies.
No code or files included - just the base environment.

Usage:
    modal deploy deploy_gpu_modal_image.py
"""

import modal
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).parent

# Create a persistent volume
downloads_vol = modal.Volume.from_name("vibevoice-volume", create_if_missing=True)

# Build the image with the same dependencies as comfyui.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "git-lfs"
    )
    .pip_install(
        "librosa",
        "torch",
        "torchvision",
        "torchaudio",
        "soundfile",
        "loguru",
        "requests"
    )
    .run_commands(
        # Clone VibeVoice-ComfyUI repository at specific commit
        "cd /root && git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI.git",
        "cd /root/VibeVoice-ComfyUI && git checkout 9185f531ac45fc67576f9877caf1a6c8c7d340b5",
        # Install repository requirements
        "cd /root/VibeVoice-ComfyUI && pip install -r requirements.txt"
    )
    .add_local_file(
        local_path=str(root_dir / "run_inference.py"),
        remote_path="/root/VibeVoice-ComfyUI/run_inference.py"
    )
)

app = modal.App(name="VibeVoice-audio-app")

@app.cls(
    image=image,
    gpu="A10",
    cpu=2.0,
    volumes={"/data": downloads_vol},
    max_containers=1,
    scaledown_window=60,
    min_containers=0,
    timeout=1800,
)
class VibeVoiceContainer:
    """
    Modal container for VibeVoice voice cloning and TTS generation.
    """

    @modal.enter()
    def enter(self):
        import subprocess
        import os

        print("Container started with GPU and dependencies")

        # Download models to persistent volume if not already present
        # Models should be in /data/models and will be symlinked to the repo structure
        persistent_models_dir = "/data/models"
        os.makedirs(persistent_models_dir, exist_ok=True)

        # Create the repo models directory structure that main.py expects
        repo_models_dir = "/root/VibeVoice-ComfyUI/models"
        os.makedirs(repo_models_dir, exist_ok=True)

        # Download VibeVoice-Large-Q8 model to persistent storage
        vibevoice_persistent_path = os.path.join(persistent_models_dir, "VibeVoice-Large-Q8")
        if not os.path.exists(vibevoice_persistent_path):
            print("Downloading VibeVoice-Large-Q8 model...")
            subprocess.run(
                ["git", "clone", "https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8", vibevoice_persistent_path],
                check=True
            )
            print("VibeVoice-Large-Q8 model downloaded successfully")
        else:
            print("VibeVoice-Large-Q8 model already exists")

        # Symlink to repo models directory
        vibevoice_repo_path = os.path.join(repo_models_dir, "VibeVoice-Large-Q8")
        if not os.path.exists(vibevoice_repo_path):
            os.symlink(vibevoice_persistent_path, vibevoice_repo_path)
            print(f"Created symlink: {vibevoice_repo_path} -> {vibevoice_persistent_path}")

        # Download Qwen2.5-1.5B model to persistent storage
        qwen_persistent_path = os.path.join(persistent_models_dir, "Qwen2.5-1.5B")
        if not os.path.exists(qwen_persistent_path):
            print("Downloading Qwen2.5-1.5B model...")
            subprocess.run(
                ["git", "clone", "https://huggingface.co/Qwen/Qwen2.5-1.5B", qwen_persistent_path],
                check=True
            )
            print("Qwen2.5-1.5B model downloaded successfully")
        else:
            print("Qwen2.5-1.5B model already exists")

        # Create tokenizer directory and symlink
        tokenizer_dir = os.path.join(repo_models_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        qwen_repo_path = os.path.join(tokenizer_dir, "Qwen2.5-1.5B")
        if not os.path.exists(qwen_repo_path):
            os.symlink(qwen_persistent_path, qwen_repo_path)
            print(f"Created symlink: {qwen_repo_path} -> {qwen_persistent_path}")

        print("All models ready")
        print(f"Model paths:")
        print(f"  VibeVoice: {vibevoice_repo_path}")
        print(f"  Tokenizer: {qwen_repo_path}")

    @modal.method()
    def hello(self):
        """Simple test method to verify the container works"""
        return {"status": "success", "message": "GPU container is running"}

    @modal.method()
    def generate(
        self,
        text: str,
        voice_audio: list[str] = None,
        cfg_scale: float = 1.33,
        diffusion_steps: int = 30,
        seed: int = 42,
    ):
        """
        Generate audio using VibeVoice.

        Args:
            text: Text to convert to speech
            voice_audio: Optional list of audio file URLs for voice cloning
            cfg_scale: CFG scale (default: 1.33)
            diffusion_steps: Diffusion steps (default: 30)
            seed: Random seed (default: 42)

        Returns:
            Path to generated audio file in /data volume
        """
        import sys
        import os
        import tempfile
        import soundfile as sf
        from loguru import logger

        # Add VibeVoice to path
        sys.path.insert(0, "/root/VibeVoice-ComfyUI")
        from run_inference import generate_audio

        # Model paths
        model_path = "/root/VibeVoice-ComfyUI/models/VibeVoice-Large-Q8"
        tokenizer_path = "/root/VibeVoice-ComfyUI/models/tokenizer/Qwen2.5-1.5B"

        # Handle voice_audio - download if URLs provided
        voice_audio_paths = None
        if voice_audio:
            import requests
            voice_audio_paths = []
            for i, audio_url in enumerate(voice_audio):
                logger.info(f"Downloading voice audio {i+1}: {audio_url}")
                response = requests.get(audio_url)
                response.raise_for_status()

                temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                temp_file.write(response.content)
                temp_file.close()
                voice_audio_paths.append(temp_file.name)
                logger.info(f"Saved to: {temp_file.name}")

            # If single audio, pass as string
            if len(voice_audio_paths) == 1:
                voice_audio_paths = voice_audio_paths[0]
                logger.info("Single voice audio - using single-speaker mode")
            else:
                logger.info(f"Multiple voice audio - using multi-speaker mode with {len(voice_audio_paths)} speakers")

        logger.info(f"Generating audio with VibeVoice")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"CFG Scale: {cfg_scale}, Diffusion Steps: {diffusion_steps}, Seed: {seed}")

        # Generate audio
        audio_data = generate_audio(
            text=text,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            voice_audio=voice_audio_paths,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
            seed=seed,
            attention_type="auto",
            quantize_llm="full precision",
            use_sampling=False,
            temperature=0.95,
            top_p=0.95,
            free_memory_after=False,
        )

        # Save to temporary file and read bytes
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_output.name, audio_data["waveform"], audio_data["sample_rate"])
        logger.info(f"Audio generated successfully")

        # Read the audio file as bytes
        with open(temp_output.name, "rb") as f:
            audio_bytes = f.read()

        # Clean up temp file
        os.remove(temp_output.name)

        # Clean up downloaded voice audio files
        if voice_audio_paths:
            if isinstance(voice_audio_paths, list):
                for audio_path in voice_audio_paths:
                    try:
                        os.remove(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up {audio_path}: {e}")
            elif isinstance(voice_audio_paths, str):
                try:
                    os.remove(voice_audio_paths)
                except Exception as e:
                    logger.warning(f"Failed to clean up {voice_audio_paths}: {e}")

        return audio_bytes


@app.local_entrypoint()
def main():
    """Test the deployment"""
    container = VibeVoiceContainer()
    result = container.hello.remote()
    print(f"Result: {result}")

    # Test VibeVoice with a simple example
    print("\nTesting VibeVoice...")
    audio_bytes = container.generate.remote(
        text="Hello, this is a test of VibeVoice text to speech.",
        cfg_scale=1.33,
        diffusion_steps=30,
        seed=42
    )
    print(f"VibeVoice Result: Received {len(audio_bytes)} bytes of audio data")

    # Save to local file for testing
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print("Saved test audio to test_output.wav")
