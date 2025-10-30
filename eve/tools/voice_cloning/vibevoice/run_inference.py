"""
Standalone VibeVoice Inference Script
======================================

This script provides a standalone Python interface to the VibeVoice model
without requiring ComfyUI. It wraps all the functionality from the ComfyUI
node into a simple generate_audio() function with full multi-speaker support.

Example usage:
    from run_inference import generate_audio
    import soundfile as sf

    # === Single Speaker Examples ===

    # Basic single speaker with synthetic voice
    audio_data = generate_audio(
        text="Hello, this is a test of VibeVoice.",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer"
    )
    sf.write("output.wav", audio_data["waveform"], audio_data["sample_rate"])

    # Single speaker with voice cloning from file
    audio_data = generate_audio(
        text="I will speak in the reference voice.",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer",
        voice_audio="/path/to/reference.wav",  # Any sample rate supported
        cfg_scale=1.5,
        diffusion_steps=30
    )

    # Single speaker with voice cloning from numpy array
    import numpy as np
    reference_audio = np.load("reference_voice.npy")  # Your reference audio at 24000 Hz
    audio_data = generate_audio(
        text="I will speak in the reference voice.",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer",
        voice_audio=reference_audio
    )

    # === Multi-Speaker Examples ===

    # Multi-speaker with synthetic voices (automatic)
    audio_data = generate_audio(
        text="Speaker 1: Hello there! Speaker 2: Hi, how are you doing?",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer"
    )
    sf.write("conversation.wav", audio_data["waveform"], audio_data["sample_rate"])

    # Multi-speaker with separate voice cloning for each speaker
    audio_data = generate_audio(
        text="Speaker 1: Welcome to the show. Speaker 2: Thanks for having me!",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer",
        voice_audio=["speaker1_reference.wav", "speaker2_reference.wav"]
    )

    # Multi-speaker with mixed numpy arrays and files
    audio_data = generate_audio(
        text="Speaker 1: First person. Speaker 2: Second person.",
        model_path="/path/to/models/vibevoice/VibeVoice-1.5B",
        tokenizer_path="/path/to/tokenizer",
        voice_audio=[np.load("voice1.npy"), "voice2.wav"]
    )
"""

import logging
import os
import sys
import re
import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[VibeVoice] %(message)s'
)
logger = logging.getLogger("VibeVoice")

# Add vvembed to path
current_dir = os.path.dirname(os.path.abspath(__file__))
vvembed_path = os.path.join(current_dir, 'vvembed')
if vvembed_path not in sys.path:
    sys.path.insert(0, vvembed_path)


def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_map():
    """Get device map for model loading"""
    device = get_optimal_device()
    return device if device != "mps" else "mps"


class VibeVoiceInference:
    """Standalone VibeVoice inference class"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None
        self.current_attention_type = None
        self.current_quantize_llm = "full precision"

    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            self.current_model_path = None
            self.current_quantize_llm = "full precision"

            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Model and processor memory freed successfully")

        except Exception as e:
            logger.error(f"Error freeing memory: {e}")

    def load_model(self, model_path: str, tokenizer_path: str, attention_type: str = "auto",
                   quantize_llm: str = "full precision"):
        """Load VibeVoice model

        Args:
            model_path: Path to the model directory containing config.json and model files
            attention_type: Attention implementation ("auto", "eager", "sdpa", "flash_attention_2", "sage")
            quantize_llm: LLM quantization mode ("full precision", "8bit", or "4bit")
        """
        # Check if we need to reload
        if (self.model is None or
            self.current_model_path != model_path or
            self.current_attention_type != attention_type or
            self.current_quantize_llm != quantize_llm):

            if self.model is not None:
                logger.info(f"Reloading model with new settings...")
                self.free_memory()

            try:
                # Import the inference model
                from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                from processor.vibevoice_processor import VibeVoiceProcessor

                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)

                # Verify model path exists
                if not os.path.exists(model_path):
                    raise Exception(f"Model path does not exist: {model_path}")

                if not os.path.exists(os.path.join(model_path, "config.json")):
                    raise Exception(f"config.json not found in {model_path}")

                logger.info(f"Loading model from: {model_path}")

                # Prepare model kwargs
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": get_device_map(),
                    "local_files_only": True,
                }

                # Handle quantization
                if quantize_llm != "full precision":
                    if not torch.cuda.is_available():
                        raise Exception("Quantization requires CUDA GPU. Please use 'full precision' on CPU/MPS.")

                    from transformers import BitsAndBytesConfig

                    if quantize_llm == "4bit":
                        logger.info("Quantizing LLM component to 4-bit...")
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                            llm_int8_skip_modules=["lm_head", "prediction_head", "acoustic_connector",
                                                   "semantic_connector", "diffusion_head"]
                        )
                    else:  # 8bit
                        logger.info("Quantizing LLM component to 8-bit...")
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                            llm_int8_skip_modules=["lm_head", "prediction_head", "acoustic_connector",
                                                   "semantic_connector", "acoustic_tokenizer", "semantic_tokenizer"],
                            llm_int8_threshold=3.0,
                            llm_int8_has_fp16_weight=False,
                        )

                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"

                # Set attention type
                if attention_type != "auto" and attention_type != "sage":
                    model_kwargs["attn_implementation"] = attention_type

                # Load model
                import time
                start_time = time.time()
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")

                # Load processor
                logger.info("Loading VibeVoice processor...")
                processor_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True
                }

                # Try to find Qwen tokenizer
                tokenizer_path = self._find_qwen_tokenizer(tokenizer_path)
                if tokenizer_path:
                    logger.info(f"Using Qwen tokenizer from: {tokenizer_path}")
                    processor_kwargs["language_model_pretrained_name"] = tokenizer_path

                self.processor = VibeVoiceProcessor.from_pretrained(
                    model_path,
                    **processor_kwargs
                )

                # Move to device if not quantized
                if quantize_llm == "full precision":
                    device = get_optimal_device()
                    if device == "cuda":
                        try:
                            self.model = self.model.cuda()
                        except:
                            pass
                    elif device == "mps":
                        self.model = self.model.to("mps")

                self.current_model_path = model_path
                self.current_attention_type = attention_type
                self.current_quantize_llm = quantize_llm

                logger.info("Model and processor loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def _find_qwen_tokenizer(self, model_path: str) -> Optional[str]:
        """Try to find Qwen tokenizer in various locations"""
        # Check in model directory
        tokenizer_dir = model_path
        if os.path.exists(tokenizer_dir):
            required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
            if all(os.path.exists(os.path.join(tokenizer_dir, f)) for f in required_files):
                return tokenizer_dir

        print(f"Couldn't find needed files for tokenizer in {tokenizer_dir}, searching in HF hub dir...")

        # Check HuggingFace cache
        hf_cache_paths = [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
        ]

        for cache_path in hf_cache_paths:
            if cache_path and os.path.exists(cache_path):
                qwen_cache = os.path.join(cache_path, "models--Qwen--Qwen2.5-1.5B")
                if os.path.exists(qwen_cache):
                    snapshots_dir = os.path.join(qwen_cache, "snapshots")
                    if os.path.exists(snapshots_dir):
                        for snapshot in os.listdir(snapshots_dir):
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            if os.path.isdir(snapshot_path):
                                if os.path.exists(os.path.join(snapshot_path, "tokenizer_config.json")):
                                    return snapshot_path

        print("WARNING: Qwen tokenizer not found. Using synthetic voice.")
        return None

    def _create_synthetic_voice_sample(self, speaker_idx: int = 0) -> np.ndarray:
        """Create synthetic voice sample"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)

        t = np.linspace(0, duration, samples, False)

        base_frequencies = [120, 180, 140, 200]
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]

        formant1 = 800 + speaker_idx * 100
        formant2 = 1200 + speaker_idx * 150

        voice_sample = (
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            0.02 * np.random.normal(0, 1, len(t))
        )

        vibrato_freq = 4 + speaker_idx * 0.3
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08

        return voice_sample.astype(np.float32)

    def _prepare_voice_audio(self, voice_audio: Union[np.ndarray, str]) -> np.ndarray:
        """Prepare voice audio for processing

        Args:
            voice_audio: Either a file path (str) or numpy array of audio samples

        Returns:
            Prepared audio as 1D float32 numpy array at 24000 Hz
        """
        target_sample_rate = 24000

        # Load from file if string path provided
        if isinstance(voice_audio, str):
            try:
                import librosa
                # librosa.load automatically resamples to target sample rate
                voice_audio, sr = librosa.load(voice_audio, sr=target_sample_rate, mono=True)
                logger.info(f"Loaded audio and resampled to {target_sample_rate} Hz")
            except ImportError:
                logger.warning("librosa not installed, falling back to soundfile")
                try:
                    import soundfile as sf
                    voice_audio, sr = sf.read(voice_audio)
                    voice_audio = voice_audio.astype(np.float32)

                    # Ensure mono
                    if voice_audio.ndim > 1:
                        voice_audio = voice_audio[0] if voice_audio.shape[0] == 1 else voice_audio[:, 0]

                    # Resample using librosa if available, otherwise use basic resampling
                    if sr != target_sample_rate:
                        try:
                            import librosa
                            voice_audio = librosa.resample(voice_audio, orig_sr=sr, target_sr=target_sample_rate)
                        except ImportError:
                            # Fallback to basic linear interpolation
                            logger.warning(f"Resampling from {sr} Hz to {target_sample_rate} Hz using basic interpolation. Install librosa for better quality.")
                            target_length = int(len(voice_audio) * target_sample_rate / sr)
                            voice_audio = np.interp(
                                np.linspace(0, len(voice_audio), target_length),
                                np.arange(len(voice_audio)),
                                voice_audio
                            )
                except Exception as e:
                    raise Exception(f"Failed to load audio file: {e}")
        else:
            # If numpy array provided, ensure it's 1D
            if voice_audio.ndim > 1:
                voice_audio = voice_audio[0] if voice_audio.shape[0] == 1 else voice_audio[:, 0]

            # For numpy arrays, we assume they're already at the target sample rate
            # since we can't auto-detect the sample rate from raw arrays
            logger.info("Received numpy array input - assuming it's already at 24000 Hz")

        # Normalize
        audio_max = np.abs(voice_audio).max()
        if audio_max > 0:
            voice_audio = voice_audio / max(audio_max, 1.0)

        return voice_audio.astype(np.float32)

    def _format_text_for_vibevoice(self, text: str) -> str:
        """Format text with speaker information for VibeVoice

        Args:
            text: Input text, can be plain text or multi-speaker format
                 Supports formats: "[1]: text" or "Speaker 1: text"

        Returns:
            Formatted text with proper "Speaker N:" prefixes, with newlines between speakers
        """
        # First check if text has [N]: format and convert to Speaker N: format
        bracket_pattern = r'\[(\d+)\]\s*:'
        if re.search(bracket_pattern, text):
            # Convert [N]: to Speaker N:
            def replace_bracket(match):
                speaker_num = match.group(1)
                return f'Speaker {speaker_num}:'
            text = re.sub(bracket_pattern, replace_bracket, text)

        # Detect if this is multi-speaker text
        speaker_pattern = r'Speaker\s+(\d+)\s*:'
        speaker_matches = list(re.finditer(speaker_pattern, text, re.IGNORECASE))
        unique_speakers = len(set([m.group(1) for m in speaker_matches]))
        is_multi_speaker = unique_speakers > 1

        # For multi-speaker: ensure newlines between speakers for proper voice mapping
        # For single-speaker: remove newlines completely
        if is_multi_speaker:
            # Split text by Speaker N: markers and reconstruct with newlines
            segments = []
            for i, match in enumerate(speaker_matches):
                speaker_label = match.group(0)  # "Speaker N:"
                start = match.end()

                # Find where this speaker's text ends (next speaker or end of text)
                if i + 1 < len(speaker_matches):
                    end = speaker_matches[i + 1].start()
                else:
                    end = len(text)

                # Extract and clean the speaker's text
                speaker_text = text[start:end].strip()
                # Clean up excessive whitespace within the text
                speaker_text = ' '.join(speaker_text.split())

                if speaker_text:  # Only add non-empty segments
                    segments.append(f"{speaker_label} {speaker_text}")

            # Join segments with newlines so processor can map voices correctly
            text = '\n'.join(segments)
        else:
            # Single speaker: remove newlines and clean up whitespace
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())

        # If text already has proper "Speaker N:" format, return as-is
        if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
            return text

        # Otherwise, prefix with Speaker 1
        return f"Speaker 1: {text}"

    def _detect_num_speakers(self, text: str) -> int:
        """Detect the number of speakers from text format

        Args:
            text: Text potentially containing "Speaker N:" or "[N]:" markers

        Returns:
            Number of unique speakers detected
        """
        # First check for [N]: format
        bracket_pattern = r'\[(\d+)\]\s*:'
        bracket_matches = re.findall(bracket_pattern, text)

        # Also check for "Speaker N:" patterns
        speaker_pattern = r'Speaker\s+(\d+)\s*:'
        speaker_matches = re.findall(speaker_pattern, text, re.IGNORECASE)

        # Combine both patterns
        all_matches = bracket_matches + speaker_matches

        if not all_matches:
            return 1  # Default to single speaker

        # Get unique speaker numbers
        speaker_nums = set(int(m) for m in all_matches)
        return len(speaker_nums)

    def generate(self, text: str, voice_samples: Optional[List[np.ndarray]] = None,
                 cfg_scale: float = 1.3, seed: int = 42, diffusion_steps: int = 20,
                 use_sampling: bool = False, temperature: float = 0.95,
                 top_p: float = 0.95) -> Dict[str, Any]:
        """Generate audio using VibeVoice model

        Args:
            text: Text to convert to speech. For multi-speaker, use format:
                  "Speaker 1: Hello Speaker 2: Hi there"
            voice_samples: List of voice samples (numpy arrays) for voice cloning.
                          For multi-speaker, provide one sample per speaker.
                          If None, synthetic voices will be used.
            cfg_scale: Classifier-free guidance scale (default: 1.3)
            seed: Random seed for reproducibility (default: 42)
            diffusion_steps: Number of diffusion steps (default: 20)
            use_sampling: Enable sampling mode (default: False for deterministic)
            temperature: Temperature for sampling (only used if use_sampling=True)
            top_p: Top-p for sampling (only used if use_sampling=True)

        Returns:
            Dictionary with "waveform" (numpy array) and "sample_rate" (int)
        """
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded. Call load_model() first.")

        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Set diffusion steps
        self.model.set_ddpm_inference_steps(diffusion_steps)

        # Format text for VibeVoice
        formatted_text = self._format_text_for_vibevoice(text)
        print(f"Formatted text")
        print(formatted_text)

        # Detect number of speakers
        num_speakers = self._detect_num_speakers(formatted_text)
        logger.info(f"Detected {num_speakers} speaker(s) in text")

        # Handle voice samples
        if voice_samples is None:
            # Create synthetic voices for each speaker
            logger.info(f"Creating {num_speakers} synthetic voice(s)")
            voice_samples = [self._create_synthetic_voice_sample(i) for i in range(num_speakers)]
        else:
            # Validate voice samples count
            if len(voice_samples) < num_speakers:
                logger.warning(f"Only {len(voice_samples)} voice sample(s) provided for {num_speakers} speaker(s)")
                logger.warning("Generating synthetic voices for missing speakers")
                # Pad with synthetic voices
                for i in range(len(voice_samples), num_speakers):
                    voice_samples.append(self._create_synthetic_voice_sample(i))
            elif len(voice_samples) > num_speakers:
                logger.warning(f"More voice samples ({len(voice_samples)}) than speakers ({num_speakers})")
                logger.warning(f"Using only first {num_speakers} voice sample(s)")
                voice_samples = voice_samples[:num_speakers]

        # Prepare inputs
        inputs = self.processor(
            [formatted_text],
            voice_samples=[voice_samples],  # Wrap in list as expected by processor
            return_tensors="pt",
            return_attention_mask=True
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")

        # Generate
        with torch.no_grad():
            if use_sampling:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=False,
                )

            if hasattr(output, 'speech_outputs') and output.speech_outputs:
                speech_tensors = output.speech_outputs

                if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                    audio_tensor = torch.cat(speech_tensors, dim=-1)
                else:
                    audio_tensor = speech_tensors

                # Convert to numpy
                audio_np = audio_tensor.cpu().float().numpy()

                # Ensure 1D
                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze()

                return {
                    "waveform": audio_np,
                    "sample_rate": 24000
                }
            else:
                raise Exception("Failed to generate audio - no speech outputs")


# Global inference instance
_inference = None


def generate_audio(
    text: str,
    model_path: str,
    tokenizer_path: str,
    voice_audio: Optional[Union[np.ndarray, str, List[Union[np.ndarray, str]]]] = None,
    attention_type: str = "auto",
    quantize_llm: str = "full precision",
    cfg_scale: float = 1.3,
    seed: int = 42,
    diffusion_steps: int = 20,
    use_sampling: bool = False,
    temperature: float = 0.95,
    top_p: float = 0.95,
    free_memory_after: bool = False,
) -> Dict[str, Any]:
    """
    Generate audio from text using VibeVoice.

    This is the main entry point for standalone inference. It wraps all the
    functionality from the ComfyUI node into a simple function call.

    Args:
        text: Text to convert to speech. For multi-speaker, use format:
              "Speaker 1: Hello Speaker 2: Hi there"
        model_path: Absolute path to the VibeVoice model directory containing
                   config.json and model files (e.g., "/path/to/VibeVoice-1.5B")
        tokenizer_path: Absolute path to the Qwen tokenizer directory containing
                   tokenizer_config.json, vocab.json, and merges.txt
        voice_audio: Optional reference audio for voice cloning. Can be:
                    - Single speaker: numpy array or path to audio file
                    - Multi-speaker: List of numpy arrays or paths (one per speaker)
                    - numpy arrays assumed to be at 24000 Hz
                    - audio files will be auto-resampled to 24000 Hz
                    If None, uses synthetic voices
        attention_type: Attention implementation to use:
                       - "auto": Let transformers choose the best
                       - "eager": Standard PyTorch attention
                       - "sdpa": Scaled dot product attention (optimized)
                       - "flash_attention_2": Flash Attention 2 (requires GPU)
                       - "sage": SageAttention (requires sageattention package + GPU)
        quantize_llm: Quantization mode for the LLM component:
                     - "full precision": No quantization (default)
                     - "4bit": 4-bit quantization (major VRAM savings, requires CUDA)
                     - "8bit": 8-bit quantization (balanced, requires CUDA)
        cfg_scale: Classifier-free guidance scale. Higher values = more guidance.
                  Official default is 1.3. Range: 0.5 - 3.5
        seed: Random seed for reproducibility (default: 42)
        diffusion_steps: Number of denoising diffusion steps. More = better quality
                        but slower. Official default is 20. Range: 1 - 100
        use_sampling: Enable sampling mode for more varied output (default: False)
        temperature: Temperature for sampling (only used if use_sampling=True).
                    Higher = more random. Range: 0.1 - 2.0
        top_p: Top-p (nucleus sampling) for sampling (only used if use_sampling=True).
              Range: 0.1 - 1.0
        free_memory_after: Free model from memory after generation (default: False).
                          Set to True if you're only doing one-off generations.

    Returns:
        Dictionary containing:
            - "waveform": numpy array of audio samples (1D float32 array)
            - "sample_rate": sample rate of the audio (typically 24000 Hz)

    Examples:
        # Single speaker with voice cloning
        audio = generate_audio(
            text="Hello, this is a test.",
            model_path="/path/to/VibeVoice-1.5B",
            tokenizer_path="/path/to/tokenizer",
            voice_audio="reference.wav"
        )

        # Multi-speaker with separate voices
        audio = generate_audio(
            text="Speaker 1: Hello there! Speaker 2: Hi, how are you?",
            model_path="/path/to/VibeVoice-1.5B",
            tokenizer_path="/path/to/tokenizer",
            voice_audio=["speaker1.wav", "speaker2.wav"]
        )

        # Multi-speaker with synthetic voices (no voice_audio provided)
        audio = generate_audio(
            text="Speaker 1: First person speaking. Speaker 2: Second person responding.",
            model_path="/path/to/VibeVoice-1.5B",
            tokenizer_path="/path/to/tokenizer"
        )
    """
    global _inference

    # Initialize inference instance if needed
    if _inference is None:
        _inference = VibeVoiceInference()

    # Load model
    _inference.load_model(model_path, tokenizer_path, attention_type, quantize_llm)

    # Prepare voice samples if provided
    voice_samples = None
    if voice_audio is not None:
        # Check if it's a list (multi-speaker) or single audio
        if isinstance(voice_audio, list):
            # Multi-speaker: prepare each voice audio
            voice_samples = []
            for i, audio in enumerate(voice_audio):
                prepared_audio = _inference._prepare_voice_audio(audio)
                voice_samples.append(prepared_audio)
            logger.info(f"Prepared {len(voice_samples)} voice sample(s) for multi-speaker")
        else:
            # Single speaker: prepare single audio
            prepared_audio = _inference._prepare_voice_audio(voice_audio)
            voice_samples = [prepared_audio]
            logger.info("Prepared single voice sample")
    else:
        # No voice audio provided, will use synthetic voices
        # The generate() method will handle creating synthetic voices based on detected speakers
        logger.info("No voice audio provided, will use synthetic voices")
        voice_samples = None

    # Generate audio
    result = _inference.generate(
        text=text,
        voice_samples=voice_samples,
        cfg_scale=cfg_scale,
        seed=seed,
        diffusion_steps=diffusion_steps,
        use_sampling=use_sampling,
        temperature=temperature,
        top_p=top_p
    )

    # Free memory if requested
    if free_memory_after:
        _inference.free_memory()
        _inference = None

    return result


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="VibeVoice Standalone Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--voice_audio", type=str, help="Reference audio for voice cloning")
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--diffusion_steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--quantize_llm", type=str, default="full precision",
                       choices=["full precision", "4bit", "8bit"])

    args = parser.parse_args()

    # Generate audio
    audio = generate_audio(
        text=args.text,
        model_path=args.model_path,
        voice_audio=args.voice_audio,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        quantize_llm=args.quantize_llm,
        free_memory_after=True
    )

    # Save audio
    try:
        import soundfile as sf
        sf.write(args.output, audio["waveform"], audio["sample_rate"])
        print(f"Audio saved to {args.output}")
    except ImportError:
        print("soundfile not installed. Install with: pip install soundfile")
        print("Saving as numpy array instead...")
        np.save(args.output.replace(".wav", ".npy"), audio["waveform"])
