import os
import math
import tempfile
import subprocess
import numpy as np
from PIL import Image
from .... import eden_utils

def cosine_interpolation(x):
    """Cosine interpolation for smoother blending"""
    return (1 - np.cos(x * math.pi)) / 2

def smart_frame_selection(orig_size, target_size):
    """Intelligently select frames to keep when reducing, or duplicate when expanding"""
    if target_size >= orig_size:
        # For expansion, calculate how many times to repeat each frame
        repeats = np.zeros(orig_size, dtype=int)
        base_repeat = target_size // orig_size
        remainder = target_size % orig_size
        
        # Start with base repeats for all frames
        repeats += base_repeat
        
        if remainder > 0:
            # Distribute remaining repeats evenly, prioritizing middle frames
            if orig_size > 2:
                middle_indices = np.linspace(0, orig_size-1, remainder).round().astype(int)
                repeats[middle_indices] += 1
            else:
                repeats[0] += remainder

        # Generate frame indices
        indices = []
        for i, r in enumerate(repeats):
            indices.extend([i] * r)
        
        return np.array(indices)
    else:
        # For reduction, keep frames at regular intervals
        if target_size == 1:
            return np.array([0])
        elif target_size == 2:
            return np.array([0, orig_size-1])
        else:
            # Calculate middle frame indices
            middle_indices = np.linspace(1, orig_size-2, target_size-2).round().astype(int)
            return np.concatenate([[0], middle_indices, [orig_size-1]])

async def handler(args: dict, user: str = None, requester: str = None):
    # Get parameters
    video_url = args["video"]
    target_fps = args.get("target_fps")
    total_frames = args.get("total_frames")
    duration = args.get("duration")
    blend_strength = args["blend_strength"]
    method = args["method"]
    loop_seamless = args["loop_seamless"]

    # Download video
    input_video = eden_utils.download_file(video_url, video_url.split("/")[-1])
    
    # Create temporary directories for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        input_frames_dir = os.path.join(temp_dir, "input_frames")
        output_frames_dir = os.path.join(temp_dir, "output_frames")
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)

        # Get video info
        probe_command = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration",
            "-of", "json",
            input_video
        ]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True)
        if probe_result.returncode != 0:
            raise Exception(f"Error probing video: {probe_result.stderr}")

        import json
        video_info = json.loads(probe_result.stdout)
        fps_num, fps_den = map(int, video_info["streams"][0]["r_frame_rate"].split("/"))
        source_fps = fps_num / fps_den
        source_duration = float(video_info["streams"][0]["duration"])

        # Determine output timing parameters
        output_fps = target_fps if target_fps is not None else source_fps
        
        if total_frames is not None:
            # If total_frames is specified, it takes precedence
            target_frames = total_frames
            output_duration = target_frames / output_fps
        elif duration is not None:
            # If duration is specified, calculate frames from duration and FPS
            output_duration = duration
            target_frames = int(duration * output_fps)
        else:
            # Keep original duration if neither is specified
            output_duration = source_duration
            target_frames = int(source_duration * output_fps)

        # Extract frames at source FPS with explicit PNG format
        extract_command = [
            "ffmpeg", "-i", input_video,
            "-vf", f"fps={source_fps}",
            "-pix_fmt", "rgb24",  # Ensure RGB format
            "-f", "image2",       # Force image2 format
            "-c:v", "png",        # Force PNG codec
            os.path.join(input_frames_dir, "frame_%04d.png")
        ]
        print(f"Running frame extraction: {' '.join(extract_command)}")
        result = subprocess.run(extract_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise Exception(f"Error extracting frames: {result.stderr}")

        # Load frames
        input_frames = []
        frame_files = sorted(os.listdir(input_frames_dir))
        print(f"Found {len(frame_files)} frames")
        for frame_file in frame_files:
            frame_path = os.path.join(input_frames_dir, frame_file)
            try:
                img = Image.open(frame_path)
                img_array = np.array(img).astype(np.float32) / 255.0
                input_frames.append(img_array)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {str(e)}")
                raise

        images = np.stack(input_frames)
        orig_size = len(images)
        
        # Process frames using selected interpolation method
        if args.get("use_rife", False):
            from .rife import interpolate_sequence
            output = interpolate_sequence(images, target_frames, loop_seamless=loop_seamless)
        else:
            # Process frames using the same interpolation logic
            if orig_size == 1:
                output = np.tile(images, (target_frames, 1, 1, 1))
            elif orig_size == target_frames:
                output = images
            elif target_frames <= 1:
                output = images[:1]
            elif method == "nearest":
                if loop_seamless:
                    indices = (np.linspace(0, orig_size, target_frames) % orig_size).astype(int)
                    output = images[indices]
                else:
                    indices = smart_frame_selection(orig_size, target_frames)
                    output = images[indices]
            else:
                # Initialize output array
                output = np.empty((target_frames,) + images.shape[1:], dtype=np.float32)
                
                if loop_seamless:
                    # For looping, interpolate in a circular fashion
                    positions = np.linspace(0, orig_size, target_frames + 1)[:-1]
                    for i in range(target_frames):
                        pos = positions[i]
                        idx_low = int(pos % orig_size)
                        idx_high = int((pos + 1) % orig_size)
                        w = pos - int(pos)
                        
                        if method == "cosine":
                            w = cosine_interpolation(w)
                        
                        w = w * blend_strength
                        output[i] = (1 - w) * images[idx_low] + w * images[idx_high]
                else:
                    if target_frames > orig_size:
                        frame_indices = np.linspace(0, target_frames-1, orig_size).round().astype(int)
                        output[frame_indices] = images

                        for i in range(orig_size - 1):
                            start_idx = frame_indices[i]
                            end_idx = frame_indices[i + 1]
                            if end_idx - start_idx > 1:
                                steps = end_idx - start_idx
                                weights = np.linspace(0, 1, steps + 1)[1:-1]
                                
                                if method == "cosine":
                                    weights = cosine_interpolation(weights)
                                
                                weights = weights * blend_strength
                                
                                for j, w in enumerate(weights, 1):
                                    output[start_idx + j] = (1 - w) * images[i] + w * images[i + 1]
                    else:
                        scale = (orig_size - 1) / (target_frames - 1)
                        
                        for i in range(target_frames):
                            pos = i * scale
                            idx_low = int(pos)
                            idx_high = min(idx_low + 1, orig_size - 1)
                            w = pos - idx_low
                            
                            if method == "cosine":
                                w = cosine_interpolation(w)
                            
                            w = w * blend_strength
                            output[i] = (1 - w) * images[idx_low] + w * images[idx_high]

        # Save interpolated frames
        output = (output * 255).astype(np.uint8)
        for i, frame in enumerate(output):
            output_path = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
            Image.fromarray(frame).save(output_path)

        # Create output video with audio
        output_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        
        # Check if input video has audio
        probe_command = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0",
            input_video
        ]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True)
        has_audio = probe_result.stdout.strip() != ""

        # Construct ffmpeg command
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-framerate", str(output_fps),
            "-i", os.path.join(output_frames_dir, "frame_%04d.png")
        ]

        if has_audio:
            # Add input video for audio stream
            speed_factor = output_duration / source_duration
            preserve_pitch = args.get("preserve_pitch", True)
            
            if preserve_pitch:
                # Use rubberband filter for high-quality pitch-preserved stretching
                audio_filter = f"aresample=44100,rubberband=tempo={1/speed_factor}"
            else:
                # Use sample rate adjustment for pitch-varying stretching
                stretch_factor = 1/speed_factor
                original_rate = 44100
                intermediate_rate = int(original_rate * stretch_factor)
                audio_filter = (
                    f"asetrate={intermediate_rate},"  # Change sample rate to stretch/compress
                    f"aresample={original_rate}"      # Resample back to original rate
                )
            
            ffmpeg_command.extend([
                "-i", input_video,
                # Video settings
                "-map", "0:v:0",        # Use video from first input (PNG frames)
                "-map", "1:a:0",        # Use audio from second input (original video)
                "-c:v", "h264",         # Video codec
                "-crf", "23",           # Video quality
                "-preset", "fast",      # Encoding speed
                "-c:a", "aac",          # Audio codec
                "-b:a", "128k",         # Audio bitrate
                "-filter:a", audio_filter  # Audio stretching
            ])
        else:
            # No audio, just handle video
            ffmpeg_command.extend([
                "-c:v", "h264",
                "-crf", "23",
                "-preset", "fast",
                # Add silent audio
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest"
            ])

        ffmpeg_command.extend(["-movflags", "+faststart", output_video])
        
        print(f"Running video creation: {' '.join(ffmpeg_command)}")
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise Exception(f"Error creating output video: {result.stderr}")

    return {
        "output": output_video
    } 