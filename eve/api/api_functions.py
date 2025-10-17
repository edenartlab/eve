"""
Non-route functions for the API module.
These are Modal functions and helper utilities that are not FastAPI routes.
"""

import os
import time
import replicate
import sentry_sdk

from eve import utils
from eve.mongo import get_collection
from eve.s3 import get_full_url
from eve.task import task_handler_func, Task
from eve.tool import Tool
from eve.tools.tool_handlers import load_handler
from eve.tools.replicate_tool import replicate_update_task
from eve.api.runner_tasks import (
    cancel_stuck_tasks,
    generate_lora_thumbnails,
    rotate_agent_metadata,
)
from eve.api.helpers import busy_state_dict
from loguru import logger

db = os.getenv("DB", "STAGE").upper()


# Modal scheduled functions


async def cancel_stuck_tasks_fn():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        logger.error(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)


async def generate_lora_thumbnails_fn():
    try:
        await generate_lora_thumbnails()
    except Exception as e:
        logger.error(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


async def rotate_agent_metadata_fn():
    try:
        await rotate_agent_metadata()
    except Exception as e:
        logger.error(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


# Modal task functions


async def run(
    tool_key: str, args: dict, user: str = None, agent: str = None, session: str = None
):
    handler = load_handler(tool_key)
    result = await handler(args, user, agent, session)
    return utils.upload_result(result)


@task_handler_func
async def run_task(
    tool_key: str, args: dict, user: str = None, agent: str = None, session: str = None
):
    handler = load_handler(tool_key)
    return await handler(args, user, agent, session)


async def run_task_replicate(task: Task):
    task.update(status="running")
    tool = Tool.load(task.tool)
    n_samples = task.args.get("n_samples", 1)
    n_runs = 1 if tool.parameters.get("n_samples") else n_samples
    replicate_model = tool._get_replicate_model(task.args)
    args = tool.prepare_args(task.args)
    args = tool._format_args_for_replicate(args)
    try:
        outputs = []
        for i in range(n_runs):
            task_args = args.copy()
            if "seed" in task_args:
                task_args["seed"] = task_args["seed"] + i
            output = await replicate.async_run(replicate_model, input=task_args)
            outputs.append(output)
        outputs = flatten_list(outputs)
        result = replicate_update_task(task, "succeeded", None, outputs, "normal")
    except Exception as e:
        logger.error(f"Error running replicate: {e}")
        sentry_sdk.capture_exception(e)
        result = replicate_update_task(task, "failed", str(e), None, "normal")
    return result


async def cleanup_stale_busy_states():
    """Clean up any stale busy states in the shared modal.Dict"""
    try:
        current_time = time.time()
        stale_threshold = 300
        logger.info("Starting stale busy state cleanup...")

        # Get all keys from the dictionary first
        all_keys = list(busy_state_dict.keys())  # This is not atomic but necessary

        for key in all_keys:
            try:
                # Get current state
                current_state = busy_state_dict.get(key)
                # Check if state exists and is a dictionary with expected structure
                if (
                    not current_state
                    or not isinstance(current_state, dict)
                    or not all(
                        k in current_state
                        for k in ["requests", "timestamps", "context_map"]
                    )
                ):
                    logger.warning(
                        f"Removing invalid/stale state for key {key}: {current_state}"
                    )
                    # Delete directly if possible and safe
                    if key in busy_state_dict:
                        busy_state_dict.pop(key)
                    continue

                requests = current_state.get("requests", [])
                timestamps = current_state.get("timestamps", {})
                context_map = current_state.get("context_map", {})

                # Ensure correct types after retrieval
                requests = list(requests)
                timestamps = dict(timestamps)
                context_map = dict(context_map)

                stale_requests = []
                active_requests = []
                updated_timestamps = {}
                updated_context_map = {}

                # Iterate over a copy of request IDs
                for request_id in list(requests):
                    timestamp = timestamps.get(request_id, 0)
                    if current_time - timestamp > stale_threshold:
                        stale_requests.append(request_id)
                        logger.info(
                            f"Marking request {request_id} as stale for key {key} (age: {current_time - timestamp:.1f}s)."
                        )
                    else:
                        active_requests.append(request_id)
                        if request_id in timestamps:
                            updated_timestamps[request_id] = timestamps[request_id]
                        if request_id in context_map:
                            updated_context_map[request_id] = context_map[request_id]

                # If any requests were found to be stale, update the state
                if stale_requests:
                    logger.info(
                        f"Cleaning up {len(stale_requests)} stale requests for {key}. Original count: {len(requests)}"
                    )
                    # Update the state in the modal.Dict
                    if not active_requests:
                        # If no active requests left, remove the whole key
                        logger.info(
                            f"Removing key '{key}' as no active requests remain after cleanup."
                        )
                        if key in busy_state_dict:  # Check existence before deleting
                            busy_state_dict.pop(key)
                    else:
                        # Otherwise, update with cleaned lists/dicts
                        new_state = {
                            "requests": active_requests,
                            "timestamps": updated_timestamps,
                            "context_map": updated_context_map,
                        }
                        busy_state_dict.put(key, new_state)
                        logger.info(
                            f"Updated state for key '{key}'. Active requests: {len(active_requests)}"
                        )
                # else: # No stale requests found for this key
                #    logger.debug(f"No stale requests found for key '{key}'.")
            except KeyError:
                logger.warning(
                    f"Key {key} was deleted concurrently during cleanup processing."
                )
                continue  # Key was likely deleted by another process or previous step
            except Exception as key_e:
                logger.error(
                    f"Error processing key '{key}' during cleanup: {key_e}",
                    exc_info=True,
                )
                # Decide how to handle errors: skip key, mark for later deletion, etc.
                # For now, just log and continue to avoid breaking the whole job.

        logger.info("Finished cleaning up stale busy states.")
    except Exception as e:
        logger.error(f"Error in cleanup_stale_busy_states job: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


async def embed_recent_creations():
    """Embed recent creations (images & videos) that don't have embeddings yet."""
    try:
        import io, json, subprocess
        import requests
        from datetime import datetime, timedelta, timezone
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from transformers import CLIPProcessor, CLIPModel
        from pymongo import UpdateOne
        from bson.binary import Binary, BinaryVectorDtype

        logger.info("Starting embed_recent_creations job")

        # ---- Settings ----
        MODEL_NAME = "openai/clip-vit-large-patch14"
        IMAGE_BATCH = 8  # how many images to embed at once
        VIDEO_FRAMES = 8  # how many frames to sample per video
        VIDEO_POOLING = "mean"  # "mean" | "self_sim" | "max"
        INCLUDE_THUMBNAIL_FRAME = (
            True  # include stored first-frame thumbnail if present
        )
        HTTP_TIMEOUT = 15

        col = get_collection("creations3")

        # Find recent (<= 1h) docs without embeddings; include image OR video
        cursor = col.find(
            {
                "embedding": {"$exists": False},
                "createdAt": {
                    "$gte": datetime.now(timezone.utc) - timedelta(minutes=30)
                },
                "$or": [
                    {"mediaAttributes.mimeType": {"$regex": "^image/"}},
                    {"mediaAttributes.mimeType": {"$regex": "^video/"}},
                ],
            },
            {"_id": 1, "filename": 1, "mediaAttributes": 1, "thumbnail": 1},
        ).limit(64)  # small batch for “recent” job

        docs = list(cursor)
        if not docs:
            logger.info("No recent creations to embed")
            return

        logger.info(f"Found {len(docs)} recent creations to embed")

        # ---- Load CLIP once for both images and video frames ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        proc = CLIPProcessor.from_pretrained(MODEL_NAME)

        # ---- Helpers ----
        @torch.no_grad()
        def embed_pil_batch(pil_images):
            if not pil_images:
                return None
            inputs = proc(images=pil_images, return_tensors="pt", padding=True).to(
                device
            )
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, p=2, dim=-1).detach().cpu()  # [N, D]
            return feats

        def _ffprobe_duration(url: str) -> float:
            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    url,
                ]
                cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return float(json.loads(cp.stdout)["format"]["duration"])
            except Exception as e:
                logger.warning(f"ffprobe failed for {url}: {e}")
                return 0.0

        def _ffmpeg_frame_at(url: str, t: float):
            try:
                cmd = [
                    "ffmpeg",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    url,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-loglevel",
                    "error",
                    "-",
                ]
                cp = subprocess.run(cmd, capture_output=True, check=True)
                return Image.open(io.BytesIO(cp.stdout)).convert("RGB")
            except Exception as e:
                logger.warning(f"ffmpeg extract failed @ {t}s for {url}: {e}")
                return None

        def sample_video_frames(url: str, k: int, thumb_url: str | None):
            frames = []
            if INCLUDE_THUMBNAIL_FRAME and thumb_url:
                try:
                    im = Image.open(
                        io.BytesIO(
                            requests.get(thumb_url, timeout=HTTP_TIMEOUT).content
                        )
                    ).convert("RGB")
                    frames.append(im)
                except Exception as e:
                    logger.warning(f"Failed to load thumbnail {thumb_url}: {e}")

            dur = _ffprobe_duration(url)
            if dur <= 0:
                times = [0.5, 1.5, 2.5, 3.5][: max(0, k)]
            else:
                if k == 1:
                    times = [0.5 * dur]
                else:
                    start, end = 0.10 * dur, 0.90 * dur
                    step = (end - start) / (k - 1)
                    times = [start + i * step for i in range(k)]
            for t in times:
                im = _ffmpeg_frame_at(url, t)
                if im is not None:
                    frames.append(im)
            return frames

        @torch.no_grad()
        def embed_video(url: str, thumb_url: str | None):
            frames = sample_video_frames(url, VIDEO_FRAMES, thumb_url)
            if not frames:
                return None
            feats = embed_pil_batch(frames)  # [N, D], unit vectors
            if feats is None or feats.shape[0] == 0:
                return None

            if VIDEO_POOLING == "self_sim" and feats.shape[0] > 1:
                sims = feats @ feats.T  # [N, N]
                w = sims.sum(dim=1)
                w = w / (w.sum() + 1e-9)
                pooled = (feats * w.unsqueeze(1)).sum(dim=0)
            elif VIDEO_POOLING == "max":
                pooled = feats.max(dim=0).values
            else:
                pooled = feats.mean(dim=0)
            pooled = F.normalize(pooled, p=2, dim=-1).detach().cpu().tolist()
            return pooled

        # ---- Partition docs ----
        def is_image_doc(d):
            mt = d.get("mediaAttributes", {}).get(
                "mimeType", d.get("mediaAttributes", {}).get("type", "")
            )
            return isinstance(mt, str) and mt.startswith("image/")

        def is_video_doc(d):
            mt = d.get("mediaAttributes", {}).get(
                "mimeType", d.get("mediaAttributes", {}).get("type", "")
            )
            return isinstance(mt, str) and mt.startswith("video/")

        image_docs = [d for d in docs if is_image_doc(d)]
        video_docs = [d for d in docs if is_video_doc(d)]

        # ---- Process images in small batch ----
        img_ops, img_done = [], 0
        if image_docs:
            ids, pil_images = [], []
            for d in image_docs:
                url = get_full_url(d["filename"])
                try:
                    im = Image.open(
                        io.BytesIO(requests.get(url, timeout=HTTP_TIMEOUT).content)
                    ).convert("RGB")
                    pil_images.append(im)
                    ids.append(d["_id"])
                except Exception as e:
                    logger.warning(f"Failed to load image {url}: {e}")

            if pil_images:
                feats = embed_pil_batch(pil_images)  # [M, D]
                if feats is not None:
                    for _id, f in zip(ids, feats):
                        v = f.tolist()
                        bindata_v = Binary.from_vector(v, BinaryVectorDtype.FLOAT32)
                        img_ops.append(
                            UpdateOne({"_id": _id}, {"$set": {"embedding": bindata_v}})
                        )
                    if img_ops:
                        result = col.bulk_write(img_ops, ordered=False)
                        img_done = result.modified_count

        # ---- Process videos one-by-one (ffmpeg cost dominates; batching adds little) ----
        vid_ops, vid_done = [], 0
        if video_docs:
            for d in video_docs:
                vurl = get_full_url(d["filename"])
                turl = get_full_url(d["thumbnail"]) if d.get("thumbnail") else None
                try:
                    v = embed_video(vurl, turl)
                    if v is None:
                        continue
                    bindata_v = Binary.from_vector(v, BinaryVectorDtype.FLOAT32)
                    vid_ops.append(
                        UpdateOne({"_id": d["_id"]}, {"$set": {"embedding": bindata_v}})
                    )
                except Exception as e:
                    logger.warning(f"Video embed failed for {vurl}: {e}")

            if vid_ops:
                result = col.bulk_write(vid_ops, ordered=False)
                vid_done = result.modified_count

        logger.info(f"Embedded {img_done} images and {vid_done} videos")

    except Exception as e:
        logger.error(f"Error in embed_recent_creations: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


# Helper functions


def flatten_list(seq):
    """Flattens a list that is either flat or nested one level deep."""
    return [x for item in seq for x in (item if isinstance(item, list) else [item])]
