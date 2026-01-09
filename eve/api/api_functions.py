"""
Non-route functions for the API module.
These are Modal functions and helper utilities that are not FastAPI routes.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import replicate
import sentry_sdk
from loguru import logger

from eve import utils
from eve.api.helpers import busy_state_dict
from eve.api.runner_tasks import (
    cancel_stuck_tasks,
    generate_lora_thumbnails,
    rotate_agent_metadata,
)
from eve.data_export import DataExport
from eve.mongo import get_collection
from eve.s3 import get_full_url, s3
from eve.task import Task, task_handler_func
from eve.tool import Tool, ToolContext
from eve.tools.replicate_tool import replicate_update_task
from eve.tools.tool_handlers import load_handler

db = os.getenv("DB", "STAGE").upper()
MARS_COLLEGE_FEATURE_FLAG = "mars_college_26"
MARS_COLLEGE_DAILY_MANNA_TARGET = 1000
DATA_EXPORT_TTL_HOURS = 48


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


async def process_cold_sessions_fn():
    """Scheduled function to process cold sessions for memory formation"""
    try:
        from eve.agent.memory.memory_cold_sessions_processor import (
            process_cold_sessions,
        )

        await process_cold_sessions()
    except Exception as e:
        logger.error(f"Error processing cold sessions: {e}")
        sentry_sdk.capture_exception(e)


async def topup_mars_college_manna_fn():
    """Top up subscription manna daily for Mars College users."""
    try:
        from eve.user import Manna, Transaction, User

        users = User.find(
            {
                "type": "user",
                "deleted": {"$ne": True},
                "featureFlags": MARS_COLLEGE_FEATURE_FLAG,
            }
        )

        if not users:
            logger.info(
                f"[MARS_COLLEGE_TOPUP] No users with flag {MARS_COLLEGE_FEATURE_FLAG}"
            )
            return

        updated_count = 0
        for user in users:
            try:
                manna = Manna.load(user.id)
                current_balance = manna.subscriptionBalance or 0
                if current_balance >= MARS_COLLEGE_DAILY_MANNA_TARGET:
                    continue

                topup_amount = MARS_COLLEGE_DAILY_MANNA_TARGET - current_balance
                manna.subscriptionBalance = MARS_COLLEGE_DAILY_MANNA_TARGET
                manna.save()

                Transaction(
                    manna=manna.id,
                    amount=topup_amount,
                    type="daily_topup_mars_college_26",
                ).save()
                updated_count += 1
            except Exception as e:
                logger.error(
                    f"[MARS_COLLEGE_TOPUP] Failed for user {user.id}: {e}",
                    exc_info=True,
                )
                sentry_sdk.capture_exception(e)

        logger.info(
            f"[MARS_COLLEGE_TOPUP] Topped up {updated_count} of {len(users)} users"
        )
    except Exception as e:
        logger.error(f"[MARS_COLLEGE_TOPUP] Error running top-up: {e}")
        sentry_sdk.capture_exception(e)


async def cleanup_expired_exports_fn():
    """Delete expired data exports from S3 and mark them expired."""
    try:
        now = datetime.now(timezone.utc)
        fallback_cutoff = now - timedelta(hours=DATA_EXPORT_TTL_HOURS)
        exports = DataExport.find(
            {
                "$or": [
                    {"expires_at": {"$lte": now}},
                    {
                        "expires_at": {"$exists": False},
                        "createdAt": {"$lte": fallback_cutoff},
                    },
                    {"expires_at": None, "createdAt": {"$lte": fallback_cutoff}},
                ],
                "status": {"$ne": "expired"},
            }
        )

        if not exports:
            return

        for export in exports:
            try:
                archive = export.archive or {}
                key = archive.get("key")
                bucket = (
                    archive.get("bucket")
                    or os.getenv("AWS_EXPORTS_BUCKET_NAME")
                    or os.getenv("AWS_BUCKET_NAME")
                    or os.getenv(f"AWS_BUCKET_NAME_{db}")
                )

                if key and key.startswith("http"):
                    try:
                        parsed = urlparse(key)
                        key = parsed.path.lstrip("/")
                    except Exception:
                        key = archive.get("key")

                if bucket and key:
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                    except Exception as s3_err:
                        logger.warning(
                            f"[EXPORT_CLEANUP] Failed to delete {bucket}/{key}: {s3_err}"
                        )

                export.status = "expired"
                export.save()
            except Exception as export_err:
                logger.error(
                    f"[EXPORT_CLEANUP] Failed to process export {export.id}: {export_err}",
                    exc_info=True,
                )
                sentry_sdk.capture_exception(export_err)
    except Exception as e:
        logger.error(f"[EXPORT_CLEANUP] Error running export cleanup: {e}")
        sentry_sdk.capture_exception(e)


# Modal task functions


async def run(
    tool_key: str,
    args: dict,
    user: str = None,
    agent: str = None,
    session: str = None,
    message: str = None,
    tool_call_id: str = None,
):
    handler = load_handler(tool_key)
    context = ToolContext(
        args=args,
        user=str(user) if user else None,
        agent=str(agent) if agent else None,
        session=str(session) if session else None,
        message=str(message) if message else None,
        tool_call_id=str(tool_call_id) if tool_call_id else None,
    )
    result = await handler(context)
    return utils.upload_result(result, tool_key=tool_key)


@task_handler_func
async def run_task(
    tool_key: str,
    args: dict,
    user: str = None,
    agent: str = None,
    session: str = None,
    message: str = None,
    tool_call_id: str = None,
):
    handler = load_handler(tool_key)
    context = ToolContext(
        args=args,
        user=str(user) if user else None,
        agent=str(agent) if agent else None,
        session=str(session) if session else None,
        message=str(message) if message else None,
        tool_call_id=str(tool_call_id) if tool_call_id else None,
    )
    return await handler(context)


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


async def cleanup_stuck_triggers():
    """Detect and auto-pause triggers stuck in 'running' state for >65 minutes.

    This function runs periodically to handle cases where triggers get stuck due to:
    - Modal container crashes (OOM, SIGKILL)
    - Network partitions
    - Infrastructure issues
    - Timeouts that bypass finally blocks

    Also fixes triggers with broken next_scheduled_run values:
    - Triggers with schedule but next_scheduled_run=None
    """
    try:
        from datetime import datetime, timedelta, timezone

        from eve.trigger import STUCK_TRIGGER_THRESHOLD_MINUTES, Trigger

        threshold = datetime.now(timezone.utc) - timedelta(
            minutes=STUCK_TRIGGER_THRESHOLD_MINUTES
        )
        logger.info(
            f"[CLEANUP_TRIGGERS] Starting stuck trigger cleanup (threshold: {STUCK_TRIGGER_THRESHOLD_MINUTES} minutes)..."
        )

        stuck_triggers = list(
            Trigger.find({"status": "running", "last_run_time": {"$lte": threshold}})
        )

        if not stuck_triggers:
            logger.info("[CLEANUP_TRIGGERS] No stuck triggers found")
            return 0

        logger.warning(f"[CLEANUP_TRIGGERS] Found {len(stuck_triggers)} stuck triggers")

        for trigger in stuck_triggers:
            logger.warning(
                f"[CLEANUP_TRIGGERS] Auto-pausing stuck trigger: id={trigger.id}, "
                f"name='{trigger.name}', last_run_time={trigger.last_run_time}"
            )

            trigger.update(
                status="paused",
                error_count=2,  # Set to MAX_ERROR_COUNT to indicate auto-paused
                last_error=f"Trigger stuck in running state since {trigger.last_run_time}. Auto-paused by cleanup job.",
            )

            # Send notification to user
            from eve.trigger import notify_trigger_paused

            notify_trigger_paused(
                trigger,
                f"Trigger was stuck in running state for over {STUCK_TRIGGER_THRESHOLD_MINUTES} minutes since {trigger.last_run_time}",
            )

        logger.info(
            f"[CLEANUP_TRIGGERS] Auto-paused {len(stuck_triggers)} stuck triggers"
        )

        # Also check for triggers with broken next_scheduled_run values
        # Conservative: only fix obviously wrong cases
        logger.info(
            "[CLEANUP_TRIGGERS] Checking for broken next_scheduled_run values..."
        )

        broken_triggers = list(
            Trigger.find(
                {
                    "status": "active",  # Only fix active triggers
                    "schedule": {"$ne": None, "$exists": True},  # Must have a schedule
                    "next_scheduled_run": None,  # But next_scheduled_run is missing
                }
            )
        )

        if broken_triggers:
            logger.warning(
                f"[CLEANUP_TRIGGERS] Found {len(broken_triggers)} triggers with schedule but no next_scheduled_run"
            )

            from eve.trigger import calculate_next_scheduled_run

            for trigger in broken_triggers:
                try:
                    next_run = calculate_next_scheduled_run(trigger.schedule)
                    if next_run:
                        trigger.update(next_scheduled_run=next_run)
                        logger.info(
                            f"[CLEANUP_TRIGGERS] Fixed trigger {trigger.id} ({trigger.name}): "
                            f"set next_scheduled_run={next_run}"
                        )
                    else:
                        # Schedule is exhausted, mark as finished
                        trigger.update(status="finished")
                        logger.info(
                            f"[CLEANUP_TRIGGERS] Trigger {trigger.id} ({trigger.name}) schedule exhausted, "
                            f"marked as finished"
                        )
                except Exception as e:
                    logger.error(
                        f"[CLEANUP_TRIGGERS] Failed to fix trigger {trigger.id}: {e}"
                    )
        else:
            logger.info("[CLEANUP_TRIGGERS] No broken next_scheduled_run values found")

        return len(stuck_triggers)

    except Exception as e:
        logger.error(f"Error in cleanup_stuck_triggers job: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return 0


async def embed_recent_creations():
    """Embed recent creations (images & videos) that don't have embeddings yet."""
    try:
        import io
        import json
        import subprocess
        from datetime import datetime, timedelta, timezone

        import requests
        import torch
        import torch.nn.functional as F
        from bson.binary import Binary, BinaryVectorDtype
        from PIL import Image
        from pymongo import UpdateOne
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Starting embed_recent_creations job")

        # ---- Settings ----
        MODEL_NAME = "openai/clip-vit-large-patch14"
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
