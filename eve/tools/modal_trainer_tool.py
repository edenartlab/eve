"""Model trainers that run as their own Modal app (the FLUX LoRA trainer).

The trainer is deployed separately from this API, as the Modal app
``flux-trainer-<DB>``, and holds none of Eden's credentials: no database user,
no media-bucket key, no admin key. Its contract is:

    train(job)  ->  result
        job     {"task_id": "<task id>", "args": {<the task's arguments>}}
        result  {"status": "completed" | "failed", ...}   (see _register_model)

    Volume flux-trainer-jobs-<DB>
        <task id>/started.json   written when the container starts working
        <task id>/output/...     the LoRA, its checkpoint versions, a thumbnail

Everything that needs Eden credentials happens here, in the API: the job is
spawned from the task, and when it ends its files are copied from the Volume to
the media bucket, the model document is written and the task is completed
(trainer_update_task).

Completion is driven by polling, not by the trainer: a blocking caller polls in
async_wait, and sweep_pending_trainer_tasks (run from the periodic worker)
finalizes tasks that nobody is waiting on, such as trainings started from the
web app.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import modal
from loguru import logger

from .. import s3
from ..models import Model
from ..mongo import get_collection
from ..task import Task
from ..tool import Tool, ToolContext, tool_context

# Handler names of tools served by this class. "gcp" is what the FLUX trainer's
# tool document says from the time the trainer ran on Vertex AI; it is still
# accepted so the document does not have to change.
TRAINER_HANDLERS = ("modal_trainer", "gcp")

TRAINER_FUNCTION = "train"

TERMINAL_STATES = ("completed", "failed", "cancelled")

# How long one finalizer may hold the claim before another may take over.
_FINALIZE_LEASE = timedelta(minutes=15)

_POLL_SECONDS = 30


def _db() -> str:
    return os.getenv("DB", "STAGE").upper()


def trainer_app_name() -> str:
    return os.getenv("FLUX_TRAINER_MODAL_APP") or f"flux-trainer-{_db()}"


def jobs_volume_name() -> str:
    return os.getenv("FLUX_TRAINER_JOBS_VOLUME") or f"flux-trainer-jobs-{_db()}"


@tool_context("modal_trainer")
class ModalTrainerTool(Tool):
    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        raise NotImplementedError("Trainers run as tasks; use async_start_task")

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        return await asyncio.to_thread(
            submit_job, task_id=str(task.id), task_args=task.args
        )

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        while True:
            task.reload()
            if task.status in TERMINAL_STATES:
                return task.model_dump(include={"status", "error", "result"})
            try:
                result = await asyncio.to_thread(poll_and_update_task, task)
            except Exception as e:
                # Transient API failure: keep polling, the watchdog bounds it.
                logger.warning(f"Trainer status poll failed for {task.id}: {e}")
                await asyncio.sleep(_POLL_SECONDS)
                continue
            if result["status"] in TERMINAL_STATES:
                return result
            await asyncio.sleep(_POLL_SECONDS)

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        await asyncio.to_thread(cancel_job, task.handler_id)


# ---------------------------------------------------------------------------
# The trainer app and its job Volume
# ---------------------------------------------------------------------------


def _jobs_volume():
    return modal.Volume.from_name(jobs_volume_name(), environment_name="main")


def _job_path(task_id: str, relative_path: str) -> str:
    """Path of a file in a task's directory on the job Volume.

    relative_path comes from the trainer's result, so it is not allowed to point
    outside the directory.
    """
    relative_path = str(relative_path)
    parts = relative_path.split("/")
    if relative_path.startswith("/") or ".." in parts or "" in parts:
        raise ValueError(f"Unsafe path in trainer result: {relative_path}")
    return f"{task_id}/{relative_path}"


def _download_job_file(task_id: str, relative_path: str, local_dir: str) -> str:
    """Copy one file of the job directory to local_dir and return its path."""
    local_path = os.path.join(local_dir, str(relative_path).split("/")[-1])
    with open(local_path, "wb") as f:
        for chunk in _jobs_volume().read_file(_job_path(task_id, relative_path)):
            f.write(chunk)
    return local_path


def _read_job_json(task_id: str, name: str) -> Optional[dict]:
    try:
        data = b"".join(_jobs_volume().read_file(_job_path(task_id, name)))
    except Exception:
        return None  # not written yet
    try:
        return json.loads(data)
    except ValueError:
        return None


def _remove_job_dir(task_id: str):
    """Best effort: the Volume is scratch space, nothing reads it afterwards."""
    try:
        _jobs_volume().remove_file(task_id, recursive=True)
    except Exception as e:
        logger.debug(f"Could not remove trainer job directory {task_id}: {e}")


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def submit_job(task_id: str, task_args: dict) -> str:
    """Start a training run and return the Modal function call id.

    The trainer gets the task's arguments and nothing else: no user, no
    credentials. Its own narrowly scoped secret (the captioning key) is attached
    to the trainer app, not passed from here.
    """
    train = modal.Function.from_name(
        trainer_app_name(), TRAINER_FUNCTION, environment_name="main"
    )
    call = train.spawn({"task_id": task_id, "args": task_args})
    return call.object_id


def get_job_state(handler_id: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """(state, trainer result, error message).

    state is COMPLETED (the function returned; the result says whether training
    itself worked), ERROR (the function call failed or timed out), CANCELLED or
    ACTIVE (queued or running).
    """
    call = modal.FunctionCall.from_id(handler_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return "ACTIVE", None, None
    except modal.exception.InputCancellation:
        return "CANCELLED", None, None
    except modal.exception.FunctionTimeoutError:
        return "ERROR", None, "The training job ran out of time"
    except modal.exception.OutputExpiredError:
        return "ERROR", None, "The training job's result is no longer available"
    except Exception as e:
        if "cancel" in str(e).lower() or "terminated" in str(e).lower():
            return "CANCELLED", None, None
        return "ERROR", None, f"The training job failed: {str(e)[:300]}"
    if not isinstance(result, dict):
        return "ERROR", None, "The training job returned no result"
    return "COMPLETED", result, None


def cancel_job(handler_id: str) -> bool:
    try:
        # terminate_containers: a training run is one long subprocess; without
        # it the GPU would keep running (and billing) until the run ended.
        modal.FunctionCall.from_id(handler_id).cancel(terminate_containers=True)
        return True
    except Exception as e:
        logger.error(f"Error canceling trainer job {handler_id}: {str(e)}")
        return False


# ---------------------------------------------------------------------------
# Finalization: copy the outputs into Eden and close the task
# ---------------------------------------------------------------------------


def _register_model(task: Task, trainer_result: dict) -> list:
    """Upload the job's files to the media bucket, write the model document and
    return the task result.

    The document and the result have exactly the fields the trainer itself used
    to write when it ran on Vertex AI with database access, so everything that
    reads them (the web app, the LoRA inference tools) is unaffected.
    """
    task_id = str(task.id)

    with tempfile.TemporaryDirectory() as tmp:

        def to_media_bucket(relative_path):
            local = _download_job_file(task_id, relative_path, tmp)
            # Content-addressed: the object key is the file's SHA-256.
            url, _, _ = s3.upload_file(local)
            os.remove(local)
            return url.split("/")[-1]

        thumbnail = to_media_bucket(trainer_result["thumbnail"])
        checkpoint = to_media_bucket(trainer_result["lora"])
        checkpoint_versions = [
            {
                "step": int(version["step"]),
                "checkpoint": to_media_bucket(version["file"]),
            }
            for version in trainer_result.get("checkpoint_versions") or []
        ]

    now = datetime.now(timezone.utc)
    model_doc = {
        "args": task.args,
        "checkpoint": checkpoint,
        "checkpoint_versions": checkpoint_versions,
        "base_model": "flux-dev",
        "name": task.args["name"],
        "public": False,
        "deleted": False,
        "task": task.id,
        "thumbnail": thumbnail,
        "lora_trigger_text": trainer_result.get("lora_trigger_text"),
        "lora_mode": trainer_result.get("lora_mode"),
        "user": task.user,
        "agent": task.agent,
        "createdAt": now,
        "updatedAt": now,
    }
    prompts = trainer_result.get("custom_validation_prompts")
    if isinstance(prompts, list) and prompts:
        model_doc["custom_validation_prompts"] = prompts

    # Written with $setOnInsert under the task id, not through Model.save(): a
    # finalizer that is retried must not create a second model or touch the
    # first, and the document must not gain fields the old trainer never wrote.
    saved = get_collection(Model.collection_name).find_one_and_update(
        {"task": task.id},
        {"$setOnInsert": model_doc},
        upsert=True,
        return_document=True,
    )

    return [
        {
            "output": [
                {
                    "filename": checkpoint,
                    "metadata": trainer_result.get("config") or {},
                    "mediaAttributes": {"mimeType": "application/zip"},
                    "thumbnail": thumbnail,
                    "model": str(saved["_id"]),
                }
            ]
        }
    ]


def _close_unsuccessful(task: Task, status: str, error: Optional[str]) -> dict:
    update = {"status": status}
    if error:
        update["error"] = str(error)[:500]
    task.update(**update)
    task.refund_manna()
    _remove_job_dir(str(task.id))
    return update


def _seconds_since_created(task: Task, moment: datetime) -> float:
    return (moment - task.createdAt.replace(tzinfo=timezone.utc)).total_seconds()


def _started_at(task: Task) -> Optional[datetime]:
    """When the trainer began working on this task, if it has."""
    started = _read_job_json(str(task.id), "started.json")
    try:
        return datetime.fromisoformat(started["started_at"])
    except (TypeError, KeyError, ValueError):
        return None


def trainer_update_task(
    task: Task,
    state: str,
    trainer_result: Optional[dict] = None,
    job_error: Optional[str] = None,
) -> dict:
    """Bring a task in line with its training job.

    state is one of get_job_state's values. Blocking waiters and the periodic
    sweep can both call this for the same task, so a terminal state is handled
    under an atomic claim on the task document; the loser does nothing. A
    finalizer that crashes loses its claim after _FINALIZE_LEASE and the sweep
    tries again.
    """
    tasks = get_collection(Task.collection_name)
    now = datetime.now(timezone.utc)

    if state == "ACTIVE":
        if task.status == "pending":
            started_at = _started_at(task)
            if started_at:
                tasks.update_one(
                    {"_id": task.id, "status": "pending"},
                    {
                        "$set": {
                            "status": "running",
                            "performance.waitTime": _seconds_since_created(
                                task, started_at
                            ),
                            "updatedAt": now,
                        }
                    },
                )
                task.reload()
        return {"status": task.status}

    claimed = tasks.find_one_and_update(
        {
            "_id": task.id,
            "status": {"$nin": list(TERMINAL_STATES)},
            "$or": [
                {"finalizing_at": None},
                {"finalizing_at": {"$lt": now - _FINALIZE_LEASE}},
            ],
        },
        {"$set": {"finalizing_at": now}},
    )
    if claimed is None:
        # Terminal already, or another finalizer holds the claim.
        task.reload()
        return {"status": task.status}

    if state == "CANCELLED":
        return _close_unsuccessful(task, "cancelled", None)

    trainer_result = trainer_result or {}
    if state != "COMPLETED" or trainer_result.get("status") != "completed":
        error = (
            trainer_result.get("error")
            or job_error
            or "The training job ended without producing a model"
        )
        return _close_unsuccessful(task, "failed", error)

    try:
        result = _register_model(task, trainer_result)
    except Exception:
        # Let the sweep retry straight away instead of after the lease.
        tasks.update_one({"_id": task.id}, {"$unset": {"finalizing_at": ""}})
        raise

    finished_at = datetime.now(timezone.utc)
    started_at = _started_at(task)
    run_time = trainer_result.get("run_time")
    if started_at:
        wait_time = _seconds_since_created(task, started_at)
    else:
        wait_time = (task.performance or {}).get("waitTime") or 0
    if not isinstance(run_time, (int, float)):
        run_time = _seconds_since_created(task, finished_at) - wait_time
    performance = {"waitTime": wait_time, "runTime": run_time}

    # Conditional write: if the user cancelled while the files were being
    # copied, the cancel already refunded and must win.
    finished = tasks.find_one_and_update(
        {"_id": task.id, "status": {"$nin": list(TERMINAL_STATES)}},
        {
            "$set": {
                "status": "completed",
                "result": result,
                "performance": performance,
                "updatedAt": finished_at,
            },
            "$unset": {"finalizing_at": ""},
        },
    )
    if finished is None:
        task.reload()
        logger.warning(
            f"Trainer task {task.id} reached terminal state ({task.status}) during "
            "finalization; completed result discarded"
        )
        return {"status": task.status}

    _remove_job_dir(str(task.id))
    task.status = "completed"
    task.result = result
    task.performance = performance
    return {"status": "completed", "result": result}


def poll_and_update_task(task: Task) -> dict:
    state, trainer_result, job_error = get_job_state(task.handler_id)
    return trainer_update_task(task, state, trainer_result, job_error)


async def sweep_pending_trainer_tasks(limit: int = 20):
    """Finalize training tasks that no caller is waiting on.

    Does nothing, and makes no Modal call, when no such task is open.
    """
    tool_keys = [
        d["key"]
        for d in get_collection("tools3").find(
            {"handler": {"$in": list(TRAINER_HANDLERS)}}, {"key": 1}
        )
    ]
    if not tool_keys:
        return

    candidates = list(
        get_collection(Task.collection_name)
        .find(
            {
                "tool": {"$in": tool_keys},
                "status": {"$in": ["pending", "running"]},
                "handler_id": {"$exists": True, "$nin": [None, ""]},
            }
        )
        .sort("createdAt", 1)
        .limit(limit)
    )
    for doc in candidates:
        try:
            task = Task.from_schema(doc)
            result = await asyncio.to_thread(poll_and_update_task, task)
            if result["status"] in TERMINAL_STATES:
                logger.info(f"trainer sweep: task {task.id} is now {result['status']}")
        except Exception as e:
            logger.warning(f"trainer sweep: task {doc.get('_id')} failed: {e}")
