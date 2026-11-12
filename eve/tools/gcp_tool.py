"""Vertex AI custom-job tools (the FLUX LoRA trainer).

The training container holds no credentials and never touches the database or
the media bucket. The contract with it is a per-task "job directory" in the
staging bucket:

    gs://<GCP_STAGING_BUCKET>/<JOB_ROOT>/<task id>/
        args.json       written here at submit time: {"task_id", "args"}
        result.json     written by the container when it finishes
        output/...      the LoRA, its checkpoint versions and a thumbnail

The container reads and writes that directory as the job's own service account
(Vertex mounts the bucket at /gcs/<bucket>). Everything that needs Eden
credentials happens here, after the job ends: the files are copied to the media
bucket, the model is registered and the task is completed (gcp_update_task).

Completion is driven by polling, not by the container: a blocking caller polls
in async_wait, and sweep_pending_gcp_tasks (run from the periodic worker)
finalizes tasks that nobody is waiting on, such as trainings started from the
web app.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from google.cloud import aiplatform
from google.oauth2 import service_account
from loguru import logger

from .. import s3
from ..models import Model
from ..mongo import get_collection
from ..task import Task
from ..tool import Tool, ToolContext, tool_context

# Folder inside the staging bucket that holds one directory per task.
JOB_ROOT = "flux-trainer"

# Hard stop for a single job. Vertex's own default is 7 days; a training run
# takes about an hour, so anything near this limit is stuck and only costs money.
MAX_JOB_SECONDS = 4 * 3600

TERMINAL_STATES = ("completed", "failed", "cancelled")

# How long one finalizer may hold the claim before another may take over.
_FINALIZE_LEASE = timedelta(minutes=15)

_POLL_SECONDS = 30


@tool_context("gcp")
class GCPTool(Tool):
    gcr_image_uri: str
    machine_type: str
    gpu: str

    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        raise NotImplementedError("Not implemented yet, need a GCP Task ID")

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        handler_id = await asyncio.to_thread(
            submit_job,
            gcr_image_uri=self.gcr_image_uri,
            machine_type=self.machine_type,
            gpu=self.gpu,
            gpu_count=1,
            task_id=str(task.id),
            task_args=task.args,
        )
        return handler_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        while True:
            task.reload()
            if task.status in TERMINAL_STATES:
                return task.model_dump(include={"status", "error", "result"})
            try:
                state, job_error = await asyncio.to_thread(
                    get_job_state, task.handler_id
                )
            except Exception as e:
                # Transient API failure: keep polling, the watchdog bounds it.
                logger.warning(f"GCP job status poll failed for {task.id}: {e}")
                await asyncio.sleep(_POLL_SECONDS)
                continue
            result = await asyncio.to_thread(gcp_update_task, task, state, job_error)
            if result["status"] in TERMINAL_STATES:
                return result
            await asyncio.sleep(_POLL_SECONDS)

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        await cancel_job(task.handler_id)


# ---------------------------------------------------------------------------
# Google Cloud clients
# ---------------------------------------------------------------------------


def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        {
            "type": os.environ["GCP_TYPE"],
            "project_id": os.environ["GCP_PROJECT_ID"],
            "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
            "private_key": os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": os.environ["GCP_CLIENT_EMAIL"],
            "client_id": os.environ["GCP_CLIENT_ID"],
            "auth_uri": os.environ["GCP_AUTH_URI"],
            "token_uri": os.environ["GCP_TOKEN_URI"],
            "auth_provider_x509_cert_url": os.environ[
                "GCP_AUTH_PROVIDER_X509_CERT_URL"
            ],
            "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
        },
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def get_staging_bucket() -> str:
    """Bucket name from GCP_STAGING_BUCKET, which may be written gs://name."""
    bucket = os.environ["GCP_STAGING_BUCKET"].strip()
    if bucket.startswith("gs://"):
        bucket = bucket[len("gs://") :]
    bucket = bucket.strip("/")
    if not bucket or "/" in bucket:
        raise ValueError("GCP_STAGING_BUCKET must name a bucket, with no path")
    return bucket


def get_ai_platform_client():
    aiplatform.init(
        project=os.environ["GCP_PROJECT_ID"],
        location=os.environ["GCP_LOCATION"],
        credentials=_get_credentials(),
        staging_bucket=f"gs://{get_staging_bucket()}",
    )
    return aiplatform


def _get_bucket():
    from google.cloud import storage

    client = storage.Client(
        project=os.environ["GCP_PROJECT_ID"], credentials=_get_credentials()
    )
    return client.bucket(get_staging_bucket())


def job_prefix(task_id: str) -> str:
    """Object prefix of a task's job directory inside the staging bucket."""
    return f"{JOB_ROOT}/{task_id}"


def job_dir(task_id: str) -> str:
    return f"gs://{get_staging_bucket()}/{job_prefix(task_id)}"


def _job_object(task_id: str, relative_path: str) -> str:
    """Object name of a file in a task's job directory.

    relative_path comes from result.json, which the container writes, so it is
    not allowed to point outside the directory.
    """
    relative_path = str(relative_path)
    parts = relative_path.split("/")
    if relative_path.startswith("/") or ".." in parts or "" in parts:
        raise ValueError(f"Unsafe path in trainer result: {relative_path}")
    return f"{job_prefix(task_id)}/{relative_path}"


def _write_job_json(task_id: str, name: str, data: dict):
    blob = _get_bucket().blob(_job_object(task_id, name))
    blob.upload_from_string(
        json.dumps(data, default=str), content_type="application/json"
    )


def _read_job_json(task_id: str, name: str) -> Optional[dict]:
    blob = _get_bucket().blob(_job_object(task_id, name))
    if not blob.exists():
        return None
    return json.loads(blob.download_as_bytes())


def _download_job_file(task_id: str, relative_path: str, local_dir: str) -> str:
    """Copy one file of the job directory to local_dir and return its path."""
    blob = _get_bucket().blob(_job_object(task_id, relative_path))
    local_path = os.path.join(local_dir, str(relative_path).split("/")[-1])
    blob.download_to_filename(local_path)
    return local_path


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

GPUs = {
    "A100": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_A100,
    "T4": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4,
}


def _job_env() -> list:
    """Per-job settings for the container.

    These are visible to anyone who can read the job in the project, so only the
    captioning key goes here. FLUX_TRAINER_OPENAI_API_KEY lets that be a
    separate, spend-capped key instead of the main one.
    """
    openai_key = (
        os.getenv("FLUX_TRAINER_OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"]
    )
    return [{"name": "OPENAI_API_KEY", "value": openai_key}]


def submit_job(gcr_image_uri, machine_type, gpu, gpu_count, task_id, task_args):
    db = os.getenv("DB", "STAGE")
    aiplatform = get_ai_platform_client()

    # The container has no database access: hand it the task arguments.
    _write_job_json(task_id, "args.json", {"task_id": task_id, "args": task_args})

    job = aiplatform.CustomJob(
        display_name=f"flux-{task_id}",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": GPUs[gpu],
                    "accelerator_count": gpu_count,
                },
                "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": gcr_image_uri,
                    "args": [f"--job_dir={job_dir(task_id)}"],
                    "env": _job_env(),
                },
            }
        ],
        base_output_dir=job_dir(task_id),
        labels={"eden-task": task_id.lower(), "eden-db": db.lower()},
    )

    job.submit(
        # Run as a dedicated account that can only use the staging bucket. When
        # unset, Vertex runs the job as the project's default service agent.
        service_account=os.getenv("GCP_JOB_SERVICE_ACCOUNT") or None,
        timeout=MAX_JOB_SECONDS,
    )

    return job.resource_name.split("/")[-1]


_JOB_STATES = {
    "JOB_STATE_SUCCEEDED": "COMPLETED",
    "JOB_STATE_FAILED": "ERROR",
    "JOB_STATE_EXPIRED": "ERROR",
    "JOB_STATE_CANCELLED": "CANCELLED",
    "JOB_STATE_CANCELLING": "RUNNING",
    "JOB_STATE_RUNNING": "RUNNING",
    "JOB_STATE_QUEUED": "PENDING",
    "JOB_STATE_PENDING": "PENDING",
}


def _get_job(handler_id: str):
    return aiplatform.CustomJob.get(
        handler_id,
        project=os.environ["GCP_PROJECT_ID"],
        location=os.environ["GCP_LOCATION"],
        credentials=_get_credentials(),
    )


def get_job_state(handler_id: str) -> Tuple[str, Optional[str]]:
    """(COMPLETED | ERROR | CANCELLED | RUNNING | PENDING, error message)."""
    job = _get_job(handler_id)
    state = _JOB_STATES.get(getattr(job.state, "name", str(job.state)), "PENDING")
    error = getattr(job.error, "message", None) if state == "ERROR" else None
    return state, error or None


async def poll_job_status(handler_id):
    while True:
        state, _ = await asyncio.to_thread(get_job_state, handler_id)
        if state in ["COMPLETED", "ERROR", "CANCELLED"]:
            return state
        await asyncio.sleep(_POLL_SECONDS)


async def cancel_job(handler_id):
    try:
        job = await asyncio.to_thread(_get_job, handler_id)
        await asyncio.to_thread(job.cancel)
        return True
    except Exception as e:
        logger.error(f"Error canceling job {handler_id}: {str(e)}")
        return False


# ---------------------------------------------------------------------------
# Finalization: copy the outputs into Eden and close the task
# ---------------------------------------------------------------------------


def _register_model(task: Task, trainer_result: dict) -> list:
    """Upload the job's files to the media bucket, save the model document and
    return the task result."""
    task_id = str(task.id)

    with tempfile.TemporaryDirectory() as tmp:

        def to_media_bucket(relative_path):
            local = _download_job_file(task_id, relative_path, tmp)
            url, _, _ = s3.upload_file(local)
            return url.split("/")[-1]

        checkpoint = to_media_bucket(trainer_result["lora"])
        checkpoint_versions = [
            {
                "step": int(version["step"]),
                "checkpoint": to_media_bucket(version["file"]),
            }
            for version in trainer_result.get("checkpoint_versions") or []
        ]
        thumbnail = to_media_bucket(trainer_result["thumbnail"])

    # Fields the Model class does not declare are passed to save() directly.
    extra = {"lora_mode": trainer_result.get("lora_mode")}
    if trainer_result.get("custom_validation_prompts"):
        extra["custom_validation_prompts"] = trainer_result["custom_validation_prompts"]

    model = Model(
        name=task.args["name"],
        user=task.user,
        agent=task.agent,
        task=task.id,
        thumbnail=thumbnail,
        args=task.args,
        checkpoint=checkpoint,
        checkpoint_versions=checkpoint_versions,
        base_model="flux-dev",
        lora_trigger_text=trainer_result.get("lora_trigger_text"),
    )
    # upsert_filter: a finalizer that is retried must not create a second model.
    model.save(upsert_filter={"task": task.id}, **extra)

    return [
        {
            "output": [
                {
                    "filename": checkpoint,
                    "metadata": trainer_result.get("config") or {},
                    "mediaAttributes": {"mimeType": "application/zip"},
                    "thumbnail": thumbnail,
                    "model": model.id,
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
    return update


def gcp_update_task(task: Task, state: str, job_error: Optional[str] = None) -> dict:
    """Bring a task in line with its Vertex job.

    state is one of get_job_state's values. Blocking waiters and the periodic
    sweep can both call this for the same task, so a terminal state is handled
    under an atomic claim on the task document; the loser does nothing. A
    finalizer that crashes loses its claim after _FINALIZE_LEASE and the sweep
    tries again.
    """
    tasks = get_collection(Task.collection_name)
    now = datetime.now(timezone.utc)

    if state in ("PENDING", "RUNNING"):
        if state == "RUNNING" and task.status == "pending":
            wait_time = (
                now - task.createdAt.replace(tzinfo=timezone.utc)
            ).total_seconds()
            tasks.update_one(
                {"_id": task.id, "status": "pending"},
                {
                    "$set": {
                        "status": "running",
                        "performance.waitTime": wait_time,
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

    trainer_result = _read_job_json(str(task.id), "result.json") or {}

    if state != "COMPLETED" or trainer_result.get("status") != "completed":
        error = (
            trainer_result.get("error")
            or job_error
            or "The training job ended without producing a model"
        )
        return _close_unsuccessful(task, "failed", error)

    result = _register_model(task, trainer_result)

    finished_at = datetime.now(timezone.utc)
    task.reload()
    performance = dict(task.performance or {})
    run_time = (
        finished_at - task.createdAt.replace(tzinfo=timezone.utc)
    ).total_seconds()
    performance["runTime"] = run_time - (performance.get("waitTime") or 0)

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
            f"GCP task {task.id} reached terminal state ({task.status}) during "
            "finalization; completed result discarded"
        )
        return {"status": task.status}

    task.status = "completed"
    task.result = result
    task.performance = performance
    return {"status": "completed", "result": result}


async def sweep_pending_gcp_tasks(limit: int = 20):
    """Finalize Vertex-job tasks that no caller is waiting on.

    Does nothing, and makes no Google Cloud call, when no such task is open.
    """
    gcp_tool_keys = [
        d["key"] for d in get_collection("tools3").find({"handler": "gcp"}, {"key": 1})
    ]
    if not gcp_tool_keys:
        return

    candidates = list(
        get_collection(Task.collection_name)
        .find(
            {
                "tool": {"$in": gcp_tool_keys},
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
            state, job_error = await asyncio.to_thread(get_job_state, task.handler_id)
            result = await asyncio.to_thread(gcp_update_task, task, state, job_error)
            if result["status"] in TERMINAL_STATES:
                logger.info(f"gcp sweep: task {task.id} is now {result['status']}")
        except Exception as e:
            logger.warning(f"gcp sweep: task {doc.get('_id')} failed: {e}")
