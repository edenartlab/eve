"""The FLUX trainer job has no Eden credentials, so eve must close its tasks.

The container only reads args.json from, and writes result.json plus its output
files to, a per-task directory in the staging bucket. These tests lock down the
eve side of that contract:

- submission hands the job its arguments and nothing secret except the
  captioning key;
- a finished job is turned into a model document and a completed task exactly
  once, even when a blocking waiter and the periodic sweep race;
- a failed or cancelled job fails the task and refunds it;
- paths in result.json cannot reach outside the task's directory.
"""

from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from eve.task import Task
from eve.tools import gcp_tool


class _FakeTasks:
    """Emulates the conditional claim / terminal write on one task document."""

    def __init__(self, status="running"):
        self.doc = {"status": status, "finalizing_at": None}
        self.updates = []

    def find_one_and_update(self, filt, update):
        if self.doc["status"] in filt["status"]["$nin"]:
            return None
        if "$or" in filt and self.doc["finalizing_at"] is not None:
            lease_floor = filt["$or"][1]["finalizing_at"]["$lt"]
            if self.doc["finalizing_at"] >= lease_floor:
                return None
        before = dict(self.doc)
        self.doc.update(update.get("$set", {}))
        for key in update.get("$unset", {}):
            self.doc.pop(key, None)
        return before

    def update_one(self, filt, update):
        if filt.get("status") and self.doc["status"] != filt["status"]:
            return
        self.updates.append(update)
        for key, value in update.get("$set", {}).items():
            if "." not in key:
                self.doc[key] = value


def _task(status="running"):
    task = Task(
        user=ObjectId(),
        tool="flux_trainer",
        output_type="lora",
        args={"name": "Banny", "lora_training_urls": ["https://x/1.jpg"]},
        cost=660,
        handler_id="1234567890",
        status=status,
    )
    task.id = ObjectId()
    return task


def _apply_update(self, **kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)


TRAINER_RESULT = {
    "status": "completed",
    "lora": "output/banny.safetensors",
    "checkpoint_versions": [
        {"step": 2250, "file": "output/banny-step00002250.safetensors"}
    ],
    "thumbnail": "output/thumbnail.jpg",
    "lora_trigger_text": "a purple banana character",
    "lora_mode": "object",
    "config": {"mode": "object"},
}


def _finalize(task, tasks, state, trainer_result, job_error=None):
    saved = {}

    class _Model:
        def __init__(self, **fields):
            saved["fields"] = fields
            self.id = ObjectId()

        def save(self, upsert_filter=None, **extra):
            saved["upsert_filter"] = upsert_filter
            saved["extra"] = extra

    def _upload(path):
        return f"https://cdn.example/{path.split('/')[-1]}", "name", 1

    with (
        patch.object(gcp_tool, "get_collection", return_value=tasks),
        patch.object(gcp_tool, "_read_job_json", return_value=trainer_result),
        patch.object(
            gcp_tool,
            "_download_job_file",
            side_effect=lambda t, rel, d: f"{d}/{rel.split('/')[-1]}",
        ),
        patch.object(gcp_tool.s3, "upload_file", side_effect=_upload),
        patch.object(gcp_tool, "Model", _Model),
        patch.object(Task, "reload", lambda self: None),
        patch.object(Task, "update", _apply_update),
        patch.object(Task, "refund_manna", autospec=True) as refund,
    ):
        result = gcp_tool.gcp_update_task(task, state, job_error)
    return result, saved, refund.call_count


def test_finished_job_registers_model_and_completes_task():
    task, tasks = _task(), _FakeTasks()
    result, saved, refunds = _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)

    assert result["status"] == "completed"
    assert tasks.doc["status"] == "completed"
    assert "finalizing_at" not in tasks.doc

    fields = saved["fields"]
    assert fields["checkpoint"] == "banny.safetensors"
    assert fields["checkpoint_versions"] == [
        {"step": 2250, "checkpoint": "banny-step00002250.safetensors"}
    ]
    assert fields["thumbnail"] == "thumbnail.jpg"
    assert fields["base_model"] == "flux-dev"
    assert fields["lora_trigger_text"] == "a purple banana character"
    assert fields["task"] == task.id
    # one model per task, even if finalization is retried
    assert saved["upsert_filter"] == {"task": task.id}
    assert saved["extra"]["lora_mode"] == "object"

    output = result["result"][0]["output"][0]
    assert output["filename"] == "banny.safetensors"
    assert output["thumbnail"] == "thumbnail.jpg"
    assert refunds == 0


def test_second_finalizer_does_nothing():
    task, tasks = _task(), _FakeTasks()
    _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)
    _, saved, _ = _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)
    assert saved == {}  # no second model


def test_live_claim_blocks_a_concurrent_finalizer():
    from datetime import datetime, timezone

    task, tasks = _task(), _FakeTasks()
    tasks.doc["finalizing_at"] = datetime.now(timezone.utc)
    _, saved, _ = _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)
    assert saved == {}


def test_failed_job_fails_and_refunds_with_trainer_error():
    task, tasks = _task(), _FakeTasks()
    result, saved, refunds = _finalize(
        task,
        tasks,
        "ERROR",
        {"status": "failed", "error": "no usable images"},
        "exit 1",
    )
    assert result == {"status": "failed", "error": "no usable images"}
    assert saved == {}
    assert refunds == 1


def test_succeeded_job_without_result_is_a_failure():
    task, tasks = _task(), _FakeTasks()
    result, saved, refunds = _finalize(task, tasks, "COMPLETED", None)
    assert result["status"] == "failed"
    assert saved == {}
    assert refunds == 1


def test_cancelled_job_refunds():
    task, tasks = _task(), _FakeTasks()
    result, _, refunds = _finalize(task, tasks, "CANCELLED", None)
    assert result == {"status": "cancelled"}
    assert refunds == 1


def test_running_job_marks_pending_task_running_without_claiming():
    task, tasks = _task(status="pending"), _FakeTasks(status="pending")
    _, _, refunds = _finalize(task, tasks, "RUNNING", None)
    assert tasks.doc["status"] == "running"
    assert tasks.doc["finalizing_at"] is None
    assert refunds == 0


@pytest.mark.parametrize(
    "path", ["../other/args.json", "/etc/passwd", "output/../../x", "output//x"]
)
def test_result_paths_cannot_leave_the_job_directory(path):
    with pytest.raises(ValueError):
        gcp_tool._job_object("abc", path)


def test_staging_bucket_accepts_gs_prefix(monkeypatch):
    monkeypatch.setenv("GCP_STAGING_BUCKET", "gs://my-bucket/")
    assert gcp_tool.get_staging_bucket() == "my-bucket"
    assert gcp_tool.job_dir("abc") == "gs://my-bucket/flux-trainer/abc"
    monkeypatch.setenv("GCP_STAGING_BUCKET", "gs://my-bucket/some/path")
    with pytest.raises(ValueError):
        gcp_tool.get_staging_bucket()


def test_submit_passes_arguments_by_file_and_only_the_captioning_key(monkeypatch):
    monkeypatch.setenv("GCP_STAGING_BUCKET", "my-bucket")
    monkeypatch.setenv("OPENAI_API_KEY", "main-key")
    monkeypatch.setenv("FLUX_TRAINER_OPENAI_API_KEY", "trainer-key")
    monkeypatch.setenv("GCP_JOB_SERVICE_ACCOUNT", "job@example.iam.gserviceaccount.com")
    monkeypatch.setenv("MONGO_URI", "mongodb://secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    written = {}
    job = MagicMock()
    job.resource_name = "projects/1/locations/us-central1/customJobs/987"
    client = MagicMock()
    client.CustomJob.return_value = job

    with (
        patch.object(gcp_tool, "get_ai_platform_client", return_value=client),
        patch.object(
            gcp_tool,
            "_write_job_json",
            side_effect=lambda task_id, name, data: written.update({name: data}),
        ),
    ):
        handler_id = gcp_tool.submit_job(
            gcr_image_uri="registry/image:tag",
            machine_type="a2-highgpu-1g",
            gpu="A100",
            gpu_count=1,
            task_id="abc123",
            task_args={"name": "Banny"},
        )

    assert handler_id == "987"
    assert written["args.json"] == {"task_id": "abc123", "args": {"name": "Banny"}}

    spec = client.CustomJob.call_args.kwargs["worker_pool_specs"][0]["container_spec"]
    assert spec["args"] == ["--job_dir=gs://my-bucket/flux-trainer/abc123"]
    assert spec["env"] == [{"name": "OPENAI_API_KEY", "value": "trainer-key"}]
    assert "secret" not in str(client.CustomJob.call_args)

    submit = job.submit.call_args.kwargs
    assert submit["service_account"] == "job@example.iam.gserviceaccount.com"
    assert submit["timeout"] == gcp_tool.MAX_JOB_SECONDS
