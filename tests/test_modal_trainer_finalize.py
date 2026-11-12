"""The FLUX trainer runs as its own Modal app with no Eden credentials, so eve
must start its jobs and close its tasks.

The trainer only receives the task's arguments, and only writes its output
files to a per-task directory on a Modal Volume. These tests lock down the eve
side of that contract:

- submission hands the job its arguments and nothing else;
- a finished job is turned into a model document and a completed task exactly
  once, even when a blocking waiter and the periodic sweep race;
- the model document and the task result have exactly the fields the trainer
  wrote itself when it had database access;
- a failed or cancelled job fails the task and refunds it;
- paths in the trainer's result cannot reach outside the task's directory;
- the existing tool document (handler "gcp") still loads.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import modal
import pytest
from bson import ObjectId

from eve.task import Task
from eve.tools import modal_trainer_tool as trainer


class _FakeTasks:
    """Emulates the conditional claim / terminal write on one task document."""

    def __init__(self, status="running"):
        self.doc = {"status": status, "finalizing_at": None}
        self.updates = []

    def find_one_and_update(self, filt, update, **kwargs):
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
        for key in update.get("$unset", {}):
            self.doc.pop(key, None)


class _FakeModels:
    """Emulates the $setOnInsert upsert keyed on the task."""

    def __init__(self):
        self.docs = []

    def find_one_and_update(self, filt, update, upsert=False, return_document=False):
        for doc in self.docs:
            if doc["task"] == filt["task"]:
                return doc
        doc = {"_id": ObjectId(), **update["$setOnInsert"]}
        self.docs.append(doc)
        return doc


def _task(status="running"):
    task = Task(
        user=ObjectId(),
        tool="flux_trainer",
        output_type="lora",
        args={
            "name": "Banny",
            "lora_training_urls": ["https://x/1.jpg"],
            "mode": "auto",
            "max_train_steps": 3000,
            "lora_rank": 8,
            "learning_rate": 5e-05,
            "seed": 7,
        },
        cost=660,
        handler_id="fc-123",
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
        {"step": 3000, "file": "output/banny-step00003000.safetensors"},
        {"step": 2250, "file": "output/banny-step00002250.safetensors"},
        {"step": 1500, "file": "output/banny-step00001500.safetensors"},
    ],
    "thumbnail": "output/thumbnail.jpg",
    "lora_trigger_text": "a purple banana character",
    "lora_mode": "object",
    "config": {"mode": "object", "max_train_steps": "3000"},
    "run_time": 4507.0,
}


def _finalize(task, tasks, state, trainer_result, job_error=None, models=None):
    models = models if models is not None else _FakeModels()
    removed = []

    def _collection(name):
        return models if name == "models3" else tasks

    def _upload(path):
        return f"https://cdn.example/{path.split('/')[-1]}", "name", 1

    def _download(task_id, rel, local_dir):
        path = f"{local_dir}/{trainer._job_path(task_id, rel).split('/')[-1]}"
        open(path, "wb").close()
        return path

    with (
        patch.object(trainer, "get_collection", side_effect=_collection),
        patch.object(trainer, "_read_job_json", return_value=None),
        patch.object(trainer, "_download_job_file", side_effect=_download),
        patch.object(trainer, "_remove_job_dir", side_effect=removed.append),
        patch.object(trainer.s3, "upload_file", side_effect=_upload),
        patch.object(Task, "reload", lambda self: None),
        patch.object(Task, "update", _apply_update),
        patch.object(Task, "refund_manna", autospec=True) as refund,
    ):
        result = trainer.trainer_update_task(task, state, trainer_result, job_error)
    return result, models.docs, refund.call_count, removed


def test_finished_job_registers_model_and_completes_task():
    task, tasks = _task(), _FakeTasks()
    result, models, refunds, removed = _finalize(
        task, tasks, "COMPLETED", TRAINER_RESULT
    )

    assert result["status"] == "completed"
    assert tasks.doc["status"] == "completed"
    assert "finalizing_at" not in tasks.doc
    assert refunds == 0
    assert removed == [str(task.id)]

    (model,) = models
    assert model["checkpoint"] == "banny.safetensors"
    assert model["checkpoint_versions"] == [
        {"step": 3000, "checkpoint": "banny-step00003000.safetensors"},
        {"step": 2250, "checkpoint": "banny-step00002250.safetensors"},
        {"step": 1500, "checkpoint": "banny-step00001500.safetensors"},
    ]
    assert model["thumbnail"] == "thumbnail.jpg"
    assert model["lora_trigger_text"] == "a purple banana character"
    assert model["lora_mode"] == "object"
    assert model["task"] == task.id and model["user"] == task.user
    assert model["args"] == task.args


def test_model_document_and_task_result_match_the_old_trainer():
    """Field for field what the trainer wrote when it had database access."""
    task, tasks = _task(), _FakeTasks()
    result, (model,), _, _ = _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)

    assert set(model) == {
        "_id",
        "args",
        "checkpoint",
        "checkpoint_versions",
        "base_model",
        "name",
        "public",
        "deleted",
        "task",
        "thumbnail",
        "lora_trigger_text",
        "lora_mode",
        "user",
        "agent",
        "createdAt",
        "updatedAt",
    }
    assert model["base_model"] == "flux-dev"
    assert model["name"] == "Banny"
    assert model["public"] is False and model["deleted"] is False
    assert model["agent"] is None

    (entry,) = result["result"]
    (output,) = entry["output"]
    assert set(entry) == {"output"}
    assert output == {
        "filename": "banny.safetensors",
        "metadata": TRAINER_RESULT["config"],
        "mediaAttributes": {"mimeType": "application/zip"},
        "thumbnail": "thumbnail.jpg",
        "model": str(model["_id"]),
    }
    assert tasks.doc["performance"]["runTime"] == 4507.0
    assert set(tasks.doc["performance"]) == {"waitTime", "runTime"}


def test_custom_validation_prompts_are_stored_only_when_given():
    task, tasks = _task(), _FakeTasks()
    prompts = ["a", "b"]
    _, (model,), _, _ = _finalize(
        task,
        tasks,
        "COMPLETED",
        {**TRAINER_RESULT, "custom_validation_prompts": prompts},
    )
    assert model["custom_validation_prompts"] == prompts


def test_second_finalizer_does_nothing():
    task, tasks, models = _task(), _FakeTasks(), _FakeModels()
    _finalize(task, tasks, "COMPLETED", TRAINER_RESULT, models=models)
    _finalize(task, tasks, "COMPLETED", TRAINER_RESULT, models=models)
    assert len(models.docs) == 1


def test_retried_registration_reuses_the_model():
    """A finalizer that died after writing the model must not write another."""
    task, models = _task(), _FakeModels()
    _finalize(task, _FakeTasks(), "COMPLETED", TRAINER_RESULT, models=models)
    result, docs, _, _ = _finalize(
        task, _FakeTasks(), "COMPLETED", TRAINER_RESULT, models=models
    )
    assert len(docs) == 1
    assert result["result"][0]["output"][0]["model"] == str(docs[0]["_id"])


def test_live_claim_blocks_a_concurrent_finalizer():
    task, tasks = _task(), _FakeTasks()
    tasks.doc["finalizing_at"] = datetime.now(timezone.utc)
    _, models, _, _ = _finalize(task, tasks, "COMPLETED", TRAINER_RESULT)
    assert models == []


def test_failed_training_fails_and_refunds_with_trainer_error():
    task, tasks = _task(), _FakeTasks()
    result, models, refunds, removed = _finalize(
        task, tasks, "COMPLETED", {"status": "failed", "error": "no usable images"}
    )
    assert result == {"status": "failed", "error": "no usable images"}
    assert models == []
    assert refunds == 1
    assert removed == [str(task.id)]


def test_failed_function_call_fails_and_refunds():
    task, tasks = _task(), _FakeTasks()
    result, models, refunds, _ = _finalize(
        task, tasks, "ERROR", None, "The training job ran out of time"
    )
    assert result == {"status": "failed", "error": "The training job ran out of time"}
    assert models == []
    assert refunds == 1


def test_cancelled_job_refunds():
    task, tasks = _task(), _FakeTasks()
    result, _, refunds, _ = _finalize(task, tasks, "CANCELLED", None)
    assert result == {"status": "cancelled"}
    assert refunds == 1


def test_started_job_marks_pending_task_running_without_claiming():
    task, tasks = _task(status="pending"), _FakeTasks(status="pending")
    started = {"started_at": datetime.now(timezone.utc).isoformat()}
    with (
        patch.object(trainer, "get_collection", return_value=tasks),
        patch.object(trainer, "_read_job_json", return_value=started),
        patch.object(Task, "reload", lambda self: None),
    ):
        trainer.trainer_update_task(task, "ACTIVE")
    assert tasks.doc["status"] == "running"
    assert tasks.doc["finalizing_at"] is None
    assert "performance.waitTime" in tasks.updates[0]["$set"]


def test_queued_job_stays_pending():
    task, tasks = _task(status="pending"), _FakeTasks(status="pending")
    with (
        patch.object(trainer, "get_collection", return_value=tasks),
        patch.object(trainer, "_read_job_json", return_value=None),
        patch.object(Task, "reload", lambda self: None),
    ):
        result = trainer.trainer_update_task(task, "ACTIVE")
    assert result == {"status": "pending"}
    assert tasks.updates == []


@pytest.mark.parametrize(
    "path", ["../other/started.json", "/etc/passwd", "output/../../x", "output//x"]
)
def test_result_paths_cannot_leave_the_job_directory(path):
    with pytest.raises(ValueError):
        trainer._job_path("abc", path)


def test_submit_passes_only_the_task_arguments(monkeypatch):
    monkeypatch.setenv("DB", "PROD")
    monkeypatch.setenv("OPENAI_API_KEY", "main-key")
    monkeypatch.setenv("MONGO_URI", "mongodb://secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    function = MagicMock()
    function.spawn.return_value.object_id = "fc-987"
    with patch.object(modal.Function, "from_name", return_value=function) as lookup:
        handler_id = trainer.submit_job(task_id="abc123", task_args={"name": "Banny"})

    assert handler_id == "fc-987"
    assert lookup.call_args.args == ("flux-trainer-PROD", "train")
    function.spawn.assert_called_once_with(
        {"task_id": "abc123", "args": {"name": "Banny"}}
    )
    assert "secret" not in str(function.spawn.call_args)
    assert "main-key" not in str(function.spawn.call_args)


def _state_for(get_side_effect):
    call = MagicMock()
    call.get.side_effect = get_side_effect
    with patch.object(modal.FunctionCall, "from_id", return_value=call):
        return trainer.get_job_state("fc-1")


def test_job_state_mapping():
    assert _state_for(TimeoutError())[0] == "ACTIVE"
    assert _state_for(modal.exception.FunctionTimeoutError("t"))[0] == "ERROR"
    assert _state_for(modal.exception.OutputExpiredError())[0] == "ERROR"
    assert _state_for(modal.exception.InputCancellation())[0] == "CANCELLED"
    state, _, error = _state_for(RuntimeError("container ran out of memory"))
    assert state == "ERROR" and "out of memory" in error
    state, result, _ = _state_for(lambda timeout: {"status": "completed"})
    assert state == "COMPLETED" and result == {"status": "completed"}
    assert _state_for(lambda timeout: None)[0] == "ERROR"


def test_cancel_terminates_the_container():
    call = MagicMock()
    with patch.object(modal.FunctionCall, "from_id", return_value=call):
        assert trainer.cancel_job("fc-1") is True
    call.cancel.assert_called_once_with(terminate_containers=True)


@pytest.mark.parametrize("handler", ["modal_trainer", "gcp"])
def test_new_and_existing_tool_documents_load_the_trainer(handler):
    """The stored tool document still says handler "gcp" and carries the old
    Vertex fields; it must load without being changed."""
    from eve.tool import Tool

    cls = Tool.get_sub_class({"handler": handler})
    assert cls is trainer.ModalTrainerTool
