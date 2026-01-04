import asyncio
import os

from google.cloud import aiplatform
from google.oauth2 import service_account
from loguru import logger

from ..task import Task
from ..tool import Tool, ToolContext, tool_context


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
        )
        return handler_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        await poll_job_status(task.handler_id)
        task.reload()
        return task.result

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        await cancel_job(task.handler_id)


def get_ai_platform_client():
    # authenticate
    credentials = service_account.Credentials.from_service_account_info(
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
        }
    )

    # initialize client
    project_id = os.environ["GCP_PROJECT_ID"]
    location = os.environ["GCP_LOCATION"]
    staging_bucket = os.environ["GCP_STAGING_BUCKET"]

    aiplatform.init(
        project=project_id,
        location=location,
        credentials=credentials,
        staging_bucket=staging_bucket,
    )

    return aiplatform


GPUs = {
    "A100": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_A100,
    "T4": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4,
}


def submit_job(gcr_image_uri, machine_type, gpu, gpu_count, task_id):
    db = os.getenv("DB", "STAGE")
    aiplatform = get_ai_platform_client()
    job_name = f"flux-{task_id}"
    job = aiplatform.CustomJob(
        display_name=job_name,
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
                    "args": [f"--task_id={task_id}", f"--db={db}"],
                },
            }
        ],
    )

    job.submit()

    output = job.to_dict()
    job_id = output["name"]

    handler_id = job_id.split("/")[-1]

    return handler_id


async def poll_job_status(handler_id):
    while True:
        # job = await aiplatform.CustomJob.get_async(handler_id)
        job = await asyncio.to_thread(aiplatform.CustomJob.get, handler_id)
        status = job.state
        if status is None:
            status_str = "UNKNOWN"
        elif status == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
            status_str = "COMPLETED"
        elif status == aiplatform.gapic.JobState.JOB_STATE_FAILED:
            status_str = "ERROR"
        elif status == aiplatform.gapic.JobState.JOB_STATE_CANCELLED:
            status_str = "CANCELLED"
        elif status == aiplatform.gapic.JobState.JOB_STATE_RUNNING:
            status_str = "RUNNING"
        elif status == aiplatform.gapic.JobState.JOB_STATE_PENDING:
            status_str = "PENDING"
        else:
            status_str = str(status)

        if status_str in ["COMPLETED", "ERROR", "CANCELLED"]:
            return status_str

        await asyncio.sleep(20)


async def cancel_job(handler_id):
    try:
        job_prefix = os.environ["GCP_JOB_PREFIX"]
        job_id = f"{job_prefix}{handler_id}"
        aiplatform = get_ai_platform_client()
        # job = await aiplatform.CustomJob.get_async(job_id)
        job = await asyncio.to_thread(aiplatform.CustomJob.get, job_id)
        await job.cancel_async()
        return True
    except Exception as e:
        logger.error(f"Error canceling job {job_id}: {str(e)}")
        return False
