import os
import re
import modal
import asyncio
import tempfile
import replicate
from bson import ObjectId
from pydantic import Field
from typing import Dict, Optional
from datetime import datetime, timezone

from .. import s3
from .. import eden_utils
from ..tool import Tool, tool_context
from ..models import Model
from ..task import Task, Creation
from ..mongo import get_collection


@tool_context("replicate")
class ReplicateTool(Tool):
    replicate_model: str
    replicate_model_substitutions: Optional[Dict[str, str]] = None
    version: Optional[str] = Field(None, description="Replicate version to use")
    output_handler: str = "normal"

    @Tool.handle_run
    async def async_run(self, args: Dict):
        check_replicate_api_token()
        if self.version:
            args = self._format_args_for_replicate(args)
            prediction = await self._create_prediction(args, webhook=False)
            prediction.wait()
            if self.output_handler == "eden":
                result = {"output": prediction.output[-1]["files"][0]}
            elif self.output_handler == "trainer":
                result = {
                    "output": prediction.output[-1]["files"][0],
                    "thumbnail": prediction.output[-1]["thumbnails"][0],
                }
            else:
                result = {"output": prediction.output}
        else:
            replicate_model = self._get_replicate_model(args)
            args = self._format_args_for_replicate(args)
            result = {"output": replicate.run(replicate_model, input=args)}

        result = eden_utils.upload_result(result)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        check_replicate_api_token()
        if self.version:
            # Default: spawn Replicate task and await it in async_wait
            args = self.prepare_args(task.args)
            args = self._format_args_for_replicate(args)
            prediction = await self._create_prediction(args, webhook=webhook)
            return prediction.id
        else:
            # Replicate doesn't allow spawning tasks for models without a public version ID.
            # So we spawn a remote task on Modal which awaits the Replicate task
            db = os.getenv("DB", "STAGE").upper()
            func = modal.Function.lookup(
                f"api-{db.lower()}",
                "run_task_replicate", 
                environment_name="main"
            )
            job = func.spawn(task)
            return job.object_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        if self.version is None:
            fc = modal.functions.FunctionCall.from_id(task.handler_id)
            await fc.get.aio()
            task.reload()
            return task.model_dump(include={"status", "error", "result"})
        else:
            prediction = await replicate.predictions.async_get(task.handler_id)
            status = "starting"
            while True:
                if prediction.status != status:
                    status = prediction.status
                    result = replicate_update_task(
                        task,
                        status,
                        prediction.error,
                        prediction.output,
                        self.output_handler,
                    )
                    if result["status"] in ["failed", "cancelled", "completed"]:
                        return result
                await asyncio.sleep(0.5)
                prediction.reload()

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        prediction = replicate.predictions.get(task.handler_id)
        prediction.cancel()

    def _format_args_for_replicate(self, args: dict):
        new_args = args.copy()
        new_args = {k: v for k, v in new_args.items() if v is not None}
        for field in self.model.model_fields.keys():
            parameter = self.parameters[field]
            is_array = parameter.get("type") == "array"
            is_number = parameter.get("type") in ["integer", "float"]
            alias = parameter.get("alias")
            lora = parameter.get("type") == "lora"

            if field in new_args:
                if lora:
                    loras = get_collection(Model.collection_name)
                    lora_doc = (
                        loras.find_one({"_id": ObjectId(args[field])})
                        if args[field]
                        else None
                    )
                    if lora_doc:
                        lora_url = s3.get_full_url(lora_doc.get("checkpoint"))
                        lora_name = lora_doc.get("name")
                        lora_trigger_text = lora_doc.get("lora_trigger_text")
                        new_args[field] = lora_url
                        if "prompt" in new_args:
                            name_pattern = f"(\\b{re.escape(lora_name)}\\b|<{re.escape(lora_name)}>|\\<concept\\>)"
                            pattern = re.compile(name_pattern, re.IGNORECASE)
                            new_args["prompt"] = pattern.sub(
                                lora_trigger_text, new_args["prompt"]
                            )
                            if lora_trigger_text:
                                if lora_trigger_text not in new_args["prompt"]:
                                    new_args["prompt"] = f"{lora_trigger_text}, {new_args['prompt']}"

                if is_number:
                    new_args[field] = float(args[field])
                elif is_array:
                    new_args[field] = "|".join([str(p) for p in args[field]])
                if alias:
                    new_args[alias] = new_args.pop(field)

        return new_args

    def _get_replicate_model(self, args: dict):
        """Use default model or a substitute model conditional on an arg"""
        replicate_model = self.replicate_model

        if self.replicate_model_substitutions:
            for cond, model in self.replicate_model_substitutions.items():
                if args.get(cond):
                    replicate_model = model
                    break
        return replicate_model

    async def _create_prediction(self, args: dict, webhook=True):
        replicate_model = self._get_replicate_model(args)
        user, model = replicate_model.split("/", 1)
        webhook_url = get_webhook_url() if webhook else None
        webhook_events_filter = ["start", "completed"] if webhook else None

        if self.version == "deployment":
            deployment = await replicate.deployments.async_get(f"{user}/{model}")
            prediction = await deployment.predictions.async_create(
                input=args,
                webhook=webhook_url,
                webhook_events_filter=webhook_events_filter,
            )
        else:
            model = await replicate.models.async_get(f"{user}/{model}")
            version = await model.versions.async_get(self.version)
            prediction = await replicate.predictions.async_create(
                version=version,
                input=args,
                webhook=webhook_url,
                webhook_events_filter=webhook_events_filter,
            )
        return prediction


def get_webhook_url():
    env = {
        "PROD": "api-prod",
        "STAGE": "api-stage",
        "WEB3-PROD": "api-web3-prod",
        "WEB3-STAGE": "api-web3-stage",
    }.get(os.getenv("DB"), "api-web3-stage")
    dev = (
        "-dev"
        if os.getenv("DB") in ["WEB3-STAGE", "STAGE"]
        and os.getenv("MODAL_SERVE") == "1"
        else ""
    )

    webhook_url = f"https://edenartlab--{env}-fastapi-app{dev}.modal.run/update"
    return webhook_url


def replicate_update_task(task: Task, status, error, output, output_handler):
    output = output if isinstance(output, list) else [output]

    if output and isinstance(output[0], replicate.helpers.FileOutput):
        output_files = []
        for out in output:
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as temp_file:
                temp_file.write(out.read())
            output_files.append(temp_file.name)
        output = output_files

    if status == "failed":
        task.update(status="failed", error=error)
        task.refund_manna()
        return {"status": "failed", "error": error}

    elif status == "canceled":
        task.update(status="cancelled")
        task.refund_manna()
        return {"status": "cancelled"}

    elif status == "processing":
        task.performance["waitTime"] = (
            datetime.now(timezone.utc) - task.createdAt.replace(tzinfo=timezone.utc)
        ).total_seconds()
        task.status = "running"
        task.save()
        return {"status": "running"}

    elif status == "succeeded":
        if output_handler in ["eden", "trainer"]:
            thumbnails = output[-1]["thumbnails"]
            output = output[-1]["files"]
            output = eden_utils.upload_result(
                output, save_thumbnails=True, save_blurhash=True
            )
            if thumbnails:
                thumbnail_results = [
                    eden_utils.upload_media(
                        thumb, 
                        save_thumbnails=True,
                        save_blurhash=True
                    ) 
                    for thumb in thumbnails
                ]
            result = [{"output": [out]} for out in output]
        else:
            output = eden_utils.upload_result(
                output, save_thumbnails=True, save_blurhash=True
            )
            result = [{"output": [out]} for out in output]

        for r, res in enumerate(result):
            for o, output in enumerate(res["output"]):
                if output_handler == "trainer":
                    filename = output["filename"]
                    thumbnail = (
                        eden_utils.upload_media(
                            thumbnails[0], save_thumbnails=True, save_blurhash=True
                        )
                        if thumbnails
                        else None
                    )
                    url = s3.get_full_url(filename)
                    checkpoint_filename = url.split("/")[-1]
                    model = Model(
                        name=task.args["name"],
                        user=task.user,
                        requester=task.requester,
                        task=task.id,
                        thumbnail=thumbnail.get("filename"),
                        args=task.args,
                        checkpoint=checkpoint_filename,
                        base_model="sdxl",
                    )
                    model.save(
                        upsert_filter={"task": ObjectId(task.id)}
                    )  # upsert_filter prevents duplicates
                    output["model"] = model.id

                    # This is a hack to support legacy models for private endpoints.
                    # Change filename to url and copy record to the old models collection
                    if str(task.user) == os.getenv("LEGACY_USER_ID"):
                        model_copy = model.model_dump(by_alias=True)
                        model_copy["checkpoint"] = s3.get_full_url(
                            model_copy["checkpoint"]
                        )
                        model_copy["slug"] = f"legacy/{str(model_copy['_id'])}"
                        get_collection("models").insert_one(model_copy)

                else:
                    name = task.args.get("prompt")
                    creation = Creation(
                        user=task.user,
                        requester=task.requester,
                        task=task.id,
                        tool=task.tool,
                        filename=output["filename"],
                        mediaAttributes=output["mediaAttributes"],
                        name=name,
                    )
                    creation.save()
                    result[r]["output"][o]["creation"] = creation.id

        run_time = (
            datetime.now(timezone.utc) - task.createdAt.replace(tzinfo=timezone.utc)
        ).total_seconds()
        if task.performance.get("waitTime"):
            run_time -= task.performance["waitTime"]
        task.performance["runTime"] = run_time

        result = result if isinstance(result, list) else [result]
        task.status = "completed"
        task.result = result
        task.save()

        return {"status": "completed", "result": result}


def check_replicate_api_token():
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise Exception("REPLICATE_API_TOKEN is not set")
