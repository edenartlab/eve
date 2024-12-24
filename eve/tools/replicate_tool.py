import os
import re
import asyncio
import random
import replicate
from bson import ObjectId
from pydantic import Field
from typing import Dict, Optional, List
from datetime import datetime, timezone

from .. import s3
from .. import eden_utils
from ..models import Model
from ..user import User
from ..task import Task, Creation
from ..tool import Tool
from ..mongo import get_collection
                    

class ReplicateTool(Tool):
    replicate_model: str
    replicate_model_substitutions: Optional[Dict[str, str]] = None
    version: Optional[str] = Field(None, description="Replicate version to use")
    output_handler: str = "normal"
    
    @Tool.handle_run
    async def async_run(self, args: Dict, db: str):
        check_replicate_api_token()
        args = self._format_args_for_replicate(args)
        if self.version:
            prediction = await self._create_prediction(args, webhook=False)        
            prediction.wait()
            if self.output_handler == "eden":
                result = {"output": prediction.output[-1]["files"][0]}
            elif self.output_handler == "trainer":
                result = {
                    "output": prediction.output[-1]["files"][0],
                    "thumbnail": prediction.output[-1]["thumbnails"][0]
                }
            else:
                result = {"output": prediction.output}
        else:
            replicate_model = self._get_replicate_model(args)
            result = {
                "output": replicate.run(replicate_model, input=args)
            }
        result = eden_utils.upload_result(result, db=db)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        check_replicate_api_token()
        args = self.prepare_args(task.args)
        args = self._format_args_for_replicate(args)
        if self.version:
            prediction = await self._create_prediction(args, webhook=webhook)
            return prediction.id
        else:
            # Replicate doesn't allow spawning tasks for models without a public version ID.
            # So just get run and finish task immediately
            replicate_model = self._get_replicate_model(task.args)
            output = replicate.run(replicate_model, input=args)
            replicate_update_task(task, "succeeded", None, output, "normal")
            handler_id = eden_utils.random_string(28)  # make up a fake Replicate id
            return handler_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        if self.version is None:
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
                        self.output_handler
                    )
                    if result["status"] in ["failed", "cancelled", "completed"]:
                        return result
                await asyncio.sleep(0.5)
                prediction.reload()

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        try:
            prediction = replicate.predictions.get(task.handler_id)
            prediction.cancel()
        except Exception as e:
            print("Replicate cancel error, probably task is timed out or already finished", e)

    def _format_args_for_replicate(self, args: dict):
        new_args = args.copy()
        new_args = {k: v for k, v in new_args.items() if v is not None}
        for field in self.model.model_fields.keys():
            parameter = self.parameters[field]
            is_array = parameter.get('type') == 'array'
            is_number = parameter.get('type') in ['integer', 'float']
            alias = parameter.get('alias')
            lora = parameter.get('type') == 'lora'
            if field in new_args:
                if lora:
                    lora_doc = get_collection(Model.collection_name, db=self.db).find_one({"_id": ObjectId(args[field])}) if args[field] else None
                    if lora_doc:
                        lora_url = lora_doc.get("checkpoint")
                        lora_name = lora_doc.get("name")
                        caption_prefix = lora_doc.get("args", {}).get("caption_prefix")
                        new_args[field] = lora_url
                        if "prompt" in new_args:
                            pattern = re.compile(re.escape(lora_name), re.IGNORECASE)
                            new_args["prompt"] = pattern.sub(caption_prefix, new_args['prompt'])
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
        user, model = replicate_model.split('/', 1)
        
        webhook_url = get_webhook_url() if webhook else None
        webhook_events_filter = ["start", "completed"] if webhook else None

        if self.version == "deployment":
            deployment = await replicate.deployments.async_get(f"{user}/{model}")
            prediction = await deployment.predictions.async_create(
                input=args,
                webhook=webhook_url,
                webhook_events_filter=webhook_events_filter
            )
        else:
            model = await replicate.models.async_get(f"{user}/{model}")
            version = await model.versions.async_get(self.version)
            prediction = await replicate.predictions.async_create(
                version=version,
                input=args,
                webhook=webhook_url,
                webhook_events_filter=webhook_events_filter
            )
        return prediction

def get_webhook_url():
    env = "tools" if os.getenv("ENV") == "PROD" else "tools-dev"
    dev = "-dev" if os.getenv("ENV") == "STAGE" and os.getenv("MODAL_SERVE") == "1" else ""
    webhook_url = f"https://edenartlab--{env}-fastapi-app{dev}.modal.run/update"
    return webhook_url


def replicate_update_task(task: Task, status, error, output, output_handler):
    output = output if isinstance(output, list) else [output]

    if status == "failed":
        task.update(status="failed", error=error)
        n_samples = task.args.get("n_samples", 1)
        refund_amount = (task.cost or 0) * (n_samples - len(task.result or [])) / n_samples
        user = User.from_mongo(task.user, db=task.db)
        user.refund_manna(refund_amount)
        return {"status": "failed", "error": error}
    
    elif status == "canceled":
        task.update(status="cancelled")
        n_samples = task.args.get("n_samples", 1)
        refund_amount = (task.cost or 0) * (n_samples - len(task.result or [])) / n_samples
        user = User.from_mongo(task.user, db=task.db)
        user.refund_manna(refund_amount)
        return {"status": "cancelled"}
    
    elif status == "processing":
        task.performance["waitTime"] = (datetime.now(timezone.utc) - task.createdAt).total_seconds()
        task.status = "running"
        task.save()
        return {"status": "running"}
    
    elif status == "succeeded":
        
        if output_handler in ["eden", "trainer"]:
            thumbnails = output[-1]["thumbnails"]
            output = output[-1]["files"]
            output = eden_utils.upload_result(output, db=task.db, save_thumbnails=True, save_blurhash=True)
            result = [{"output": [out]} for out in output]
        else:
            output = eden_utils.upload_result(output, db=task.db, save_thumbnails=True, save_blurhash=True)
            result = [{"output": [out]} for out in output]

        for r, res in enumerate(result):
            for o, output in enumerate(res["output"]):
                if output_handler == "trainer":
                    filename = output["filename"]
                    thumbnail = eden_utils.upload_media(
                        thumbnails[0], 
                        db=task.db, 
                        save_thumbnails=False, 
                        save_blurhash=False
                    ) if thumbnails else None
                    url = f"{s3.get_root_url(db=task.db)}/{filename}"
                    model = Model(
                        name=task.args["name"],
                        user=task.user,
                        requester=task.requester,
                        task=task.id,
                        thumbnail=thumbnail.get("filename"),
                        args=task.args,
                        checkpoint=url, 
                        base_model="sdxl",
                    )
                    model.save(upsert_filter={"task": ObjectId(task.id)})  # upsert_filter prevents duplicates
                    output["model"] = model.id
                
                else:
                    name = task.args.get("prompt")
                    creation = Creation(
                        user=task.user,
                        requester=task.requester,
                        task=task.id,
                        tool=task.tool,
                        filename=output['filename'],
                        mediaAttributes=output["mediaAttributes"],
                        name=name
                    )
                    creation.save(db=task.db)
                    result[r]["output"][o]["creation"] = creation.id
        
        run_time = (datetime.now(timezone.utc) - task.createdAt).total_seconds()
        if task.performance.get("waitTime"):
            run_time -= task.performance["waitTime"]
        task.performance["runTime"] = run_time
        result = result if isinstance(result, list) else [result]
        task.status = "completed"
        task.result = result
        task.save()

        return {
            "status": "completed", 
            "result": result
        }


def check_replicate_api_token():
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise Exception("REPLICATE_API_TOKEN is not set")