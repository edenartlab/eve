import os
import asyncio
import fal_client
from typing import Dict, Optional
from pydantic import Field

from .. import eden_utils
from ..tool import Tool, tool_context
from ..task import Task
from ..mongo import get_collection
from ..models import Creation

@tool_context("fal")
class FalTool(Tool):
    fal_endpoint: str
    with_logs: bool = Field(default=True, description="Whether to include logs in the response")
    output_handler: str = "normal"
    
    @Tool.handle_run
    async def async_run(self, args: Dict):
        check_fal_api_token()
        args = self._format_args_for_fal(args)
        
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(log["message"])
        
        result = fal_client.subscribe(
            self.fal_endpoint,
            arguments=args,
            with_logs=self.with_logs,
            on_queue_update=on_queue_update if self.with_logs else None,
        )
        
        result = eden_utils.upload_result({"output": result})
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        check_fal_api_token()
        args = self.prepare_args(task.args)
        args = self._format_args_for_fal(args)
        
        # Use webhook if provided
        webhook_url = get_webhook_url() if webhook else None
        if webhook_url:
            args["webhook_url"] = webhook_url
        
        # Submit the request to FAL queue and get the request_id
        handler = fal_client.submit(
            self.fal_endpoint,
            arguments=args
        )
        
        return handler.request_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        check_fal_api_token()
        request_id = task.handler_id
        
        while True:
            status = fal_client.status(self.fal_endpoint, request_id, with_logs=self.with_logs)
            
            if status.status == "FAILED":
                task.update(status="failed", error=status.error)
                task.refund_manna()
                return {"status": "failed", "error": status.error}
                
            elif status.status == "CANCELED":
                task.update(status="cancelled")
                task.refund_manna()
                return {"status": "cancelled"}
                
            elif status.status == "PROCESSING":
                task.status = "running"
                task.save()
                
            elif status.status == "COMPLETED":
                result = fal_client.result(self.fal_endpoint, request_id)
                processed_result = self._process_result(result, task)
                task.status = "completed"
                task.result = processed_result
                task.save()
                return {"status": "completed", "result": processed_result}
            
            await asyncio.sleep(1)

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        # FAL doesn't have a direct cancel API in their client yet
        # This is a placeholder for when they add that functionality
        pass
    
    def _format_args_for_fal(self, args: dict):
        """Format the arguments for FAL API call"""
        new_args = args.copy()
        new_args = {k: v for k, v in new_args.items() if v is not None}
        
        # Handle file uploads if needed
        for key, value in new_args.items():
            if isinstance(value, str) and os.path.isfile(value):
                new_args[key] = fal_client.upload_file(value)
                
        return new_args
    
    def _process_result(self, result, task):
        """Process the result from FAL API"""
        # Handle different result structures based on output_handler
        if self.output_handler == "normal":
            result = eden_utils.upload_result(
                {"output": result}, 
                save_thumbnails=True, 
                save_blurhash=True
            )
            processed_result = [{"output": [result]}]
            
            # Create creation object if needed
            for r, res in enumerate(processed_result):
                for o, output in enumerate(res["output"]):
                    name = task.args.get("prompt", "")
                    creation = Creation(
                        user=task.user,
                        agent=task.agent,
                        task=task.id,
                        tool=task.tool,
                        filename=output["filename"] if "filename" in output else None,
                        mediaAttributes=output.get("mediaAttributes", {}),
                        name=name,
                        public=task.public,
                    )
                    creation.save()
                    processed_result[r]["output"][o]["creation"] = creation.id
            
            return processed_result
        
        # Default return
        return {"output": result}


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


def check_fal_api_token():
    if not os.getenv("FAL_API_KEY"):
        raise Exception("FAL_API_KEY is not set") 