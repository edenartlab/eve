import os
import re
import yaml
import json
import random
import asyncio
import eden_utils
from pydantic import BaseModel, Field, create_model, ValidationError
from typing import Optional, List, Dict, Any, Type, Literal

import s3
from base import parse_schema
from models import Task, User


class Tool(BaseModel):
    """
    Base class for all tools.
    """

    key: str
    name: str
    description: str
    tip: Optional[str] = None
    cost_estimate: str
    output_type: Literal["bool", "str", "int", "float", "string", "image", "video", "audio", "image|video", "image|audio", "video|audio", "image|video|audio"]
    resolutions: Optional[List[str]] = None
    base_model: Optional[str] = "sdxl"
    gpu: Optional[str] = None    
    status: Optional[Literal["inactive", "stage", "prod"]] = "stage"
    allowlist: Optional[str] = None
    visible: Optional[bool] = True
    test_args: Dict[str, Any]
    handler: Literal["modal", "comfyui", "replicate", "runway", "elevenlabs", "hedra"] = "modal"
    base_model: Type[BaseModel] = Field(None, exclude=True)

    @classmethod
    def from_dir(cls, tool_dir: str, **kwargs):
        key = tool_dir.split('/')[-1]
        yaml_file = os.path.join(tool_dir, 'api.yaml')
        test_file = os.path.join(tool_dir, 'test.json')

        with open(yaml_file, 'r') as f:
            schema = yaml.safe_load(f)
                
        with open(test_file, 'r') as f:
            test_args = json.load(f)

        fields = parse_schema(schema)
        base_model = create_model(key, **fields)
        base_model.__doc__ = eden_utils.concat_sentences(schema.get('description'), schema.get('tip', ''))

        tool_data = {k: schema.pop(k) for k in cls.model_fields.keys() if k in schema}
        tool_data['test_args'] = test_args
        tool_data['base_model'] = base_model
        
        if 'cost_estimate' in tool_data:
            tool_data['cost_estimate'] = str(tool_data['cost_estimate'])

        return cls(key=key, **tool_data, **kwargs)

    def calculate_cost(self, args):
        if not self.cost_estimate:
            return 0
        cost_formula = self.cost_estimate
        cost_formula = re.sub(r'(\w+)\.length', r'len(\1)', cost_formula)  # Array length
        cost_formula = re.sub(r'(\w+)\s*\?\s*([^:]+)\s*:\s*([^,\s]+)', r'\2 if \1 else \3', cost_formula)  # Ternary operator
        
        cost_estimate = eval(cost_formula, args)
        assert isinstance(cost_estimate, (int, float)), "Cost estimate not a number"
        return cost_estimate

    def prepare_args(self, args: dict):
        unrecognized_args = set(args.keys()) - set(self.base_model.model_fields.keys())
        if unrecognized_args:
            raise ValueError(f"Unrecognized arguments provided: {', '.join(unrecognized_args)}")

        prepared_args = {}
        for field, field_info in self.base_model.model_fields.items():
            if field in args:
                prepared_args[field] = args[field]
            elif field_info.default:
                if field_info.default == "random":
                    minimum, maximum = field_info.metadata[0].ge, field_info.metadata[1].le
                    prepared_args[field] = random.randint(minimum, maximum)
                else:
                    prepared_args[field] = field_info.default
            else:
                prepared_args[field] = None
        
        try:
            self.base_model(**prepared_args)
        except ValidationError as e:
            error_str = eden_utils.get_human_readable_error(e.errors())
            raise ValueError(error_str)

        return prepared_args

    def prepare_result(self, result, env: str):
        if isinstance(result, list):
            return [self.prepare_result(r, env=env) for r in result]
        if "filename" in result:
            filename = result.pop("filename")
            result["url"] = f"{s3.get_root_url(env=env)}/{filename}"
        if "model" in result:
            result["model"] = str(result["model"])
            result.pop("metadata")  # don't need to return model metadata here
        if "intermediate_outputs" in result:
            result["intermediate_outputs"] = {
                k: self.prepare_result(v, env=env)
                for k, v in result["intermediate_outputs"].items()
            }
        return result

    def handle_run(run_function):
        async def wrapper(self, args: Dict, env: str):
            args = self.prepare_args(args)
            result = await run_function(self, args, env)
            return self.prepare_result(result, env)
        return wrapper




    
    # def handle_start_task(submit_function):
    #     async def wrapper(self, task: Task):
    #         user = User.from_id(task.user, env=env)
    #         task.args = self.prepare_args(task.args)
    #         task.cost = self.calculate_cost(task.args.copy())
    #         user.verify_manna_balance(task.cost)
    #         task.status = "pending"
    #         task.save()
    #         handler_id = await submit_function(self, task, *args, **kwargs)
    #         task.update({"handler_id": handler_id})
    #         user.spend_manna(task.cost)
    #         return handler_id
    #     return wrapper
    
    # do we need this?
    def handle_start_task(start_task_function):
        async def wrapper(self, user_id: str, args: Dict, env: str):
            args = self.prepare_args(args)
            cost = self.calculate_cost(args.copy())
            user = User.load(user_id, env=env)
            user.verify_manna_balance(cost)            
            task = Task(
                env=env, 
                workflow=self.key, 
                output_type=self.output_type, 
                args=args, 
                user=user_id, 
                cost=cost
            )
            task.save()            
            handler_id = await start_task_function(self, args, env, user_id)
            task.update(handler_id=handler_id)
            user.spend_manna(task.cost)            
            return task
        return wrapper
    

    def handle_wait(wait_function):
        async def wrapper(self, task: Task):
            if not task.handler_id:
                task.reload()
            result = await wait_function(self, task)
            return self.prepare_result(result, task.env)
        return wrapper
    
    def handle_cancel(cancel_function):
        async def wrapper(self, task: Task):
            await cancel_function(self, task)
            n_samples = task.args.get("n_samples", 1)
            refund_amount = (task.cost or 0) * (n_samples - len(task.result or [])) / n_samples
            user = User.from_id(task.user, env=task.env)
            user.refund_manna(refund_amount)
            task.update(status="cancelled")
        return wrapper

    async def async_run_task_and_wait(self, task: Task):
        await self.async_start_task(task, webhook=False)
        result = await self.async_wait(task)
        return result

    @property
    def async_run(self):
        raise NotImplementedError("Subclasses must implement async_run")

    @property 
    def async_start_task(self):
        raise NotImplementedError("Subclasses must implement async_start_task")

    @property
    def async_wait(self):
        raise NotImplementedError("Subclasses must implement async_wait")

    @property
    def async_cancel(self):
        raise NotImplementedError("Subclasses must implement async_cancel")

    def run(self, args: Dict, env: str):
        return asyncio.run(self.async_run(args, env))

    def start_task(self, task: Task):
        return asyncio.run(self.async_start_task(task))

    def wait(self, task: Task):
        return asyncio.run(self.async_start_task_and_wait(task))

    def start_task_and_wait(self, task: Task):
        return asyncio.run(self.async_start_task_and_wait(task))
    
    def cancel(self, task: Task):
        return asyncio.run(self.async_cancel(task))


def get_tools(path: str, include_inactive: bool = False) -> Dict[str, Tool]:
    """Get all tools inside a directory"""
    
    tools = {}
    
    for root, _, files in os.walk(path):
        if "api.yaml" in files and "test.json" in files:
            rel_path = os.path.relpath(root, path)
            tool_name = rel_path.replace(os.path.sep, "/")  # Normalize path separator
            tool = load_tool(root)
            if tool.status != "inactive" and not include_inactive:
                tool_key = tool_name.split("/")[-1]
                if tool_key in tools:
                    raise ValueError(f"Duplicate tool {tool_key} found.")
                tools[tool_key] = tool
            
    return tools


def load_tool(tool_dir: str, **kwargs) -> Tool:
    """Load the tool class based on the handler in api.yaml"""
    
    from comfyui_tool import ComfyUITool
    from replicate_tool import ReplicateTool
    from modal_tool import ModalTool
    
    api_file = os.path.join(tool_dir, 'api.yaml')
    with open(api_file, 'r') as f:
        schema = yaml.safe_load(f)
    
    handler = schema.get('handler')
    handler_map = {
        "comfyui": ComfyUITool,
        "replicate": ReplicateTool,
        "modal": ModalTool,
        None: ModalTool
    }
    
    tool_class = handler_map.get(handler, Tool)
    
    return tool_class.from_dir(tool_dir, **kwargs)
