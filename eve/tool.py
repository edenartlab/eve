import os
import re
import yaml
import json
import random
import asyncio
from pydantic import BaseModel, create_model, ValidationError
from typing import Optional, List, Dict, Any, Type, Literal
from datetime import datetime

from sentry_sdk import add_breadcrumb, capture_exception, capture_message
import sentry_sdk
sentry_dsn = os.getenv("SENTRY_DSN")
sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, profiles_sample_rate=1.0)


from . import eden_utils
from .base import parse_schema
from .models import User
from .task import Task
from .mongo import MongoModel, get_collection

# from .tools.comfyui_tool import ComfyUITool
# from .tools.replicate_tool import ReplicateTool
# from .tools.modal_tool import ModalTool
# from .tools.gcp_tool import GCPTool
# from .tools.local_tool import LocalTool

from functools import wraps

# class Tool(MongoModel):
from abc import ABC, abstractmethod

class Tool(BaseModel, ABC):
    """
    Base class for all tools.
    """

    key: str
    name: str
    description: str
    tip: Optional[str] = None
    
    output_type: Literal["bool", "str", "int", "float", "string", "image", "video", "audio", "lora"]
    cost_estimate: str
    resolutions: Optional[List[str]] = None
    base_model: Literal["sd15", "sdxl", "sd3", "flux-dev", "flux-schnell"] = "sdxl"
    
    status: Optional[Literal["inactive", "stage", "prod"]] = "stage"
    visible: Optional[bool] = True
    allowlist: Optional[str] = None
    
    model: Type[BaseModel] = None
    handler: Literal["local", "modal", "comfyui", "replicate", "gcp"] = "local"
    parent_tool: Optional[str] = None
    parameter_presets: Optional[Dict[str, Any]] = None
    gpu: Optional[str] = None    
    test_args: Dict[str, Any]


    @classmethod
    def load(cls, key: str, env: str, prefer_local: bool = True, **kwargs):
        """Load the tool class based on the handler in api.yaml"""
        
        tools = get_collection("tools2", env=env)
        schema = tools.find_one({"key": key})
        
        if not schema:
            raise ValueError(f"Tool with key {key} not found on env: {env}")

        return cls.load_from_schema(schema, prefer_local, **kwargs)

    @classmethod
    def load_from_dir(cls, tool_dir: str, prefer_local: bool = True, **kwargs):
        """Load the tool from an api.yaml and test.json"""
        
        schema = cls._get_schema_from_dir(tool_dir)
        schema['key'] = tool_dir.split('/')[-1]
        
        return cls.load_from_schema(schema, prefer_local, **kwargs)

    @classmethod
    def load_from_schema(cls, schema: dict, prefer_local: bool = True, **kwargs):
        """Load the tool class based on the handler in api.yaml"""
        
        key = schema.pop('key')
        test_args = schema.pop('test_args')
        tool_class = _get_tool_class(schema.get('handler'), prefer_local)
        
        return tool_class._create_tool(key, schema, test_args, **kwargs)    

    @classmethod
    def _create_tool(cls, key: str, schema: dict, test_args: dict, **kwargs):
        """Create a new tool instance from a schema"""

        fields = parse_schema(schema)
        model = create_model(key, **fields)
        model.__doc__ = eden_utils.concat_sentences(schema.get('description'), schema.get('tip', ''))

        tool_data = {k: schema.pop(k) for k in cls.model_fields.keys() if k in schema}
        tool_data['test_args'] = test_args
        tool_data['model'] = model
        if 'cost_estimate' in tool_data:
            tool_data['cost_estimate'] = str(tool_data['cost_estimate'])

        return cls(key=key, **tool_data, **kwargs)

    @classmethod
    def _get_schema_from_dir(cls, tool_dir: str):
        if not os.path.exists(tool_dir):
            raise ValueError(f"Tool directory {tool_dir} does not exist")

        api_file = os.path.join(tool_dir, 'api.yaml')
        test_file = os.path.join(tool_dir, 'test.json')

        with open(api_file, 'r') as f:
            schema = yaml.safe_load(f)
                
        with open(test_file, 'r') as f:
            schema['test_args'] = json.load(f)

        if schema.get("handler") == "comfyui":
            schema["workspace"] = tool_dir.split('/')[-3]

        parent_tool = schema.get("parent_tool")
        
        if parent_tool:
            tool_dirs = get_tool_dirs()
            if schema["parent_tool"] not in tool_dirs:
                raise ValueError(f"Parent tool {schema['parent_tool']} not found in tool_dirs")            
            parent_dir = tool_dirs[schema["parent_tool"]]
            parent_api_file = os.path.join(parent_dir, 'api.yaml')
            with open(parent_api_file, 'r') as f:
                parent_schema = yaml.safe_load(f)

            if parent_schema.get("handler") == "comfyui":
                parent_schema["workspace"] = parent_dir.split('/')[-3]

            parent_schema["parameter_presets"] = schema.pop("parameters", {})
            parent_parameters = parent_schema.pop("parameters", {})
            for k, v in parent_schema["parameter_presets"].items():
                parent_parameters[k].update(v)
            
            parent_schema.update(schema)
            parent_schema['parameters'] = parent_parameters
            schema = parent_schema
        
        return schema
    
    def _get_model_schema(self, exclude_hidden: bool = False, include_tips: bool = False):
        # TODO: include tips
        ############################################################

        schema = self.model.model_json_schema()
        if exclude_hidden:
            schema['properties'] = {
                k: v for k, v in schema['properties'].items() if not v.get('hide_from_agent')
            }
            schema['required'] = [
                k for k, v in schema['properties'].items() if k in schema['required'] and not v.get('hide_from_agent')
            ]
        return schema

    def anthropic_schema(self, exclude_hidden: bool = False, include_tips: bool = False) -> dict[str, Any]:
        schema = self._get_model_schema(exclude_hidden, include_tips)
        name = schema.pop('title')
        description = schema.pop('description')        
        return {
            "name": name,
            "description": description,
            "input_schema": schema
        }

    def openai_schema(self, exclude_hidden: bool = False, include_tips: bool = False) -> dict[str, Any]:
        schema = self._get_model_schema(exclude_hidden, include_tips)
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )
        return {
            "type": "function",
            "function": {
                "name": schema["title"],
                "description": schema["description"],
                "parameters": parameters,
            }
        }

    def calculate_cost(self, args):
        if not self.cost_estimate:
            return 0
        cost_formula = self.cost_estimate
        cost_formula = re.sub(r'(\w+)\.length', r'len(\1)', cost_formula)  # Array length
        cost_formula = re.sub(r'(\w+)\s*\?\s*([^:]+)\s*:\s*([^,\s]+)', r'\2 if \1 else \3', cost_formula)  # Ternary operator
        
        cost_estimate = eval(cost_formula, args.copy())
        assert isinstance(cost_estimate, (int, float)), "Cost estimate not a number"
        return cost_estimate

    def prepare_args(self, args: dict):
        unrecognized_args = set(args.keys()) - set(self.model.model_fields.keys())
        if unrecognized_args:
            raise ValueError(f"Unrecognized arguments provided for {self.key}: {', '.join(unrecognized_args)}")

        prepared_args = {}
        for field, field_info in self.model.model_fields.items():
            if field in args:
                prepared_args[field] = args[field]
            elif field_info.default is not None:
                if field_info.default == "random":
                    minimum, maximum = field_info.metadata[0].ge, field_info.metadata[1].le
                    prepared_args[field] = random.randint(minimum, maximum)
                else:
                    prepared_args[field] = field_info.default
            else:
                prepared_args[field] = None
        
        try:
            self.model(**prepared_args)
        except ValidationError as e:
            error_str = eden_utils.get_human_readable_error(e.errors())
            raise ValueError(error_str)

        return prepared_args

    def handle_run(run_function):
        """Wrapper for calling a tool directly and waiting for the result"""
        
        async def async_wrapper(self, args: Dict, env: str, mock: bool = False):
            try:
                args = self.prepare_args(args)
                add_breadcrumb(category="handle_run", data=args)
                if mock:
                    result = {"output": eden_utils.mock_image(args)}
                else:
                    result = await run_function(self, args, env)
                add_breadcrumb(category="handle_run", data=result)
                result = eden_utils.upload_result(result, env)
                add_breadcrumb(category="handle_run", data=result)
                result["status"] = "completed"
            except Exception as e:
                result = {"status": "failed", "error": str(e)}
                capture_exception(e)
            return eden_utils.prepare_result(result, env)
        
        return async_wrapper

    def handle_start_task(start_task_function):
        """Wrapper for starting a task process and returning a task"""

        async def async_wrapper(self, user_id: str, args: Dict, env: str, mock: bool = False):
            try:
                # validate args and user manna balance
                args = self.prepare_args(args)
                add_breadcrumb(category="handle_start_task", data=args)
                cost = self.calculate_cost(args)
                user = User.load(user_id, env=env)
                user.verify_manna_balance(cost)       
                
            except Exception as e:
                raise Exception(f"Task submission failed: {str(e)}. No manna deducted.")

            # create task and set to pending
            task = Task(
                env=env, 
                workflow=self.key, 
                parent_tool=self.parent_tool,
                output_type=self.output_type, 
                args=args, 
                user=user_id, 
                cost=cost,
                mock=mock
            )
            task.save()

            add_breadcrumb(category="handle_start_task", data=task.model_dump())

            # start task
            try:
                if mock:
                    handler_id = eden_utils.random_string()
                    output = {"output": eden_utils.mock_image(args)}
                    result = eden_utils.upload_result(output, env=env)
                    task.update(
                        handler_id=handler_id,
                        status="completed", 
                        result=result,
                        performance={"waitTime": (datetime.utcnow() - task.createdAt).total_seconds()}
                    )
                else:
                    handler_id = await start_task_function(self, task)
                    task.update(handler_id=handler_id)

                user.spend_manna(task.cost)            

            except Exception as e:
                task.update(status="failed", error=str(e))
                capture_exception(e)
                raise Exception(f"Task failed: {e}. No manna deducted.")
            
            return task

        return async_wrapper

    def handle_wait(wait_function):
        """Wrapper for waiting for a task to complete"""

        async def async_wrapper(self, task: Task):
            if not task.handler_id:
                task.reload()
            try:
                if task.mock:
                    result = task.result
                else:
                    result = await wait_function(self, task)
            except Exception as e:
                result = {"status": "failed", "error": str(e)}
            return eden_utils.prepare_result(result, task.env)
        
        return async_wrapper

    def handle_cancel(cancel_function):
        """Wrapper for cancelling a task"""

        async def async_wrapper(self, task: Task):
            await cancel_function(self, task)
            n_samples = task.args.get("n_samples", 1)
            refund_amount = (task.cost or 0) * (n_samples - len(task.result or [])) / n_samples
            user = User.from_id(task.user, env=task.env)
            user.refund_manna(refund_amount)
            task.update(status="cancelled")
        
        return async_wrapper

    @abstractmethod
    async def async_run(self):
        pass

    @abstractmethod
    async def async_start_task(self):
        pass

    @abstractmethod
    async def async_wait(self):
        pass

    @abstractmethod
    async def async_cancel(self):
        pass


def save_tool_from_dir(tool_dir: str, env: str) -> Tool:
    """Upload tool from directory to mongo"""

    schema = Tool._get_schema_from_dir(tool_dir)
    schema['key'] = tool_dir.split('/')[-1]
    
    tools = get_collection("tools2", env=env)
    tool = tools.find_one({"key": schema['key']})
    
    time = datetime.utcnow()
    schema['createdAt'] = tool.get('createdAt', time) if tool else time
    schema['updatedAt'] = time
    
    tools.replace_one({"key": schema['key']}, schema, upsert=True)


def get_tools_from_dirs(root_dir: str = None, include_inactive: bool = False) -> Dict[str, Tool]:
    """Get all tools inside a directory"""
    
    tool_dirs = get_tool_dirs(root_dir, include_inactive)
    tools = {
        key: Tool.load_from_dir(tool_dir) 
        for key, tool_dir in tool_dirs.items()
    }

    return tools


def get_tools_from_mongo(env: str, include_inactive: bool = False, prefer_local: bool = True) -> Dict[str, Tool]:
    """Get all tools from mongo"""
    
    tools = {}
    tools_collection = get_collection("tools2", env=env)
    for tool in tools_collection.find():
        tool = Tool.load_from_schema(tool, prefer_local)
        if tool.status != "inactive" and not include_inactive:
            if tool.key in tools:
                raise ValueError(f"Duplicate tool {tool.key} found.")
            tools[tool.key] = tool
        
    return tools


def get_tool_dirs(root_dir: str = None, include_inactive: bool = False) -> List[str]:
    """Get all tool directories inside a directory"""
    
    if root_dir:
        root_dirs = [root_dir]
    else:
        eve_root = os.path.dirname(os.path.abspath(__file__))
        root_dirs = [
            os.path.join(eve_root, tools_dir) 
            for tools_dir in ["tools", "../../workflows"]
        ]

    tool_dirs = {}

    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            if "api.yaml" in files and "test.json" in files:
                api_file = os.path.join(root, "api.yaml")
                with open(api_file, 'r') as f:
                    schema = yaml.safe_load(f)
                if schema.get("status") == "inactive" and not include_inactive:
                    continue
                key = os.path.relpath(root).split("/")[-1]
                if key in tool_dirs:
                    raise ValueError(f"Duplicate tool {key} found.")
                tool_dirs[key] = os.path.relpath(root)
            
    return tool_dirs


def _get_tool_class(handler: str, prefer_local: bool = True):
    from .tools.local_tool import LocalTool
    from .tools.modal_tool import ModalTool
    from .tools.comfyui_tool import ComfyUITool
    from .tools.replicate_tool import ReplicateTool
    from .tools.gcp_tool import GCPTool

    handler_map = {
        "local": LocalTool,
        "modal": ModalTool,
        "comfyui": ComfyUITool,
        "replicate": ReplicateTool,
        "gcp": GCPTool,
        None: LocalTool if prefer_local else ModalTool
    }
    
    tool_class = handler_map.get(handler, Tool)
    return tool_class
    
