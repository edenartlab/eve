import os
import re
import yaml
import json
import random
import asyncio
import traceback
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model, ValidationError
from typing import Optional, List, Dict, Any, Type, Literal
from datetime import datetime, timezone
from instructor.function_calls import openai_schema
import sentry_sdk

from eve.agent.agent import Agent
from eve.api.rate_limiter import RateLimiter

from . import eden_utils
from .base import parse_schema
from .user import User
from .task import Task
from .mongo import Document, Collection, get_collection
from sentry_sdk import trace

OUTPUT_TYPES = Literal[
    "boolean", "string", "integer", "float", "array", "image", "video", "audio", "lora"
]

HANDLERS = Literal[
    "local", "modal", "comfyui", "comfyui_legacy", "replicate", "gcp", "fal"
]

BASE_MODELS = Literal[
    "sd15",
    "sdxl",
    "sd3",
    "sd35",
    "flux-dev",
    "flux-schnell",
    "hellomeme",
    "stable-audio-open",
    "inspyrenet-rembg",
    "mochi-preview",
    "runway",
    "mmaudio",
    "librosa",
    "musicgen",
    "kling",
    "svd-xt",
    "wan21",
    "ltxv",
]

# These tools are default agent tools except Eve
BASE_TOOLS = [
    # text-to-image
    "flux_schnell",
    "flux_dev_lora",
    "flux_dev",
    "txt2img",
    # more image generation
    "flux_inpainting",
    "outpaint",
    "remix_flux_schnell",
    "flux_double_character",
    # video
    "runway",
    "kling_pro",
    "hedra",
    "vid2vid_sdxl",
    "video_FX",
    "texture_flow",
    # audio
    "musicgen",
    "elevenlabs",
    "mmaudio",
    "stable_audio",
    "zonos",
    # editing
    "media_editor",
    # search
    "search_agents",
    "search_models",
    "search_collections",
    "add_to_collection",
    # misc
    "news",
    "websearch",
    "weather",
    # inactive
    # "ominicontrol",
    # "flux_redux",
    "reel",
    # "txt2vid",
    # "animate_3d"
    # "kling_pro"
    "openai_image_edit",
    "openai_image_generate",
]

FLUX_LORA_TOOLS = ["flux_dev_lora", "flux_dev", "reel"]

SDXL_LORA_TOOLS = ["txt2img"]


class RateLimit(BaseModel):
    period: int
    count: int
    feature_flag: Optional[str] = None


@Collection("tools3")
class Tool(Document, ABC):
    """
    Base class for all tools.
    """

    key: str
    name: str
    description: str
    tip: Optional[str] = None
    thumbnail: Optional[str] = None

    output_type: OUTPUT_TYPES
    cost_estimate: str
    resolutions: Optional[List[str]] = None
    base_model: Optional[BASE_MODELS] = None

    # status: Optional[Literal["inactive", "stage", "prod"]] = "stage"
    active: Optional[bool] = True
    visible: Optional[bool] = True
    allowlist: Optional[str] = None

    model: Type[BaseModel]
    handler: HANDLERS = "local"
    parent_tool: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    parameter_presets: Optional[Dict[str, Any]] = None
    gpu: Optional[str] = None
    test_args: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, RateLimit]] = None

    @classmethod
    @trace
    def _get_schema(cls, key, from_yaml=False) -> dict:
        """Get schema for a tool, with detailed performance logging."""

        if from_yaml:
            api_files = get_api_files()  # YAML path
            if key not in api_files:
                raise ValueError(f"Tool {key} not found")

            api_file = api_files[key]
            with open(api_file, "r") as f:
                schema = yaml.safe_load(f)

            if schema.get("handler") in ["comfyui", "comfyui_legacy"]:
                schema["workspace"] = schema.get("workspace") or api_file.split("/")[-4]
        else:
            # MongoDB path
            collection = get_collection(cls.collection_name)
            schema = collection.find_one({"key": key})

        return schema

    @classmethod
    @trace
    def get_sub_class(cls, schema, from_yaml=False) -> type:
        """Lazy load tool classes only when needed"""

        handler = schema.get("handler")
        parent_tool = schema.get("parent_tool")

        # cache parent tool handler
        if parent_tool:
            if parent_tool not in _handler_cache:
                collection = get_collection(cls.collection_name)
                parent = collection.find_one({"key": parent_tool}, {"handler": 1})
                _handler_cache[parent_tool] = parent.get("handler") if parent else None
            handler = _handler_cache[parent_tool]

        # Lazy load the tool class if we haven't seen this handler before
        if handler not in _tool_classes:
            if handler == "local":
                from .tools.local_tool import LocalTool

                _tool_classes[handler] = LocalTool

            elif handler == "modal":
                from .tools.modal_tool import ModalTool

                _tool_classes[handler] = ModalTool
            elif handler == "comfyui":
                from .tools.comfyui_tool import ComfyUITool

                _tool_classes[handler] = ComfyUITool
            elif handler == "comfyui_legacy":
                from .tools.comfyui_tool import ComfyUIToolLegacy

                _tool_classes[handler] = ComfyUIToolLegacy
            elif handler == "replicate":
                from .tools.replicate_tool import ReplicateTool

                _tool_classes[handler] = ReplicateTool
            elif handler == "gcp":
                from .tools.gcp_tool import GCPTool

                _tool_classes[handler] = GCPTool
            elif handler == "fal":
                from .tools.fal_tool import FalTool

                _tool_classes[handler] = FalTool
            else:
                from .tools.modal_tool import ModalTool

                _tool_classes[handler] = ModalTool

        return _tool_classes[handler]

    @classmethod
    def convert_from_yaml(cls, schema: dict, file_path: str = None) -> dict:
        """
        Convert the schema into the format expected by the model.
        """

        from_yaml = False

        key = schema.get("key") or schema.get("parent_tool") or file_path.split("/")[-2]
        parent_tool = schema.get("parent_tool")
        if parent_tool:
            parent_schema = cls._get_schema(parent_tool, from_yaml=from_yaml)
            parent_schema["parameter_presets"] = schema.pop("parameters", {})
            if not from_yaml:
                parent_parameters = {
                    p["name"]: {**(p.pop("schema")), **p}
                    for p in parent_schema.pop("parameters", [])
                }
            for k, v in parent_schema["parameter_presets"].items():
                if k in parent_parameters:
                    parent_parameters[k].update(v)
            schema.pop("workspace", None)  # we want the parent workspace
            parent_schema.update(schema)
            parent_schema["parameters"] = parent_parameters
            schema = parent_schema

        schema["key"] = key
        fields, model_config = parse_schema(schema)
        model = create_model(schema["key"], __config__=model_config, **fields)
        model.__doc__ = eden_utils.concat_sentences(
            schema.get("description"), schema.get("tip", "")
        )
        schema["model"] = model

        # Populate parameters from input_schema for saving
        if "input_schema" in schema and "properties" in schema["input_schema"]:
            schema["parameters"] = schema["input_schema"].get("properties")

        # cast any numbers to strings
        if "cost_estimate" in schema:
            schema["cost_estimate"] = str(schema["cost_estimate"])

        if file_path:
            test_file = file_path.replace("api.yaml", "test.json")
            # Check if test file exists before attempting to load
            if os.path.exists(test_file):
                with open(test_file, "r") as f:
                    schema["test_args"] = json.load(f)
            else:
                # Set to None or an empty dict if test file is optional/missing
                schema["test_args"] = None

        return schema

    @classmethod
    def convert_from_mongo(cls, schema) -> dict:
        schema["parameters"] = {
            p["name"]: {**(p.pop("schema")), **p} for p in schema["parameters"]
        }
        fields, model_config = parse_schema(schema)
        model = create_model(schema["key"], __config__=model_config, **fields)
        model.__doc__ = eden_utils.concat_sentences(
            schema.get("description"), schema.get("tip", "")
        )
        schema["model"] = model

        return schema

    @classmethod
    def convert_to_mongo(cls, schema: dict) -> dict:
        parameters = []
        for k, v in schema["parameters"].items():
            v["schema"] = {
                key: v.pop(key) for key in ["type", "items", "anyOf"] if key in v
            }
            parameters.append({"name": k, **v})

        schema["parameters"] = parameters
        schema.pop("model")

        return schema

    def save(self, **kwargs):
        return super().save({"key": self.key}, **kwargs)

    @classmethod
    def from_raw_yaml(cls, schema, from_yaml=False):
        schema = cls.convert_from_yaml(schema)
        sub_cls = cls.get_sub_class(schema, from_yaml=from_yaml)
        return sub_cls.model_validate(schema)

    @classmethod
    def from_yaml(cls, file_path, cache=False):
        if cache:
            if file_path not in _tool_cache:
                _tool_cache[file_path] = super().from_yaml(file_path)
            return _tool_cache[file_path]
        else:
            return super().from_yaml(file_path)

    @classmethod
    def from_mongo(cls, document_id, cache=False):
        if cache:
            if document_id not in _tool_cache:
                _tool_cache[str(document_id)] = super().from_mongo(document_id)
            return _tool_cache[str(document_id)]
        else:
            return super().from_mongo(document_id)

    @classmethod
    @trace
    def load(cls, key, cache=False):
        if cache:
            if key not in _tool_cache:
                _tool_cache[key] = super().load(key=key)
            return _tool_cache[key]
        else:
            return super().load(key=key)

    def _remove_hidden_fields(self, parameters):
        hidden_parameters = [
            k
            for k, v in parameters["properties"].items()
            if self.parameters[k].get("hide_from_agent")
        ]
        for k in hidden_parameters:
            del parameters["properties"][k]
        parameters["required"] = [
            k for k in parameters.get("required", []) if k not in hidden_parameters
        ]

    def anthropic_schema(self, exclude_hidden: bool = False) -> dict[str, Any]:
        schema = openai_schema(self.model).anthropic_schema
        schema["input_schema"].pop("description")  # duplicated
        if exclude_hidden:
            self._remove_hidden_fields(schema["input_schema"])
        return schema

    def openai_schema(self, exclude_hidden: bool = False) -> dict[str, Any]:
        schema = openai_schema(self.model).openai_schema
        if exclude_hidden:
            self._remove_hidden_fields(schema["parameters"])
        schema["description"] = schema["description"][
            :1024
        ]  # OpenAI tool description limit
        return {"type": "function", "function": schema}

    def calculate_cost(self, args):
        if not self.cost_estimate:
            return 0
        cost_formula = self.cost_estimate
        cost_formula = re.sub(
            r"(\w+)\.length", r"len(\1)", cost_formula
        )  # Array length
        cost_formula = re.sub(
            r"(\w+)\s*\?\s*([^:]+)\s*:\s*([^,\s]+)", r"\2 if \1 else \3", cost_formula
        )  # Ternary operator
        cost_estimate = eval(cost_formula, args.copy())
        assert isinstance(
            cost_estimate, (int, float)
        ), f"Cost estimate ({cost_estimate}) not a number (formula: {cost_formula})"
        return cost_estimate

    def prepare_args(self, args: dict):
        unrecognized_args = set(args.keys()) - set(self.model.model_fields.keys())
        if unrecognized_args:
            # raise ValueError(
            print(
                f"Warning: Unrecognized arguments provided for {self.key}: {', '.join(unrecognized_args)}"
            )

        prepared_args = {}
        for field in self.model.model_fields.keys():
            parameter = self.parameters[field]
            if field in args:
                prepared_args[field] = args[field]
            elif parameter.get("default") == "random":
                minimum, maximum = parameter["minimum"], parameter["maximum"]
                prepared_args[field] = random.randint(minimum, maximum)
            elif parameter.get("default") is not None:
                prepared_args[field] = parameter["default"]

        try:
            self.model(**prepared_args)

        except ValidationError as e:
            print(traceback.format_exc())
            error_str = eden_utils.get_human_readable_error(e.errors())
            raise ValueError(error_str)

        return prepared_args

    def handle_run(run_function):
        """Wrapper for calling a tool directly and waiting for the result"""

        async def async_wrapper(self, args: Dict, mock: bool = False):
            try:
                args = self.prepare_args(args)
                sentry_sdk.add_breadcrumb(category="handle_run", data=args)
                if mock:
                    result = {"output": eden_utils.mock_image(args)}
                else:
                    result = await run_function(self, args)
                result["output"] = (
                    result["output"]
                    if isinstance(result["output"], list)
                    else [result["output"]]
                )
                result = eden_utils.upload_result(result)
                result["status"] = "completed"
                sentry_sdk.add_breadcrumb(category="handle_run", data=result)

            except Exception as e:
                print(traceback.format_exc())
                result = {"status": "failed", "error": str(e)}
                sentry_sdk.capture_exception(e)

            return result

        return async_wrapper

    @trace
    def handle_start_task(start_task_function):
        """Wrapper for starting a task process and returning a task"""

        async def async_wrapper(
            self,
            user_id: str,
            agent_id: str,
            args: Dict,
            public: bool = False,
            mock: bool = False,
            is_client_platform: bool = False,
        ):
            try:
                args = self.prepare_args(args)
                sentry_sdk.add_breadcrumb(category="handle_start_task", data=args)
                cost = self.calculate_cost(args)
                user = User.from_mongo(user_id)

                if agent_id:
                    agent = Agent.from_mongo(agent_id)
                    if agent.owner_pays and is_client_platform:
                        paying_user_id = agent.owner
                        paying_user = User.from_mongo(paying_user_id)
                    else:
                        paying_user = user
                else:
                    paying_user = user

                # validate args and user manna balance
                paying_user.check_manna(cost)
            except Exception as e:
                print(traceback.format_exc())
                raise Exception(f"Task submission failed: {str(e)}. No manna deducted.")

            # Check rate limit before creating the task
            if os.environ.get("FF_RATE_LIMITS") == "yes":
                print("checking rate limit", agent_id, is_client_platform)
                rate_limiter = RateLimiter()
                if agent_id and is_client_platform:
                    await rate_limiter.check_agent_rate_limit(user, agent_id)
                else:
                    await rate_limiter.check_manna_spend_rate_limit(user)

            task = Task(
                user=user_id,
                agent=agent_id,
                tool=self.key,
                parent_tool=self.parent_tool,
                output_type=self.output_type,
                args=args,
                mock=mock,
                cost=cost,
                public=public,
                paying_user=paying_user_id,
            )
            task.save()
            sentry_sdk.add_breadcrumb(
                category="handle_start_task", data=task.model_dump()
            )

            # start task
            try:
                if mock:
                    handler_id = eden_utils.random_string()
                    output = [{"output": [eden_utils.mock_image(args)]}]
                    result = eden_utils.upload_result(output)
                    task.update(
                        handler_id=handler_id,
                        status="completed",
                        result=result,
                        performance={
                            "waitTime": (
                                datetime.now(timezone.utc) - task.createdAt
                            ).total_seconds()
                        },
                    )
                else:
                    handler_id = await start_task_function(self, task)
                    task.update(handler_id=handler_id)

                if "free_tools" not in (paying_user.featureFlags or []):
                    task.spend_manna()

            except Exception as e:
                print(traceback.format_exc())
                task.update(status="failed", error=str(e))
                sentry_sdk.capture_exception(e)
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
                print(traceback.format_exc())
                result = {"status": "failed", "error": str(e)}
            return result

        return async_wrapper

    def handle_cancel(cancel_function):
        """Wrapper for cancelling a task"""

        async def async_wrapper(self, task: Task, force: bool = False):
            try:
                await cancel_function(self, task)
            except Exception as e:
                sentry_sdk.capture_exception(f"Error cancelling task: {e}")
                traceback.print_exc()
            finally:
                task.refund_manna()
                if force:
                    # Forced cancellation from the server due to stuck task
                    task.update(status="failed", error="Timed out")
                else:
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

    def run(self, args: Dict, mock: bool = False):
        return asyncio.run(self.async_run(args, mock))

    def start_task(
        self,
        user_id: str,
        agent_id: str,
        args: Dict,
        mock: bool = False,
        public: bool = False,
    ):
        return asyncio.run(self.async_start_task(user_id, agent_id, args, mock, public))

    def wait(self, task: Task):
        return asyncio.run(self.async_wait(task))

    def cancel(self, task: Task, force: bool = False):
        return asyncio.run(self.async_cancel(task, force))


def get_tools_from_api_files(
    root_dir: str = None,
    tools: List[str] = None,
    include_inactive: bool = False,
    cache: bool = False,
) -> Dict[str, Tool]:
    """Get all tools inside a directory"""

    api_files = get_api_files(root_dir)
    tools = {
        key: _tool_cache.get(api_file) or Tool.from_yaml(api_file, cache=cache)
        for key, api_file in api_files.items()
        if tools is None or key in tools
    }

    if not include_inactive:
        tools = {k: v for k, v in tools.items() if v.status != "inactive"}

    return tools


def get_tools_from_mongo(
    tools: List[str] = None,
    include_inactive: bool = False,
    cache: bool = False,
) -> Dict[str, Tool]:
    """Get all tools from mongo"""

    tools_collection = get_collection(Tool.collection_name)

    # Batch fetch all tools and their parents
    filter = {"key": {"$in": tools}} if tools else {}
    tool_docs = list(tools_collection.find(filter))

    found_tools = {}
    for tool in tool_docs:
        try:
            if tool.get("key") in _tool_cache:
                tool = _tool_cache[tool.get("key")]
            else:
                tool = Tool.convert_from_mongo(tool)
                tool = Tool.from_schema(tool, from_yaml=False)
                if cache:
                    _tool_cache[tool.key] = tool
            if tool.active and not include_inactive:
                if tool.key in found_tools:
                    raise ValueError(f"Duplicate tool {tool.key} found.")
                found_tools[tool.key] = tool
        except Exception as e:
            print(traceback.format_exc())

    return found_tools


@trace
def get_api_files(root_dir: str = None) -> List[str]:
    """Get all api.yaml files inside a directory"""

    if root_dir:
        root_dirs = [root_dir]
    else:
        eve_root = os.path.dirname(os.path.abspath(__file__))
        root_dirs = [
            os.path.join(eve_root, tools_dir)
            for tools_dir in ["tools", "../../workflows", "../../private_workflows"]
        ]

    api_files = {}
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            if "api.yaml" in files and "test.json" in files:
                api_file = os.path.join(root, "api.yaml")
                key = os.path.relpath(root).split("/")[-1]
                if "legacy" in root:
                    key = f"legacy_{key}"
                api_files[key] = api_file

    return api_files


def tool_context(tool_type):
    def decorator(cls):
        for name, method in cls.__dict__.items():
            if asyncio.iscoroutinefunction(method):

                async def wrapped_method(self, *args, __method=method, **kwargs):
                    with sentry_sdk.configure_scope() as scope:
                        scope.set_tag("package", "eve-tools")
                        scope.set_tag("tool_type", tool_type)
                        scope.set_tag("tool_name", self.key)
                    return await __method(self, *args, **kwargs)

                setattr(cls, name, wrapped_method)
        return cls

    return decorator


# Tool cache for fetching commonly used tools
_tool_cache: Dict[str, Dict[str, Tool]] = {}

# Move cache to module level
_handler_cache = {}

# Cache for lazy loading tool classes
_tool_classes = {}
