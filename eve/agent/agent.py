import os
import time
import json
import traceback
from elevenlabs import Model
import openai
import instructor
import sentry_sdk
from datetime import timezone
from pathlib import Path
from bson import ObjectId
from typing import Optional, Literal, Any, Dict, List
from datetime import datetime
from dotenv import dotenv_values
from pydantic import SecretStr, Field, BaseModel, ConfigDict
from functools import wraps
# from pydantic.json_schema import SkipJsonSchema

from ..tool_constants import (
    TWITTER_TOOLS,
    DISCORD_TOOLS,
    FARCASTER_TOOLS,
    TELEGRAM_TOOLS,
    SOCIAL_MEDIA_TOOLS,
    TOOL_SETS,
)
from ..mongo import Collection, get_collection
from ..models import Model
from ..user import User, Manna
from ..eden_utils import load_template
from .thread import Thread

agent_updated_at = {}


def profile_method(method_name):
    """Decorator to profile method execution time"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"PERF_PROFILE: {method_name} took {elapsed:.3f}s")
            return result

        return wrapper

    return decorator


class KnowledgeDescription(BaseModel):
    """Defines when and why a reference document should be consulted to enhance responses."""

    summary: str = Field(
        ...,
        description="A precise, content-focused summary of the document, detailing what information it contains without unnecessary adjectives or filler words.",
    )
    retrieval_criteria: str = Field(
        ...,
        description="A clear, specific description of when the reference document is needed to answer a user query. This should specify what topics, types of questions, or gaps in the assistant's knowledge require consulting the document.",
    )


class Suggestion(BaseModel):
    """A prompt suggestion for an Agent in two parts: a concise tagline, and a longer prompt for an LLM. The prompt should be appropriate for the agent, but not exaggerated."""

    label: Optional[str] = Field(
        ...,
        description="A short and catchy tagline, no more than 7 words, to go into a home page button. Shorten, omit stop words (the, a, an, etc) when possible.",
    )
    prompt: Optional[str] = Field(
        ...,
        description="A longer version of the tagline, a prompt to be sent to the agent following its greeting. The prompt should be no more than one sentence or 30 words.",
    )


class AgentPermissions(BaseModel):
    """Permissions configuration for an agent."""

    editors: Optional[List[ObjectId]] = Field(
        None,
        description="List of user IDs who can edit this agent (in addition to the owner)",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentExtras(BaseModel):
    """Additional configuration and metadata for an agent."""

    permissions: Optional[AgentPermissions] = Field(
        None,
        description="Permissions configuration for the agent",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("users3")
class Agent(User):
    """
    Base class for all agents.
    """

    type: Literal["agent"] = "agent"
    owner: ObjectId
    secrets: Optional[Dict[str, SecretStr]] = Field(None, exclude=True)

    # status: Optional[Literal["inactive", "stage", "prod"]] = "stage"
    public: Optional[bool] = False
    allowlist: Optional[List[str]] = None
    featured: Optional[bool] = False

    name: str
    description: Optional[str] = None
    suggestions: Optional[List[Suggestion]] = None
    greeting: Optional[str] = None
    persona: Optional[str] = None
    knowledge: Optional[str] = None
    knowledge_summary: Optional[str] = None
    knowledge_description: Optional[KnowledgeDescription] = None
    voice: Optional[str] = None
    refreshed_at: Optional[datetime] = None

    mute: Optional[bool] = False
    reply_criteria: Optional[str] = None
    model: Optional[ObjectId] = None  # deprecated
    models: Optional[List[Dict[str, Any]]] = None
    test_args: Optional[List[Dict[str, Any]]] = None

    tools: Optional[Dict[str, bool]] = {}  # tool sets specified by user
    tools_: Optional[Dict[str, Dict]] = Field({}, exclude=True)  # actual loaded tools
    lora_docs: Optional[List[Dict[str, Any]]] = Field([], exclude=True)
    deployments: Optional[List[str]] = Field({}, exclude=True)

    owner_pays: Optional[bool] = False
    agent_extras: Optional[AgentExtras] = None

    def __init__(self, **data):
        if isinstance(data.get("owner"), str):
            data["owner"] = ObjectId(data["owner"])
        if isinstance(data.get("owner"), str):
            data["model"] = ObjectId(data["model"])
        # Load environment variables into secrets dictionary
        db = os.getenv("DB")
        env_dir = Path(__file__).parent / "agents"
        env_vars = dotenv_values(f"{str(env_dir)}/{db.lower()}/{data['username']}/.env")
        data["secrets"] = {key: SecretStr(value) for key, value in env_vars.items()}
        super().__init__(**data)

    @classmethod
    def convert_from_yaml(cls, schema: dict, file_path: str = None) -> dict:
        """
        Convert the schema into the format expected by the model.
        """
        test_file = file_path.replace("api.yaml", "test.json")
        with open(test_file, "r") as f:
            schema["test_args"] = json.load(f)

        schema["owner"] = (
            ObjectId(schema.get("owner"))
            if isinstance(schema.get("owner"), str)
            else schema.get("owner")
        )
        schema["model"] = (
            ObjectId(schema.get("model"))
            if isinstance(schema.get("model"), str)
            else schema.get("model")
        )  # deprecated
        for model in schema.get("models", []):
            model["lora"] = (
                ObjectId(model["lora"])
                if isinstance(model["lora"], str)
                else model["lora"]
            )
        schema["username"] = schema.get("username") or file_path.split("/")[-2]

        return schema

    def save(self, **kwargs):
        # do not overwrite any username if it already exists
        users = get_collection(User.collection_name)
        if users.find_one({"username": self.username, "type": "user"}):
            raise ValueError(f"Username {self.username} already taken")

        # save user, and create mannas record if it doesn't exist
        super().save(
            upsert_filter={"username": self.username, "type": "agent"}, **kwargs
        )
        Manna.load(user=self.id)  # create manna record if it doesn't exist

    @classmethod
    def from_yaml(cls, file_path, cache=False):
        return super().from_yaml(file_path)

    @classmethod
    def from_mongo(cls, document_id, cache=False):
        return super().from_mongo(document_id)

    @classmethod
    def load(cls, username, cache=False):
        return super().load(username=username)

    def request_thread(self, key=None, user=None, message_limit=25):
        thread = Thread(key=key, agent=self.id, user=user, message_limit=message_limit)
        thread.save()
        return thread

    @profile_method("_reload (original)")
    def _reload(self):
        from ..tool import Tool
        from ..agent.session.models import Deployment

        # load deployments to memory
        self.deployments = {
            deployment.platform.value: deployment
            for deployment in Deployment.find({"agent": ObjectId(str(self.id))})
        }

        # load loras to memory
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in self.models or []}
        lora_docs = (
            models_collection.find(
                {"_id": {"$in": list(loras_dict.keys())}, "deleted": {"$ne": True}}
            )
            if self.models
            else []
        )
        lora_docs = list(lora_docs or [])
        self.lora_docs = lora_docs

        # load tools to memory
        tools = {}
        for tool_set, set_tools in TOOL_SETS.items():
            if not self.tools.get(tool_set):
                continue

            for t in set_tools:
                try:
                    tool = Tool.from_raw_yaml({"parent_tool": t})
                    tools[t] = tool
                except Exception as e:
                    print(f"Error loading tool {t}: {e}")
                    print(traceback.format_exc())

                    # Graceful failure with Sentry tracking
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("component", "tool_loading")
                        scope.set_tag("agent_username", self.username)
                        scope.set_tag("agent_id", str(self.id))
                        scope.set_tag("tool_name", t)
                        scope.set_tag("tool_set", tool_set)
                        scope.set_context(
                            "tool_loading_context",
                            {
                                "tool_name": t,
                                "tool_set": tool_set,
                                "agent_username": self.username,
                                "agent_id": str(self.id),
                                "error_message": str(e),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        sentry_sdk.capture_exception(e)

                    # Continue loading other tools instead of crashing
                    continue

        self.tools_ = tools

    @profile_method("_reload_optimized")
    def _reload_optimized(self):
        """Optimized version of _reload that batches MongoDB queries"""
        from ..tool import Tool
        from ..agent.session.models import Deployment

        # load deployments to memory
        self.deployments = {
            deployment.platform.value: deployment
            for deployment in Deployment.find({"agent": ObjectId(str(self.id))})
        }

        # load loras to memory - batch query
        start_time = time.time()
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in self.models or []}

        if self.models:
            lora_ids = list(loras_dict.keys())
            print(f"PERF_PROFILE: batch loading {len(lora_ids)} parent schemas...")

            # Single batch query for all loras
            lora_docs = list(
                models_collection.find(
                    {"_id": {"$in": lora_ids}, "deleted": {"$ne": True}}
                )
            )
            self.lora_docs = lora_docs
            print(
                f"PERF_PROFILE: batch loaded {len(lora_docs)} parent schemas in {time.time() - start_time:.3f}s"
            )
        else:
            self.lora_docs = []

        # Collect all tools needed
        needed_tools = []
        for tool_set, set_tools in TOOL_SETS.items():
            if self.tools.get(tool_set):
                needed_tools.extend(set_tools)

        # Batch load ALL tool schemas from MongoDB
        if needed_tools:
            print(f"PERF_PROFILE: batch loading {len(needed_tools)} tool schemas...")
            start_time = time.time()

            # Single batch query to MongoDB
            tools_collection = get_collection(Tool.collection_name)
            batch_schemas = list(
                tools_collection.find({"key": {"$in": needed_tools}})
            )

            # Create a dict for easy lookup
            schemas_by_key = {schema.get("key"): schema for schema in batch_schemas}

            print(
                f"PERF_PROFILE: batch loaded {len(batch_schemas)} schemas in {time.time() - start_time:.3f}s"
            )

            # Convert ALL schemas to tools (no caching, fresh every time)
            tools = {}
            for t in needed_tools:
                try:
                    start_time = time.time()

                    if t in schemas_by_key:
                        schema = schemas_by_key[t]

                        # Check if parameters are in list format (older MongoDB format) or dict format (newer)
                        parameters = schema.get("parameters", {})
                        if isinstance(parameters, list):
                            # Use convert_from_mongo for list format
                            print(
                                f"PERF_PROFILE: using convert_from_mongo for {t} (list format)"
                            )
                            schema = Tool.convert_from_mongo(schema)
                        else:
                            # Parameters are already in dict format, use them directly
                            print(
                                f"PERF_PROFILE: using schema directly for {t} (dict format)"
                            )
                            schema = schema.copy()
                            # Create the model for this schema
                            from ..base import parse_schema
                            from pydantic import create_model
                            from .. import eden_utils

                            fields, model_config = parse_schema(schema)
                            model = create_model(
                                schema["key"], __config__=model_config, **fields
                            )
                            model.__doc__ = eden_utils.concat_sentences(
                                schema.get("description"), schema.get("tip", "")
                            )
                            schema["model"] = model
                    else:
                        # Fallback to original YAML-based method if not in DB
                        print(f"PERF_PROFILE: falling back to YAML method for {t}")
                        schema = Tool.convert_from_yaml({"parent_tool": t})

                    sub_cls = Tool.get_sub_class(schema, from_yaml=False)
                    tool = sub_cls.model_validate(schema)
                    tools[t] = tool

                    print(
                        f"PERF_PROFILE: tool creation took {time.time() - start_time:.3f}s for {t}"
                    )

                except Exception as e:
                    print(f"Error loading tool {t}: {e}")
                    print(traceback.format_exc())

                    # Graceful failure with Sentry tracking
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("component", "tool_loading_optimized")
                        scope.set_tag("agent_username", self.username)
                        scope.set_tag("agent_id", str(self.id))
                        scope.set_tag("tool_name", t)
                        scope.set_context(
                            "tool_loading_context",
                            {
                                "tool_name": t,
                                "agent_username": self.username,
                                "agent_id": str(self.id),
                                "error_message": str(e),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        sentry_sdk.capture_exception(e)

                    continue

            self.tools_ = tools
        else:
            self.tools_ = {}

    @profile_method("get_tools (original)")
    def get_tools(self, cache=True, auth_user: str = None):
        """
        Cache is disabled until bug is fixed.
        Problem is agent gets into agent_updated_at, but then loaded as new object later, so tools are not populated.
        """
        if cache:
            if self.username in agent_updated_at:
                # outdated, reload tools
                if self.updatedAt > agent_updated_at[self.username]["updatedAt"]:
                    self._reload()
                    agent_updated_at[self.username] = {
                        "updatedAt": self.updatedAt,
                        "tools": self.tools_,
                    }

                # grab cached tools
                else:
                    self.tools_ = agent_updated_at[self.username]["tools"]
            else:
                # first time, load tools, set cache
                self._reload()
                agent_updated_at[self.username] = {
                    "updatedAt": self.updatedAt,
                    "tools": self.tools_,
                }
        else:
            self._reload()
            agent_updated_at[self.username] = {
                "updatedAt": self.updatedAt,
                "tools": self.tools_,
            }

        # self._reload()
        tools = self.tools_

        # update tools with platform-specific args
        # update discord post tool with allowed channels
        try:
            if "discord" in self.deployments:
                if "discord_post" in tools:
                    allowed_channels = self.deployments[
                        "discord"
                    ].get_allowed_channels()
                    channels_description = " | ".join(
                        [f"ID {c.id} ({c.note})" for c in allowed_channels]
                    )
                    tools["discord_post"].update_parameters(
                        {
                            "channel_id": {
                                "choices": [c.id for c in allowed_channels],
                                "tip": f"Some hints about the available channels: {channels_description}",
                            },
                        }
                    )
        except Exception as e:
            _log_tool_operation_error(
                self, "discord_channel_update", e, platform="discord"
            )

        # update telegram post tool with allowed channels
        try:
            if "telegram" in self.deployments:
                if "telegram_post" in tools:
                    allowed_channels = self.deployments[
                        "telegram"
                    ].get_allowed_channels()
                    channels_description = " | ".join(
                        [f"ID {c.id} ({c.note})" for c in allowed_channels]
                    )
                    tools["telegram_post"].update_parameters(
                        {
                            "channel_id": {
                                "choices": [c.id for c in allowed_channels],
                                "tip": f"Some hints about the available topics: {channels_description}",
                            },
                        }
                    )
        except Exception as e:
            _log_tool_operation_error(
                self, "telegram_channel_update", e, platform="telegram"
            )

        # if a platform is not deployed, remove all tools for that platform
        if "discord" not in self.deployments:
            for tool in DISCORD_TOOLS:
                tools.pop(tool, None)
        if "telegram" not in self.deployments:
            for tool in TELEGRAM_TOOLS:
                tools.pop(tool, None)
        if "twitter" not in self.deployments:
            for tool in TWITTER_TOOLS:
                tools.pop(tool, None)
        if "farcaster" not in self.deployments:
            for tool in FARCASTER_TOOLS:
                tools.pop(tool, None)

        # remove tools that only the owner can use
        if str(auth_user) != str(self.owner):
            for tool in SOCIAL_MEDIA_TOOLS:
                tools.pop(tool, None)

        # if models are found, inject them as defaults for any tools that use lora
        for tool in tools:
            try:
                if "lora" not in tools[tool].parameters:
                    continue

                lora_docs = self.lora_docs

                if not lora_docs:
                    continue

                # Build LoRA information for the tip
                lora_info = []
                for m in lora_docs:
                    lora_id = m["_id"]
                    lora_doc = {m["lora"]: m for m in self.models}[lora_id]
                    lora_info.append(
                        f"{{ ID: {lora_id}, Name: {m['name']}, Description: {m['lora_trigger_text']}, Use When: {lora_doc['use_when']} }}"
                    )

                params = {
                    "lora": {
                        "default": str(lora_docs[0]["_id"]),
                        "tip": "Users may request one of your known LoRAs, or a different unknown one, or no LoRA at all. When referring to a LoRA, strictly use its name, not its description. Notes on when to use the known LoRAs: "
                        + " | ".join(lora_info),
                    },
                }
                if "use_lora" in tools[tool].parameters:
                    params["use_lora"] = {"default": True}

                # if len(lora_docs) > 1 and "lora2" in tools[tool].parameters:
                #     params["lora2"] = {"default": str(lora_docs[1]["_id"])}
                #     if "use_lora2" in tools[tool].parameters:
                #         params["use_lora2"] = {
                #             "default": True,
                #             "tip": "Same behavior as first lora"
                #         }

                tools[tool].update_parameters(params)

            except Exception as e:
                _log_tool_operation_error(self, "lora_parameter_injection", e)

        try:
            if "elevenlabs" in tools and self.voice:
                tools["elevenlabs"].update_parameters(
                    {"voice": {"default": self.voice}}
                )
        except Exception as e:
            _log_tool_operation_error(
                self, "elevenlabs_voice_update", e, voice=self.voice
            )

        return tools

    @profile_method("get_tools_optimized")
    def get_tools_optimized(self, cache=True, auth_user: str = None):
        """
        Optimized version of get_tools that uses batch loading.
        """
        # Always reload tools fresh - no caching
        self._reload_optimized()

        # Deep copy tools to avoid any shared state between agents
        import copy
        tools = copy.deepcopy(self.tools_)

        # Apply the same post-processing as original method
        # (All the platform-specific logic remains the same)

        # update tools with platform-specific args
        # update discord post tool with allowed channels
        try:
            if "discord" in self.deployments:
                if "discord_post" in tools:
                    allowed_channels = self.deployments[
                        "discord"
                    ].get_allowed_channels()
                    channels_description = " | ".join(
                        [f"ID {c.id} ({c.note})" for c in allowed_channels]
                    )
                    tools["discord_post"].update_parameters(
                        {
                            "channel_id": {
                                "choices": [c.id for c in allowed_channels],
                                "tip": f"Some hints about the available channels: {channels_description}",
                            },
                        }
                    )
        except Exception as e:
            _log_tool_operation_error(
                self, "discord_channel_update", e, platform="discord"
            )

        # update telegram post tool with allowed channels
        try:
            if "telegram" in self.deployments:
                if "telegram_post" in tools:
                    allowed_channels = self.deployments[
                        "telegram"
                    ].get_allowed_channels()
                    channels_description = " | ".join(
                        [f"ID {c.id} ({c.note})" for c in allowed_channels]
                    )
                    tools["telegram_post"].update_parameters(
                        {
                            "channel_id": {
                                "choices": [c.id for c in allowed_channels],
                                "tip": f"Some hints about the available topics: {channels_description}",
                            },
                        }
                    )
        except Exception as e:
            _log_tool_operation_error(
                self, "telegram_channel_update", e, platform="telegram"
            )

        # if a platform is not deployed, remove all tools for that platform
        if "discord" not in self.deployments:
            for tool in DISCORD_TOOLS:
                tools.pop(tool, None)
        if "telegram" not in self.deployments:
            for tool in TELEGRAM_TOOLS:
                tools.pop(tool, None)
        if "twitter" not in self.deployments:
            for tool in TWITTER_TOOLS:
                tools.pop(tool, None)
        if "farcaster" not in self.deployments:
            for tool in FARCASTER_TOOLS:
                tools.pop(tool, None)

        # remove tools that only the owner can use
        if str(auth_user) != str(self.owner):
            for tool in SOCIAL_MEDIA_TOOLS:
                tools.pop(tool, None)

        # if models are found, inject them as defaults for any tools that use lora
        for tool in tools:
            try:
                if "lora" not in tools[tool].parameters:
                    continue

                lora_docs = self.lora_docs

                if not lora_docs:
                    continue

                # Build LoRA information for the tip
                lora_info = []
                for m in lora_docs:
                    lora_id = m["_id"]
                    lora_doc = {m["lora"]: m for m in self.models}[lora_id]
                    lora_info.append(
                        f"{{ ID: {lora_id}, Name: {m['name']}, Description: {m['lora_trigger_text']}, Use When: {lora_doc['use_when']} }}"
                    )

                params = {
                    "lora": {
                        "default": str(lora_docs[0]["_id"]),
                        "tip": "Users may request one of your known LoRAs, or a different unknown one, or no LoRA at all. When referring to a LoRA, strictly use its name, not its description. Notes on when to use the known LoRAs: "
                        + " | ".join(lora_info),
                    },
                }
                if "use_lora" in tools[tool].parameters:
                    params["use_lora"] = {"default": True}

                tools[tool].update_parameters(params)

            except Exception as e:
                _log_tool_operation_error(self, "lora_parameter_injection", e)

        try:
            if "elevenlabs" in tools and self.voice:
                tools["elevenlabs"].update_parameters(
                    {"voice": {"default": self.voice}}
                )
        except Exception as e:
            _log_tool_operation_error(
                self, "elevenlabs_voice_update", e, voice=self.voice
            )

        return tools

    def benchmark_tool_loading(self, auth_user: str = None):
        """
        Benchmark method to compare original vs optimized tool loading performance.
        Runs both methods and prints detailed timing comparisons.
        """
        print("\n" + "=" * 60)
        print("TOOL LOADING PERFORMANCE BENCHMARK")
        print("=" * 60)

        # Clear any existing cache to ensure fair comparison
        global agent_updated_at, _tool_schema_cache, _cache_timestamp
        if self.username in agent_updated_at:
            del agent_updated_at[self.username]
        _tool_schema_cache.clear()
        _cache_timestamp = None

        print(f"Agent: {self.username}")
        enabled_tool_sets = [ts for ts, enabled in self.tools.items() if enabled]
        total_tools = sum(len(TOOL_SETS.get(ts, [])) for ts in enabled_tool_sets)
        print(f"Enabled tool sets: {enabled_tool_sets}")
        print(f"Total tools to load: {total_tools}")
        print("\n" + "-" * 60)

        # Test original method
        print("🐌 TESTING ORIGINAL METHOD:")
        original_start = time.time()
        original_tools = self.get_tools(cache=False, auth_user=auth_user)
        original_time = time.time() - original_start
        print(f"✅ Original method completed: {original_time:.3f}s")
        print(f"   Loaded {len(original_tools)} tools")

        # Clear cache for optimized test
        if self.username in agent_updated_at:
            del agent_updated_at[self.username]
        _tool_schema_cache.clear()
        _cache_timestamp = None

        print("\n" + "-" * 60)
        print("🚀 TESTING OPTIMIZED METHOD:")
        optimized_start = time.time()
        optimized_tools = self.get_tools_optimized(cache=False, auth_user=auth_user)
        optimized_time = time.time() - optimized_start
        print(f"✅ Optimized method completed: {optimized_time:.3f}s")
        print(f"   Loaded {len(optimized_tools)} tools")

        # Calculate improvement
        improvement = ((original_time - optimized_time) / original_time) * 100
        speedup = original_time / optimized_time if optimized_time > 0 else float("inf")

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY:")
        print("=" * 60)
        print(f"Original method:  {original_time:.3f}s")
        print(f"Optimized method: {optimized_time:.3f}s")
        print(f"Performance improvement: {improvement:.1f}%")
        print(f"Speed multiplier: {speedup:.1f}x faster")

        if optimized_time < original_time:
            print(f"🎉 SUCCESS: Optimized method is {speedup:.1f}x faster!")
        else:
            print("⚠️  REGRESSION: Optimized method is slower")

        # Verify tools are equivalent
        original_keys = set(original_tools.keys())
        optimized_keys = set(optimized_tools.keys())
        if original_keys == optimized_keys:
            print("✅ Tool sets are identical")
        else:
            print("❌ Tool sets differ!")
            print(f"   Original only: {original_keys - optimized_keys}")
            print(f"   Optimized only: {optimized_keys - original_keys}")

        print("=" * 60)

        return {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "improvement_percent": improvement,
            "speedup": speedup,
            "tools_match": original_keys == optimized_keys,
        }


def get_agents_from_mongo(
    agents: List[str] = None, include_inactive: bool = False
) -> Dict[str, Agent]:
    """Get all agents from mongo"""

    filter = {"key": {"$in": agents}} if agents else {}
    agents = {}
    agents_collection = get_collection(Agent.collection_name)
    for agent in agents_collection.find(filter):
        try:
            agent = Agent.convert_from_mongo(agent)
            agent = Agent.from_schema(agent)
            if agent.status != "inactive" and not include_inactive:
                if agent.key in agents:
                    raise ValueError(f"Duplicate agent {agent.key} found.")
                agents[agent.key] = agent
        except Exception as e:
            print(f"Error loading agent {agent.key}: {e}")
            print(traceback.format_exc())

    return agents


def get_api_files(root_dir: str = None) -> List[str]:
    """Get all agent directories inside a directory"""

    db = os.getenv("DB").lower()

    if root_dir:
        root_dirs = [root_dir]
    else:
        eve_root = os.path.dirname(os.path.abspath(__file__))
        root_dirs = [
            os.path.join(eve_root, agents_dir) for agents_dir in [f"agents/{db}"]
        ]

    api_files = {}
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            if "api.yaml" in files and "test.json" in files:
                api_path = os.path.join(root, "api.yaml")
                key = os.path.relpath(root).split("/")[-1]
                api_files[key] = api_path

    return api_files


class AgentText(BaseModel):
    """
    Auto-generated greeting and suggestions for prompts and taglines that are specific to an Agent's description.
    """

    suggestions: List[Suggestion] = Field(
        ...,
        description="A list of prompt suggestions and corresponding taglines for the agent. Should be appropriate to the agent's description.",
    )
    greeting: str = Field(
        ...,
        description="A very short greeting for the agent to use as a conversation starter with a new user. Should be no more than 10 words.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "greeting": "I'm your personal creative assistant! How can I help you?",
                    "suggestions": [
                        {
                            "label": "What tools can you use?",
                            "prompt": "Give me a list of all of your tools, and explain your capabilities.",
                        },
                        {
                            "label": "Help me make live visuals",
                            "prompt": "I'm making live visuals for an upcoming event. Can you help me?",
                        },
                        {
                            "label": "Turn a sketch into a painting",
                            "prompt": "I'm making sketches and doodles in my notebook, and I want to transform them into a digital painting.",
                        },
                        {
                            "label": "Draft a character",
                            "prompt": "Help me write out a character description for a video game I am producing.",
                        },
                    ],
                },
                {
                    "greeting": "What kind of a story would you like to write together?",
                    "suggestions": [
                        {
                            "label": "Make a romantic story",
                            "prompt": "I want to write a romantic comedy about a couple who meet at a party. Help me write it.",
                        },
                        {
                            "label": "Imagine a character",
                            "prompt": "I would like to draft a protagonist for a novel I'm writing about the sea.",
                        },
                        {
                            "label": "What have you written before?",
                            "prompt": "Tell me about some of the previous stories you've written.",
                        },
                        {
                            "label": "Revise the style of my essay",
                            "prompt": "I've made an essay about the history of the internet, but I'm not sure if it's written in the style I want. Help me revise it.",
                        },
                    ],
                },
            ]
        }
    )


async def generate_agent_knowledge_description(agent: Agent):
    """
    Given a knowledge document / reference, generate a summary and retrieval criteria
    """

    system_message = "You receive a description of an agent, along with a large document of information the agent must memorize, and you come up with instructions for the agent on when they should consult the reference document."

    knowledge_template = load_template("knowledge_summarize")

    prompt = knowledge_template.render(
        name=agent.username,
        agent_description=agent.persona,
        knowledge=agent.knowledge,
    )

    client = instructor.from_openai(openai.AsyncOpenAI())

    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=KnowledgeDescription,
    )

    return result


async def generate_agent_text(agent: Agent):
    """
    Given an agent's description, generate a greeting and suggestions for prompts and taglines (labels) for the prompts
    """

    system_message = "You receive a description of an agent and come up with a greeting and suggestions for those agents' example prompts and taglines."

    prompt = f"""Come up with exactly FOUR (4, no more, no less) suggestions for sample prompts for the agent {agent.username}, as well as a simple greeting for the agent to begin a conversation with. Make sure all of the text is especially unique to or appropriate to {agent.username}, given their description. Do not use exclamation marks. Here is the description of {agent.username}:\n\n{agent.persona}."""

    client = instructor.from_openai(openai.AsyncOpenAI())
    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=AgentText,
    )

    return result


async def refresh_agent(agent: Agent):
    """
    Refresh an agent's suggestions, greetings, and knowledge descriptions
    """
    # get suggestions and greeting
    agent_text = await generate_agent_text(agent)

    # get knowledge description if there is any knowledge
    if agent.knowledge:
        knowledge_description = await generate_agent_knowledge_description(agent)
        knowledge_description_dict = {
            "summary": knowledge_description.summary,
            "retrieval_criteria": knowledge_description.retrieval_criteria,
        }
    else:
        knowledge_description_dict = None

    time = datetime.now(timezone.utc)

    update = {
        "knowledge_description": knowledge_description_dict,
        "greeting": agent_text.greeting,
        "suggestions": [s.model_dump() for s in agent_text.suggestions],
        "refreshed_at": time,
        "updatedAt": time,
    }

    print(update)

    agents = get_collection(Agent.collection_name)
    agents.update_one({"_id": agent.id}, {"$set": update})


def _log_tool_operation_error(
    agent: Agent, operation_name: str, error: Exception, **context
):
    """Helper to log tool operation errors with Sentry"""
    print(f"Error in {operation_name}: {error}")
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("component", "tool_operation")
        scope.set_tag("operation", operation_name)
        scope.set_tag("agent_username", agent.username)
        scope.set_tag("agent_id", str(agent.id))
        scope.set_context(
            "tool_operation_context",
            {
                "operation": operation_name,
                "agent_username": agent.username,
                "agent_id": str(agent.id),
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                **context,
            },
        )
        sentry_sdk.capture_exception(error)
