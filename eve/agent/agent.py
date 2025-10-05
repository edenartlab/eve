import time
import json
import traceback
from elevenlabs import Model
import openai
import instructor
import sentry_sdk
from datetime import timezone

# from pathlib import Path
from bson import ObjectId
from typing import Optional, Literal, Any, Dict, List
from datetime import datetime

# from dotenv import dotenv_values
from pydantic import Field, BaseModel, ConfigDict
from functools import wraps
# from pydantic.json_schema import SkipJsonSchema

from ..tool_constants import (
    TWITTER_TOOLS,
    DISCORD_TOOLS,
    FARCASTER_TOOLS,
    TELEGRAM_TOOLS,
    SHOPIFY_TOOLS,
    PRINTIFY_TOOLS,
    CAPTIONS_TOOLS,
    TIKTOK_TOOLS,
    SOCIAL_MEDIA_TOOLS,
    TOOL_SETS,
)
from ..mongo import Collection, get_collection
from ..models import Model
from ..user import User, Manna
from ..utils import load_template
from .thread import Thread


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


@Collection("agent_permissions")
class AgentPermission(BaseModel):
    """Permissions for agents stored in agent_permissions collection."""
    
    agent: ObjectId
    user: ObjectId
    level: Literal["editor", "owner"]
    grantedBy: ObjectId
    grantedAt: Optional[datetime] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentPermissions(BaseModel):
    """Permissions configuration for an agent."""

    # Permissions are now stored in separate agent_permissions collection
    # This class kept for backwards compatibility with existing documents
    editors: Optional[List[ObjectId]] = Field(
        None,
        description="DEPRECATED: Use agent_permissions collection instead",
    )
    owners: Optional[List[ObjectId]] = Field(
        None,
        description="DEPRECATED: Use agent_permissions collection instead",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentLLMSettings(BaseModel):
    """LLM configuration and thinking settings for an agent."""
    
    model_profile: Optional[str] = "medium"  # "low", "medium", "high"
    thinking_policy: Optional[str] = "auto"  # "auto", "off", "always"
    thinking_effort_cap: Optional[str] = "medium"  # "low", "medium", "high"
    thinking_effort_instructions: Optional[str] = None  # Custom instructions when thinking_policy == "auto"
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentExtras(BaseModel):
    """Additional configuration and metadata for an agent."""

    permissions: Optional[AgentPermissions] = Field(
        None,
        description="DEPRECATED: Permissions moved to agent_permissions collection",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("users3")
class Agent(User):
    """
    Base class for all agents.
    """

    type: Literal["agent"] = "agent"
    owner: ObjectId
    # secrets: Optional[Dict[str, SecretStr]] = Field(None, exclude=True)

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
    # model: Optional[ObjectId] = None  # deprecated
    models: Optional[List[Dict[str, Any]]] = None
    test_args: Optional[List[Dict[str, Any]]] = None

    llm_settings: Optional[AgentLLMSettings] = Field(default_factory=AgentLLMSettings)
    tools: Optional[Dict[str, bool]] = {}  # tool sets specified by user
    tools_: Optional[Dict[str, Dict]] = Field({}, exclude=True)  # actual loaded tools
    lora_docs: Optional[List[Dict[str, Any]]] = Field([], exclude=True)
    deployments: Optional[List[str]] = Field({}, exclude=True)

    owner_pays: Optional[Literal["off", "deployments", "full"]] = "off"
    agent_extras: Optional[AgentExtras] = None

    user_memory_enabled: Optional[bool] = False
    agent_memory_enabled: Optional[bool] = False # Not yet used anywhere yet
    
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

    # @profile_method("_reload")
    def _reload(self):
        """Reload all tools, loras, and deployments from mongo"""
        from ..tool import get_tools_from_mongo
        from ..agent.session.models import Deployment

        # load deployments to memory
        self.deployments = {
            deployment.platform.value: deployment
            for deployment in Deployment.find({"agent": ObjectId(str(self.id))})
        }

        # load loras to memory
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in self.models or []}

        # load loras to memory
        if self.models:
            lora_ids = list(loras_dict.keys())

            # Single batch query for all loras
            lora_docs = list(
                models_collection.find(
                    {"_id": {"$in": lora_ids}, "deleted": {"$ne": True}}
                )
            )
            self.lora_docs = lora_docs
        else:
            self.lora_docs = []

        # Collect all tools needed
        tools_to_load = []
        for tool_set, set_tools in TOOL_SETS.items():
            if not self.tools.get(tool_set):
                continue
            tools_to_load.extend(set_tools)

        # agent-specific tools
        # todo: systemize this for other agents
        if self.username == "abraham":
            tools_to_load.append("abraham_publish")

        if tools_to_load:
            self.tools_ = get_tools_from_mongo(tools_to_load)
        else:
            self.tools_ = {}

    # @profile_method("get_tools")
    def get_tools(self, cache=True, auth_user: str = None):
        self._reload()
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
        if "shopify" not in self.deployments:
            for tool in SHOPIFY_TOOLS:
                tools.pop(tool, None)
        if "printify" not in self.deployments:
            for tool in PRINTIFY_TOOLS:
                tools.pop(tool, None)
        if "captions" not in self.deployments:
            for tool in CAPTIONS_TOOLS:
                tools.pop(tool, None)
        if "tiktok" not in self.deployments:
            for tool in TIKTOK_TOOLS:
                tools.pop(tool, None)

        # remove tools that only the owner can use
        # Check if user is the owner or has owner-level permissions
        has_owner_permission = False
        if auth_user:
            if str(auth_user) == str(self.owner):
                has_owner_permission = True
            else:
                # Check agent_permissions collection for owner-level access
                permissions_collection = get_collection("agent_permissions")
                permission = permissions_collection.find_one({
                    "agent": self.id,
                    "user": ObjectId(str(auth_user)),
                    "level": "owner"
                })
                if permission:
                    has_owner_permission = True
        
        if not has_owner_permission:
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


def get_agents_from_mongo(
    agents: List[str] = None, 
    include_inactive: bool = False
) -> Dict[str, Agent]:
    """Get all agents from mongo"""

    filter = {"key": {"$in": agents}} if agents else {}
    agents = {}
    agents_collection = get_collection(Agent.collection_name)
    for agent in agents_collection.find(filter):
        try:
            agent = Agent.convert_from_mongo(agent)
            agent = Agent.from_schema(agent)
            if agent.active or include_inactive:
                if agent.key in agents:
                    raise ValueError(f"Duplicate agent {agent.key} found.")
                agents[agent.key] = agent
        except Exception as e:
            print(f"Error loading agent {agent.key}: {e}")
            print(traceback.format_exc())

    return agents


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
