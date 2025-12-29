import json
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import instructor
import openai
from bson import ObjectId
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from ..models import Model
from ..mongo import Collection, get_collection
from ..tool_constants import TOOL_SETS
from ..user import Manna, User

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None


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
    level: Literal["editor", "owner", "member"]
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
    thinking_effort_instructions: Optional[str] = (
        None  # Custom instructions when thinking_policy == "auto"
    )

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
    voice: Optional[str] = None
    website: Optional[str] = None
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
    agent_memory_enabled: Optional[bool] = False  # Not yet used anywhere yet

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

    def _reload(self, extra_tools: list[str] = []):
        """Reload all tools, loras, and deployments from mongo"""
        from ..agent.session.models import Deployment
        from ..tool import get_tools_from_mongo
        from .tool_loaders import (
            get_agent_specific_tools,
            load_deployments,
            load_lora_docs,
        )

        # Load deployments to memory (triggers KMS decryption for all deployments)
        span_context = (
            sentry_sdk.start_span(
                op="agent.load_deployments", description=f"agent={self.username}"
            )
            if sentry_sdk
            else nullcontext()
        )
        with span_context:
            self.deployments = load_deployments(self.id, Deployment)

        # Load loras to memory
        span_context = (
            sentry_sdk.start_span(op="agent.load_loras")
            if sentry_sdk
            else nullcontext()
        )
        with span_context:
            models_collection = get_collection(Model.collection_name)
            self.lora_docs = load_lora_docs(self.models, models_collection)

        # Load tools from mongo
        span_context = (
            sentry_sdk.start_span(op="agent.load_tools")
            if sentry_sdk
            else nullcontext()
        )
        with span_context:
            # Collect all tools needed
            tools_to_load = []
            for tool_set, set_tools in TOOL_SETS.items():
                if not self.tools.get(tool_set):
                    continue
                tools_to_load.extend(set_tools)

            # Load extra tools
            tools_to_load.extend(extra_tools)

            # Load agent-specific tools
            tools_to_load.extend(get_agent_specific_tools(self.username, self.tools))

            if tools_to_load:
                self.tools_ = get_tools_from_mongo(tools_to_load)
            else:
                self.tools_ = {}

    def get_tools(self, cache=True, auth_user: str = None, extra_tools: list[str] = []):
        from .tool_loaders import (
            filter_tools_by_feature_flags,
            inject_deployment_parameters,
            inject_lora_parameters,
            inject_voice_parameters,
            remove_non_deployed_platform_tools,
        )

        # for Solienne only, make all tools unavailable except for admin
        if self.username == "solienne":
            user = User.from_mongo(auth_user)
            solienne_whitelist = [
                "ameesia77",
                "farcaster_ameesia",
                "farcaster_kristicoronado",
                "farcaster_seth",
                "farcaster_gene",
                "farcaster_sethgoldstein",
                "farcaster_xanderst",
                "farcaster_jmill",
            ]
            if (
                "eden_admin" not in user.featureFlags
                and user.username not in solienne_whitelist
            ):
                return {}

        self._reload(extra_tools)
        tools = self.tools_

        # Inject deployment-specific parameters (channels, etc.)
        tools = inject_deployment_parameters(tools, self.deployments, self.username)

        # Remove tools for non-deployed platforms
        tools = remove_non_deployed_platform_tools(tools, self.deployments)

        # Filter tools based on feature flags
        tools = filter_tools_by_feature_flags(tools, self.featureFlags, {})

        # Inject LoRA parameters for tools that use loras
        tools = inject_lora_parameters(
            tools, self.lora_docs, self.models or [], self.username
        )

        # Inject voice parameter for elevenlabs
        tools = inject_voice_parameters(tools, self.voice, self.username)

        return tools


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
            if agent.active or include_inactive:
                if agent.key in agents:
                    raise ValueError(f"Duplicate agent {agent.key} found.")
                agents[agent.key] = agent
        except Exception as e:
            logger.error(f"Error loading agent {agent.key}: {e}")
            logger.error(traceback.format_exc())

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
    Refresh an agent's suggestions
    """
    # get suggestions and greeting
    agent_text = await generate_agent_text(agent)

    time = datetime.now(timezone.utc)

    update = {
        # "greeting": agent_text.greeting,
        "suggestions": [s.model_dump() for s in agent_text.suggestions],
        "refreshed_at": time,
        "updatedAt": time,
    }

    agents = get_collection(Agent.collection_name)
    agents.update_one({"_id": agent.id}, {"$set": update})
