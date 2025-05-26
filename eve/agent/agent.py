import os
import json
import traceback
from elevenlabs import Model
import openai
import instructor
from datetime import timezone
from pathlib import Path
from bson import ObjectId
from typing import Optional, Literal, Any, Dict, List
from datetime import datetime
from dotenv import dotenv_values
from pydantic import SecretStr, Field, BaseModel, ConfigDict

from ..tool_constants import (
    BASE_TOOLS,
    FLUX_LORA_TOOLS,
    SDXL_LORA_TOOLS,
    OWNER_ONLY_TOOLS,
    AGENTIC_TOOLS,
)
from ..mongo import Collection, get_collection
from ..models import Model
from ..user import User, Manna
from ..eden_utils import load_template
from .thread import Thread


last_tools_update = None
agent_tools_cache = {}


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

    tools: Optional[Dict[str, Dict]] = {}
    add_base_tools: Optional[bool] = True

    owner_pays: Optional[bool] = False

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
        schema = cls._setup_tools(schema)

        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict) -> dict:
        schema = cls._setup_tools(schema)
        return schema

    def save(self, **kwargs):
        # do not overwrite any username if it already exists
        users = get_collection(User.collection_name)
        if users.find_one({"username": self.username, "type": "user"}):
            raise ValueError(f"Username {self.username} already taken")

        # save user, and create mannas record if it doesn't exist
        kwargs["featureFlags"] = ["freeTools"]  # give agents free tools for now
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

    @classmethod
    def _setup_tools(cls, schema: dict) -> dict:
        """
        Sets up the agent's tools based on the tools defined in the schema.
        If a model (lora) is set, hardcode it into the tools.
        """
        tools = schema.get("tools") or {}

        # if tools are set explicitly, start with them
        schema["tools"] = {k: v or {} for k, v in tools.items()}

        # if add_base_tools is set, add the base tools
        if schema.get("add_base_tools", True):
            schema["tools"].update(
                {k: {} for k in BASE_TOOLS if k not in schema["tools"]}
            )

        models = schema.get("models") or (
            [{"lora": schema.get("model"), "use_when": "This is your default model."}]
            if schema.get("model")
            else []
        )
        for m in models:
            m["doc"] = Model.from_mongo(m["lora"])

        if models:
            base_models = [
                {
                    "type": "flux-dev",
                    "models": [m for m in models if m["doc"].base_model == "flux-dev"],
                    "tools": [
                        t for t in schema["tools"].keys() if t in FLUX_LORA_TOOLS
                    ],
                },
                {
                    "type": "sdxl",
                    "models": [m for m in models if m["doc"].base_model == "sdxl"],
                    "tools": [
                        t for t in schema["tools"].keys() if t in SDXL_LORA_TOOLS
                    ],
                },
            ]

            for base_model in base_models:
                model_list, tools_list = base_model["models"], base_model["tools"]

                if tools_list and model_list:
                    if len(model_list) == 1:
                        tip = f'Only use "{base_model["type"]}" models. Set the "lora" argument to the ID of the default lora (ID: {str(model_list[0]["lora"])}, Name: "{model_list[0]["doc"].name}", Description: "{model_list[0]["doc"].lora_trigger_text}"), if the following conditions are true: "{model_list[0]["use_when"]}"). If no lora is desired, leave this blank. If a different lora is desired, use its ID instead.'
                        default_lora = model_list[0]["doc"].id
                    else:
                        models_info = " | ".join(
                            [
                                f'ID: {m["lora"]}, Name: "{m["doc"].name}", Description: "{m["doc"].lora_trigger_text}", Use When: "{m["use_when"]}"'
                                for m in model_list
                            ]
                        )
                        tip = f'Only use "{base_model["type"]}" models. You are can use the following loras under the "Use When" circumstances: {models_info}. To use no lora, leave the "lora" argument blank.'
                        default_lora = model_list[0]["doc"].id
                        for model in model_list:
                            if "default" in model["use_when"].lower():
                                default_lora = model["doc"].id
                                break

                    tip += " If you use a lora, make sure to refer to it in the prompt using its exact Name. Avoid restating the Description in the prompt as it's implicit in the lora already and is redundant."

                    # Update all related tools with the tip
                    for tool in tools_list:
                        schema["tools"][tool] = schema["tools"][tool] or {
                            "parameters": {}
                        }
                        schema["tools"][tool]["parameters"].update(
                            {"lora": {"tip": tip, "default": str(default_lora)}}
                        )

        return schema

    def get_tools(self, cache=False, auth_user: str = None):
        global last_tools_update

        if cache:
            # get latest updatedAt timestamp for tools
            from ..tool import Tool  # avoid circular import

            tools = get_collection(Tool.collection_name)
            timestamps = tools.find({}, {"updatedAt": 1})
            last_tools_update_ = max(
                (doc.get("updatedAt") for doc in timestamps if doc.get("updatedAt")),
                default=None,
            )
            if last_tools_update is None:
                last_tools_update = last_tools_update_

            # reset cache if outdated
            cache_outdated = last_tools_update < last_tools_update_
            last_tools_update = max(last_tools_update, last_tools_update_)
            if self.username not in agent_tools_cache or cache_outdated:
                print("Cache is outdated, resetting...")
                agent_tools_cache[self.username] = {}

            # insert new tools into cache
            for k, v in self.tools.items():
                if k not in agent_tools_cache[self.username]:
                    try:
                        tool = Tool.from_raw_yaml({"parent_tool": k, **v})
                    except Exception as e:
                        print(f"Error loading tool {k}: {e}")
                        print(traceback.format_exc())
                        continue
                    agent_tools_cache[self.username][k] = tool

            tools = agent_tools_cache[self.username]
        else:
            # Import Tool class when needed to avoid circular imports
            from ..tool import Tool

            tools = {
                k: Tool.from_raw_yaml({"parent_tool": k, **v})
                for k, v in self.tools.items()
            }

        # remove tools that only the owner can use
        if str(auth_user) != str(self.owner):
            for tool in OWNER_ONLY_TOOLS:
                tools.pop(tool, None)

        # inject agent arg
        for tool in AGENTIC_TOOLS:
            if tool in tools:
                tools[tool].parameters.update(
                    {
                        "agent": {
                            "default": str(self.id),
                            "hide_from_agent": True,
                        }
                    }
                )

        return tools

    def get_tool(self, tool_name, cache=False):
        return self.get_tools(cache=cache)[tool_name]


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
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
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
