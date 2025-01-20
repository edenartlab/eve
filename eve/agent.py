import os
import json
import traceback
from pathlib import Path
from bson import ObjectId
from typing import Optional, Literal, Any, Dict, List, ClassVar
from dotenv import dotenv_values
from pydantic import SecretStr, Field
from pydantic.json_schema import SkipJsonSchema

from .thread import Thread
from .tool import Tool
from .mongo import Collection, get_collection
from .user import User, Manna
from .models import Model

CHECK_INTERVAL = 30


default_presets_flux = {
    "flux_schnell": {
        "tip": "This must be your primary tool for making images. The other flux tools are only used for inpainting, remixing, and variations."
    },
    "flux_inpainting": {},
    "flux_redux": {},
    "vid2vid_sdxl": {
        "tip": "Only use this tool if asked to restyle an existing video with a style image"
    },
    "video_FX": {
        "tip": "Only use this tool if asked to make subtle or targeted variations on an existing video"
    },
    "texture_flow": {"tip": "Just use this tool if asked to make VJing material."},
    "outpaint": {},
    "remix_flux_schnell": {},
    "stable_audio": {},
    "musicgen": {},
    "runway": {
        "tip": "This should be your primary tool for making videos or animations. Only use the other video tools if specifically asked to or asked to make VJing material."
    },
    "reel": {
        "tip": "This is a tool for making short films with vocals, music, and several video cuts. This can be used to make commercials, films, music videos, and other kinds of shortform content. But it takes a while to run, around 5 minutes."
    },
    "news": {},
    "websearch": {},
    "audio_video_combine": {
        "tip": "This and video_concat can merge a video track with an audio track, so it's good for compositing/mixing or manually creating reels."
    },
    "video_concat": {},
}


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
    description: str
    instructions: str
    model: Optional[ObjectId] = None
    test_args: Optional[List[Dict[str, Any]]] = None

    tools: Optional[Dict[str, Dict]] = None
    tools_cache: SkipJsonSchema[Optional[Dict[str, Tool]]] = Field(None, exclude=True)
    last_check: ClassVar[Dict[str, float]] = {}  # seconds

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

        owner = schema.get("owner")
        schema["owner"] = ObjectId(owner) if isinstance(owner, str) else owner
        model = schema.get("model")
        schema["model"] = ObjectId(model) if isinstance(model, str) else model
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

    def request_thread(self, key=None, user=None):
        thread = Thread(
            key=key,
            agent=self.id,
            user=user,
        )
        thread.save()
        return thread

    @classmethod
    def _setup_tools(cls, schema: dict) -> dict:
        """
        Sets up the agent's tools based on the tools defined in the schema.
        If a model (lora) is set, hardcode it into the tools.
        """
        tools = schema.get("tools")

        # if tools are defined, use those
        if tools:
            schema["tools"] = {k: v or {} for k, v in tools.items()}
        
        # if no tools are defined, use the default presets
        else:
            schema["tools"] = default_presets_flux.copy()

            # if a model is set, remove flux_schnell and replace it with flux_dev_lora
            if schema.get("model"):
                model = Model.from_mongo(schema["model"])
                if model.base_model == "flux-dev":
                    schema["tools"].pop("flux_schnell", None)
                    schema["tools"]["flux_dev_lora"] = {
                        "description": f"This is your primary and default tool for making images. The other flux tools are only used for inpainting, remixing, and variations. In particular, this will generate an image of {model.name}",
                        "tip": f"If you want to depict {model.name} in the image, make sure to include {model.name} in the prompt.",
                        "parameters": {
                            "prompt": {
                                "tip": 'Try to enhance or embellish prompts. For example, if the user requests "Verdelis as a mermaid smoking a cigar", you would make it much longer and more intricate and detailed, like "Verdelis as a dried-out crusty old mermaid, wrinkled and weathered skin, tangled and brittle seaweed-like hair, smoking a smoldering cigarette underwater with tiny bubbles rising, jagged and cracked tail with faded iridescent scales, adorned with a tarnished coral crown, holding a rusted trident, faint sunlight beams coming through." If the user provides a lot of detail, just stay faithful to their wishes.'
                            },
                            "lora": {
                                "default": str(model.id),
                                "hide_from_agent": True,
                            },
                            "lora_strength": {
                                "default": 1.0,
                                "hide_from_agent": True,
                            },
                        },
                    }
                    schema["tools"]["reel"] = {
                        "tip": f"If you want to depict {model.name} in the image, make sure to include {model.name} in the prompt.",
                        "parameters": {
                            "use_lora": {
                                "default": True,
                                "hide_from_agent": True,
                            },
                            "lora": {
                                "default": str(model.id),
                                "hide_from_agent": True,
                            },
                            "lora_strength": {
                                "default": 1.0,
                                "hide_from_agent": True,
                            },
                        },
                    }
                elif model.base_model == "sdxl":
                    # schema["tools"] = default_presets_sdxl.copy()
                    pass

        return schema

    def get_tools(self, cache=False):
        if not hasattr(self, "tools") or not self.tools:
            self.tools = {}

        if cache:
            self.tools_cache = self.tools_cache or {}
            for k, v in self.tools.items():
                if k not in self.tools_cache:
                    tool = Tool.from_raw_yaml({"parent_tool": k, **v})
                    self.tools_cache[k] = tool
            return self.tools_cache
        else:
            return {
                k: Tool.from_raw_yaml({"parent_tool": k, **v})
                for k, v in self.tools.items()
            }

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
            print(traceback.format_exc())
            print(f"Error loading agent {agent['key']}: {e}")

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
