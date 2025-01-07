import os
import time
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

CHECK_INTERVAL = 30  # how often to check cached agents for updates

default_presets_flux = {
    "flux_dev_lora": {},
    "runway": {},
    "reel": {},
}
default_presets_sdxl = {
    "txt2img": {},
    "runway": {},
    "reel": {},
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

    name: str
    description: str
    instructions: str
    model: Optional[ObjectId] = None
    test_args: Optional[List[Dict[str, Any]]] = None
    
    tools: Optional[Dict[str, Dict]] = None
    tools_cache: SkipJsonSchema[Optional[Dict[str, Tool]]] = Field(None, exclude=True)
    last_check: ClassVar[Dict[str, float]] = {}  # seconds

    def __init__(self, **data):
        if isinstance(data.get('owner'), str):
            data['owner'] = ObjectId(data['owner'])
        # Load environment variables into secrets dictionary
        db = os.getenv("DB")
        env_dir = Path(__file__).parent / "agents"
        env_vars = dotenv_values(f"{str(env_dir)}/{db.lower()}/{data['username']}/.env")
        data['secrets'] = {key: SecretStr(value) for key, value in env_vars.items()}
        super().__init__(**data)
            
    @classmethod
    def convert_from_yaml(cls, schema: dict, file_path: str = None) -> dict:
        """
        Convert the schema into the format expected by the model.
        """
        test_file = file_path.replace("api.yaml", "test.json")
        with open(test_file, 'r') as f:
            schema["test_args"] = json.load(f)

        owner = schema.get('owner')
        schema["owner"] = ObjectId(owner) if isinstance(owner, str) else owner
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
            upsert_filter={"username": self.username, "type": "agent"}, 
            **kwargs
        )
        Manna.load(user=self.id)  # create manna record if it doesn't exist
       
    @classmethod
    def from_yaml(cls, file_path, cache=False):
        if cache:
            if file_path not in _agent_cache:
                _agent_cache[file_path] = super().from_yaml(file_path)
            return _agent_cache[file_path]
        else:
            return super().from_yaml(file_path)

    @classmethod
    def from_mongo(cls, document_id, cache=False):
        if cache:
            id = str(document_id)
            if id not in _agent_cache:
                _agent_cache[id] = super().from_mongo(document_id)
            cls._check_for_updates(id, document_id)
            return _agent_cache[id]
        else:
            return super().from_mongo(document_id)
    
    @classmethod
    def load(cls, username, cache=False):
        if cache:
            if username not in _agent_cache:
                _agent_cache[username] = super().load(username=username)
            cls._check_for_updates(username, _agent_cache[username].id)
            return _agent_cache[username]
        else:
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
        if tools:
            schema["tools"] = {k: v or {} for k, v in tools.items()}
        else:
            schema["tools"] = default_presets_flux
            if "model" in schema:
                model = Model.from_mongo(schema["model"])
                if model.base_model == "flux-dev":
                    schema["tools"] = default_presets_flux
                    schema["tools"]["flux_dev_lora"] = {
                        "name": f"Generate {model.name}",
                        "description": f"Generate an image of {model.name}",
                        "parameters": {
                            "prompt": {
                                "description": f"The text prompt. Always mention {model.name}."
                            },
                            "lora": {
                                "default": str(model.id),
                                "hide_from_agent": True,
                            },
                            "lora_strength": {
                                "default": 1.0,
                                "hide_from_agent": True,
                            }
                        }
                    }
                    schema["tools"]["reel"] = {
                        "name": f"Generate {model.name}",
                        "tip": f"Make sure to always include {model.name} in all of the prompts.",
                        "parameters": {
                            "lora": {
                                "default": str(model.id),
                                "hide_from_agent": True,
                            },
                            "lora_strength": {
                                "default": 1.0,
                                "hide_from_agent": True,
                            }
                        }
                    }
                elif model.base_model == "sdxl":
                    schema["tools"] = default_presets_sdxl

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
    
    @classmethod
    def _check_for_updates(cls, cache_key: str, agent_id: ObjectId):
        """Check if agent needs to be updated based on updatedAt field"""
        current_time = time.time()
        last_check = cls.last_check.get(cache_key, 0)

        if current_time - last_check >= CHECK_INTERVAL:
            cls.last_check[cache_key] = current_time
            collection = get_collection(cls.collection_name)
            db_agent = collection.find_one({"_id": agent_id})
            if db_agent and db_agent.get("updatedAt") != _agent_cache[cache_key].updatedAt:
                _agent_cache[cache_key].reload()


def get_agents_from_mongo(agents: List[str] = None, include_inactive: bool = False) -> Dict[str, Agent]:
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
            os.path.join(eve_root, agents_dir) 
            for agents_dir in [f"agents/{db}"]
        ]

    api_files = {}
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            if "api.yaml" in files and "test.json" in files:
                api_path = os.path.join(root, "api.yaml")
                key = os.path.relpath(root).split("/")[-1]
                api_files[key] = api_path
            
    return api_files

# Agent cache for fetching commonly used agents
_agent_cache: Dict[str, Dict[str, Agent]] = {}
