from enum import Enum
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict

from bson import ObjectId

from eve.mongo import Collection, Document, get_collection


REPO_URL = "https://github.com/edenartlab/eve.git"
DEPLOYMENT_ENV_NAME = "deployments"
db = os.getenv("DB", "STAGE").upper()
REPO_BRANCH = "main" if db == "PROD" else "staging"


class ClientType(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"


@Collection("deployments")
class Deployment(Document):
    agent: ObjectId
    platform: str  # Store the string value instead of enum

    def __init__(self, **data):
        # Convert ClientType enum to string if needed
        if isinstance(data.get("platform"), ClientType):
            data["platform"] = data["platform"].value
        super().__init__(**data)

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent", 1), ("platform", 1)], unique=True)

    @classmethod
    def find(cls, query):
        """Find all deployments matching the query"""
        collection = get_collection(cls.collection_name)
        return [cls(**doc) for doc in collection.find(query)]

    @classmethod
    def delete_deployment(cls, agent_id: ObjectId, platform: str):
        """Delete a deployment record"""
        collection = get_collection(cls.collection_name)
        collection.delete_one({"agent": agent_id, "platform": platform})


def authenticate_modal_key() -> bool:
    token_id = os.getenv("MODAL_DEPLOYER_TOKEN_ID")
    token_secret = os.getenv("MODAL_DEPLOYER_TOKEN_SECRET")
    result = subprocess.run(
        [
            "modal",
            "token",
            "set",
            "--token-id",
            token_id,
            "--token-secret",
            token_secret,
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)


def check_environment_exists(env_name: str) -> bool:
    result = subprocess.run(
        ["modal", "environment", "list"], capture_output=True, text=True
    )
    return f"│ {env_name} " in result.stdout


def create_environment(env_name: str):
    print(f"Creating environment {env_name}")
    subprocess.run(["modal", "environment", "create", env_name])


def create_modal_secrets(secrets_dict: Dict[str, str], group_name: str):
    if not secrets_dict:
        return

    cmd_parts = ["modal", "secret", "create", group_name]
    for key, value in secrets_dict.items():
        if value is not None:
            value = str(value).strip().strip("'\"")
            cmd_parts.append(f"{key}={value}")
    cmd_parts.extend(["-e", DEPLOYMENT_ENV_NAME, "--force"])

    subprocess.run(cmd_parts)


def clone_repo(temp_dir: str, branch: str = None):
    """Clone the eve repository to a temporary directory"""
    branch = branch or REPO_BRANCH
    subprocess.run(
        ["git", "clone", "-b", branch, "--single-branch", REPO_URL, temp_dir],
        check=True,
    )


def prepare_client_file(file_path: str, agent_key: str, env: str) -> None:
    """Modify the client file to use correct secret name"""
    with open(file_path, "r") as f:
        content = f.read()

    repo_root = Path(file_path).parent.parent.parent.parent
    print("REPO ROOT", repo_root)
    pyproject_path = repo_root / "pyproject.toml"

    # Replace the static secret name with the dynamic one
    modified_content = content.replace(
        'modal.Secret.from_name("client-secrets")',
        f'modal.Secret.from_name("{agent_key}-secrets-{env}")',
    )

    # Fix pyproject.toml path to use absolute path from cloned repo
    modified_content = modified_content.replace(
        '.pip_install_from_pyproject("pyproject.toml")',
        f'.pip_install_from_pyproject("{pyproject_path}")',
    )

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "modal_client.py"
    with open(temp_file, "w") as f:
        f.write(modified_content)

    return str(temp_file)


def deploy_client(agent_key: str, client_name: str, env: str, repo_branch: str = None):
    """Deploy a Modal client for an agent."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repo using provided branch or default
        branch = repo_branch or REPO_BRANCH
        clone_repo(temp_dir, branch)

        # Check for client file in the cloned repo
        client_path = os.path.join(
            temp_dir, "eve", "clients", client_name, "modal_client.py"
        )
        if os.path.exists(client_path):
            # Modify the client file to use the correct secret name
            temp_file = prepare_client_file(client_path, agent_key, env)
            app_name = f"{agent_key}-{client_name}-{env}"

            subprocess.run(
                [
                    "modal",
                    "deploy",
                    "--name",
                    app_name,
                    temp_file,
                    "-e",
                    DEPLOYMENT_ENV_NAME,
                ],
                check=True,
            )
        else:
            raise Exception(f"Client modal file not found: {client_path}")


def stop_client(agent_key: str, client_name: str):
    """Stop a Modal client. Raises an exception if the stop fails."""
    result = subprocess.run(
        [
            "modal",
            "app",
            "stop",
            f"{agent_key}-{client_name}-{db.lower()}",
            "-e",
            DEPLOYMENT_ENV_NAME,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Failed to stop client: {result.stderr}")
