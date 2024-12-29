import os
import subprocess
import tempfile
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict

from eve.models import ClientType

REPO_URL = "https://github.com/edenartlab/eve.git"
REPO_BRANCH = "feat/merge-deployer-api"
DEPLOYMENT_ENV_NAME = "deployments"


class DeployCommand(str, Enum):
    DEPLOY = "deploy"
    STOP = "stop"


class DeployRequest(BaseModel):
    agent_key: str
    platform: ClientType
    command: DeployCommand
    credentials: Optional[Dict[str, str]] = None


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


def clone_repo(temp_dir: str):
    """Clone the eve repository to a temporary directory"""
    subprocess.run(
        ["git", "clone", "-b", REPO_BRANCH, "--single-branch", REPO_URL, temp_dir],
        check=True,
    )


def modify_client_file(file_path: str, agent_key: str) -> None:
    """Modify the client file to use correct secret name and fix pyproject path"""
    with open(file_path, "r") as f:
        content = f.read()

    # Get the repo root directory (three levels up from the client file)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    )
    pyproject_path = os.path.join(repo_root, "pyproject.toml")

    # Replace the static secret name with the dynamic one
    modified_content = content.replace(
        'modal.Secret.from_name("client-secrets")',
        f'modal.Secret.from_name("{agent_key}-client-secrets")',
    )

    # Fix pyproject.toml path to use absolute path
    modified_content = modified_content.replace(
        '.pip_install_from_pyproject("pyproject.toml")',
        f'.pip_install_from_pyproject("{pyproject_path}")',
    )

    with open(file_path, "w") as f:
        f.write(modified_content)


def deploy_client(agent_key: str, client_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repo
        clone_repo(temp_dir)

        # Check for client file in the cloned repo
        client_path = os.path.join(
            temp_dir, "eve", "clients", client_name, "modal_client.py"
        )
        if os.path.exists(client_path):
            # Modify the client file to use the correct secret name
            modify_client_file(client_path, agent_key)
            subprocess.run(
                ["modal", "deploy", client_path, "-e", DEPLOYMENT_ENV_NAME], check=True
            )
        else:
            raise Exception(f"Client modal file not found: {client_path}")


def stop_client(agent_key: str, client_name: str):
    subprocess.run(
        [
            "modal",
            "app",
            "stop",
            f"{agent_key}-client-{client_name}",
            "-e",
            DEPLOYMENT_ENV_NAME,
        ],
        check=True,
    )
