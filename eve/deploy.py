import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict


REPO_URL = "https://github.com/edenartlab/eve.git"
REPO_BRANCH = "main"
DEPLOYMENT_ENV_NAME = "deployments"
db = os.getenv("DB", "STAGE").upper()


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


def prepare_client_file(file_path: str, agent_key: str, env: str) -> None:
    """Modify the client file to use correct secret name and fix pyproject path"""
    with open(file_path, "r") as f:
        content = f.read()

    # Get the repo root directory (three levels up from the client file)
    repo_root = Path(__file__).parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    # Replace the static secret name with the dynamic one
    modified_content = content.replace(
        'modal.Secret.from_name("client-secrets")',
        f'modal.Secret.from_name("{agent_key}-secrets-{env}")',
    )
    modified_content = modified_content.replace(
        'modal.Secret.from_name("eve-secrets-{db}")',
        f'modal.Secret.from_name("eve-secrets-{db}")',
    )
    print(f"Modified content: {modified_content}")

    # Fix pyproject.toml path to use absolute path
    modified_content = modified_content.replace(
        '.pip_install_from_pyproject("pyproject.toml")',
        f'.pip_install_from_pyproject("{pyproject_path}")',
    )

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "modal_client.py"
    with open(temp_file, "w") as f:
        f.write(modified_content)

    return str(temp_file)


def deploy_client(agent_key: str, client_name: str, env: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repo
        clone_repo(temp_dir)

        # Check for client file in the cloned repo
        client_path = os.path.join(
            temp_dir, "eve", "clients", client_name, "modal_client.py"
        )
        if os.path.exists(client_path):
            # Modify the client file to use the correct secret name
            prepare_client_file(client_path, agent_key, env)
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
            f"{agent_key}-{client_name}-{db}",
            "-e",
            DEPLOYMENT_ENV_NAME,
        ],
        check=True,
    )
