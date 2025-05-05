from enum import Enum
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Optional
import secrets as python_secrets
import aiohttp
from ably import AblyRest

from bson import ObjectId
from pydantic import BaseModel

from .agent import Agent
from .mongo import Collection, Document, get_collection


REPO_URL = "https://github.com/edenartlab/eve.git"
DEPLOYMENT_ENV_NAME = "deployments"
db = os.getenv("DB", "STAGE").upper()
REPO_BRANCH = "main" if db == "PROD" else "staging"


class ClientType(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
    TWITTER = "twitter"
    TELEGRAM_MODAL = "telegram_modal"
    DISCORD_MODAL = "discord_modal"
    WEB = "web"


modal_platforms = [ClientType.DISCORD_MODAL, ClientType.TELEGRAM_MODAL]


class AllowlistItem(BaseModel):
    id: str
    note: Optional[str] = None


class DeploymentSettingsDiscord(BaseModel):
    oauth_client_id: Optional[str] = None
    oauth_url: Optional[str] = None
    channel_allowlist: Optional[List[AllowlistItem]] = None


class DeploymentSettingsTelegram(BaseModel):
    topic_allowlist: Optional[List[AllowlistItem]] = None


class DeploymentSettingsFarcaster(BaseModel):
    pass


class DeploymentSettingsTwitter(BaseModel):
    username: Optional[str] = None


class DeploymentSecretsDiscord(BaseModel):
    token: str
    application_id: Optional[str] = None


class DeploymentSecretsTelegram(BaseModel):
    token: str
    webhook_secret: Optional[str] = None


class DeploymentSecretsFarcaster(BaseModel):
    mnemonic: str
    neynar_webhook_secret: str


class DeploymentSecretsTwitter(BaseModel):
    user_id: str
    bearer_token: str
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str


class DeploymentSecrets(BaseModel):
    discord: DeploymentSecretsDiscord | None = None
    telegram: DeploymentSecretsTelegram | None = None
    farcaster: DeploymentSecretsFarcaster | None = None
    twitter: DeploymentSecretsTwitter | None = None


class DeploymentConfig(BaseModel):
    discord: DeploymentSettingsDiscord | None = None
    telegram: DeploymentSettingsTelegram | None = None
    farcaster: DeploymentSettingsFarcaster | None = None
    twitter: DeploymentSettingsTwitter | None = None


@Collection("deployments")
class Deployment(Document):
    agent: ObjectId
    user: ObjectId
    platform: ClientType
    valid: Optional[bool] = None
    secrets: Optional[DeploymentSecrets]
    config: Optional[DeploymentConfig]

    def __init__(self, **data):
        # Convert string to ClientType enum if needed
        if "platform" in data and isinstance(data["platform"], str):
            data["platform"] = ClientType(data["platform"])
        super().__init__(**data)

    def model_dump(self, *args, **kwargs):
        """Override model_dump to convert enum to string for MongoDB"""
        data = super().model_dump(*args, **kwargs)
        if "platform" in data and isinstance(data["platform"], ClientType):
            data["platform"] = data["platform"].value
        return data

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent", 1), ("platform", 1)], unique=True)

    # @classmethod
    # def find(cls, query):
    #     """Find all deployments matching the query"""
    #     collection = get_collection(cls.collection_name)
    #     return [cls(**doc) for doc in collection.find(query)]

    @classmethod
    def delete_deployment(cls, agent_id: ObjectId, platform: str):
        """Delete a deployment record"""
        collection = get_collection(cls.collection_name)
        collection.delete_one({"agent": agent_id, "platform": platform})

    @classmethod
    def create_twitter_deployment(cls, agent_id: ObjectId, credentials: dict):
        """Create a new Twitter deployment with encrypted credentials"""
        deployment = cls(
            agent=agent_id,
            platform="twitter",
            twitter_bearer_token=credentials.get("TWITTER_BEARER_TOKEN"),
            twitter_consumer_key=credentials.get("TWITTER_CONSUMER_KEY"),
            twitter_consumer_secret=credentials.get("TWITTER_CONSUMER_SECRET"),
            twitter_user_id=credentials.get("TWITTER_USER_ID"),
            twitter_access_token=credentials.get("TWITTER_ACCESS_TOKEN"),
            twitter_access_token_secret=credentials.get("TWITTER_ACCESS_TOKEN_SECRET"),
        )
        deployment.save()
        return deployment

    def get_twitter_credentials(self) -> dict:
        """Get decrypted Twitter credentials"""
        if self.platform != "twitter":
            raise ValueError("Not a Twitter deployment")

        return {
            "TWITTER_BEARER_TOKEN": self.twitter_bearer_token.get_secret_value()
            if self.twitter_bearer_token
            else None,
            "TWITTER_CONSUMER_KEY": self.twitter_consumer_key.get_secret_value()
            if self.twitter_consumer_key
            else None,
            "TWITTER_CONSUMER_SECRET": self.twitter_consumer_secret.get_secret_value()
            if self.twitter_consumer_secret
            else None,
            "TWITTER_USER_ID": self.twitter_user_id,
            "TWITTER_ACCESS_TOKEN": self.twitter_access_token.get_secret_value()
            if self.twitter_access_token
            else None,
            "TWITTER_ACCESS_TOKEN_SECRET": self.twitter_access_token_secret.get_secret_value()
            if self.twitter_access_token_secret
            else None,
        }


def authenticate_modal_key() -> bool:
    token_id = os.getenv("MODAL_DEPLOYER_TOKEN_ID")
    token_secret = os.getenv("MODAL_DEPLOYER_TOKEN_SECRET")
    subprocess.run(
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


def get_container_name(agent_id: str, agent_key: str, platform: str, env: str) -> str:
    return f"{agent_id}-{agent_key}-{platform}-{env}"


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
            key = key.upper()
            value = str(value).strip().strip("'\"")
            cmd_parts.append(f"{key}={value}")
    cmd_parts.extend(["-e", DEPLOYMENT_ENV_NAME, "--force"])

    subprocess.run(cmd_parts)


def clone_repo(temp_dir: str, branch: str = None):
    """Clone the eve repository to a temporary directory"""
    branch = branch or REPO_BRANCH
    print(f"Cloning repo {REPO_URL} to {temp_dir} on branch {branch}")
    subprocess.run(
        ["git", "clone", "-b", branch, "--single-branch", REPO_URL, temp_dir],
        check=True,
    )


def prepare_client_file(
    file_path: str,
    agent_id: str,
    agent_key: str,
    platform: str,
    secrets: DeploymentSecrets,
    env: str,
) -> None:
    """Modify the client file to use correct secret name"""
    with open(file_path, "r") as f:
        content = f.read()

    repo_root = Path(file_path).parent.parent.parent.parent
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

    # Fix environment variable replacement
    modified_content = modified_content.replace(
        '.env({"AGENT_ID": ""})',
        f'.env({{"AGENT_ID": "{agent_id}"}})',  # Note the double curly braces
    )

    if platform == "discord":
        discord_token = secrets.discord.token
        if not discord_token:
            raise Exception("Discord token not found")
        modified_content = modified_content.replace(
            '.env({"CLIENT_DISCORD_TOKEN": ""})',
            f'.env({{"CLIENT_DISCORD_TOKEN": "{discord_token}"}})',
        )
    elif platform == "telegram":
        telegram_token = secrets.telegram.token
        if not telegram_token:
            raise Exception("Telegram token not found")
        modified_content = modified_content.replace(
            '.env({"CLIENT_TELEGRAM_TOKEN": ""})',
            f'.env({{"CLIENT_TELEGRAM_TOKEN": "{telegram_token}"}})',
        )
    elif platform == "farcaster":
        pass
    elif platform == "twitter":
        pass

    print(modified_content)

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "modal_client.py"
    with open(temp_file, "w") as f:
        f.write(modified_content)

    return str(temp_file)


async def modify_secrets(secrets: DeploymentSecrets, platform: ClientType):
    if platform == ClientType.TELEGRAM:
        webhook_secret = python_secrets.token_urlsafe(32)
        secrets.telegram.webhook_secret = webhook_secret
    elif platform == ClientType.DISCORD:
        headers = {"Authorization": f"Bot {secrets.discord.token}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://discord.com/api/v10/users/@me", headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception("Invalid Discord token")

                # Get application ID if not provided
                if not secrets.discord.application_id:
                    bot_data = await response.json()
                    application_id = bot_data.get("id")
                    print(application_id)
                    if application_id:
                        secrets.discord.application_id = application_id
                        # Create OAuth URL with the same permissions integer
                        permissions_integer = "309237771264"
                        oauth_url = f"https://discord.com/oauth2/authorize?client_id={application_id}&permissions={permissions_integer}&integration_type=0&scope=bot"
                        # Update the deployment config with the OAuth URL and client ID
                        return secrets, {
                            "oauth_client_id": application_id,
                            "oauth_url": oauth_url,
                            "valid": True,
                        }
    return secrets, None


async def deploy_client(
    deployment: Deployment,
    agent: Agent,
    platform: ClientType,
    secrets: DeploymentSecrets,
    env: str,
    repo_branch: str = None,
):
    if platform in modal_platforms:
        deploy_client_modal(agent, platform, secrets, env, repo_branch)

    elif platform == ClientType.DISCORD:
        await deploy_client_discord(deployment, secrets)

    elif platform == ClientType.TELEGRAM:
        await deploy_client_telegram(secrets, str(deployment.id))

    elif platform == ClientType.FARCASTER:
        pass

    elif platform == ClientType.TWITTER:
        await deploy_client_twitter(deployment, secrets)

    else:
        raise Exception(f"Unsupported platform: {platform}")


def deploy_client_modal(
    agent: Agent,
    platform: ClientType,
    secrets: DeploymentSecrets,
    env: str,
    repo_branch: str = None,
):
    agent_id = str(agent.id)
    agent_key = agent.username
    platform_value = platform.value
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repo using provided branch or default
        branch = repo_branch or REPO_BRANCH
        clone_repo(temp_dir, branch)

        # Check for client file in the cloned repo
        client_path = os.path.join(
            temp_dir, "eve", "clients", platform_value, "modal_client.py"
        )
        if os.path.exists(client_path):
            # Modify the client file to use the correct secret name
            temp_file = prepare_client_file(
                client_path, agent_id, agent_key, platform_value, secrets, env
            )
            app_name = get_container_name(agent_id, agent_key, platform_value, env)

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


async def deploy_client_discord(deployment: Deployment, secrets: DeploymentSecrets):
    """
    For HTTP-based Discord bots, we verify the token and notify the gateway service via Ably.
    """
    try:
        ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        channel = ably_client.channels.get(f"discord-gateway-{db}")

        await channel.publish(
            "command", {"command": "start", "deployment_id": str(deployment.id)}
        )
        print(f"Sent start command for deployment {deployment.id} via Ably")
    except Exception as e:
        print(f"Failed to notify gateway service: {e}")
        raise Exception("Failed to start gateway client")


async def deploy_client_telegram(secrets: DeploymentSecrets, deployment_id: str):
    from telegram import Bot

    webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/telegram"

    # Update bot webhook
    response = await Bot(secrets.telegram.token).set_webhook(
        url=webhook_url,
        secret_token=secrets.telegram.webhook_secret,
        drop_pending_updates=True,
        max_connections=100,
    )

    if not response:
        raise Exception("Failed to set Telegram webhook")

    # Notify gateway about the new deployment
    try:
        ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        channel = ably_client.channels.get(f"discord-gateway-{db}")

        await channel.publish(
            "command",
            {
                "command": "register_telegram",
                "deployment_id": deployment_id,
                "token": secrets.telegram.token,
            },
        )
        print(
            f"Sent Telegram registration command for deployment {deployment_id} via Ably"
        )
    except Exception as e:
        print(f"Failed to notify gateway service for Telegram: {e}")


async def deploy_client_twitter(deployment: Deployment, secrets: DeploymentSecrets):
    agent = Agent.from_mongo(deployment.agent)
    if not agent:
        raise Exception("Agent not found")

    # Add Twitter tools to agent's tools
    if not agent.tools:
        agent.tools = {}

    agent.tools["tweet"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }

    agent.tools["mentions"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }

    agent.tools["search"] = {
        "parameters": {"agent": {"default": str(agent.id), "hide_from_agent": True}}
    }

    agent.save()


async def stop_client(agent: Agent, platform: ClientType):
    """Stop a Modal client. For Discord HTTP, notify the gateway service via Ably."""
    if platform == ClientType.DISCORD:
        # Find the deployment
        deployment = Deployment.load(agent=agent.id, platform=platform.value)
        if deployment:
            await stop_client_discord(deployment)
    elif platform == ClientType.TELEGRAM:
        # Find the deployment
        deployment = Deployment.load(agent=agent.id, platform=platform.value)
        if deployment:
            await stop_client_telegram(deployment)
    elif platform == ClientType.TWITTER:
        await stop_client_twitter(agent)
    elif platform in modal_platforms:
        await stop_client_modal(agent, platform)


async def stop_client_discord(deployment: Deployment):
    if deployment:
        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {"command": "stop", "deployment_id": str(deployment.id)},
            )
            print(f"Sent stop command for deployment {deployment.id} via Ably")
        except Exception as e:
            print(f"Failed to notify gateway service: {e}")


async def stop_client_modal(agent: Agent, platform: ClientType):
    container_name = get_container_name(
        agent.id, agent.username, platform.value, db.lower()
    )
    result = subprocess.run(
        [
            "modal",
            "app",
            "stop",
            container_name,
            "-e",
            DEPLOYMENT_ENV_NAME,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Failed to stop client: {result.stderr}")


async def stop_client_twitter(agent: Agent):
    if agent.tools:
        for tool in ["tweet", "mentions", "search"]:
            if tool in agent.tools:
                agent.tools.pop(tool)
        agent.save()


# Add the new function for stopping Telegram client
async def stop_client_telegram(deployment: Deployment):
    if deployment:
        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {"command": "unregister_telegram", "deployment_id": str(deployment.id)},
            )
            print(
                f"Sent unregister command for Telegram deployment {deployment.id} via Ably"
            )
        except Exception as e:
            print(f"Failed to notify gateway service for Telegram unregistration: {e}")
