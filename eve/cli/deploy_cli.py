import sys
import click
import traceback
import requests
from pathlib import Path
from dotenv import dotenv_values
import os

from eve.deploy import ClientType
from .. import load_env

root_dir = Path(__file__).parent.parent.parent


def get_api_url():
    api_url = os.getenv("EDEN_API_URL", "http://localhost:8000").rstrip("/")
    return api_url


def api_request(method, endpoint, json=None):
    """Make an API request with error handling"""
    api_url = get_api_url()
    api_key = os.getenv("EDEN_ADMIN_KEY")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.request(
        method, f"{api_url}{endpoint}", headers=headers, json=json
    )

    response.raise_for_status()
    return response.json()


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.argument("platform", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def deploy(agent: str, platform: str, db: str):
    """Deploy a Modal client for an agent."""
    try:
        load_env(db)

        api_request(
            "POST",
            "/deployments/create",
            {"agent_username": agent, "platform": platform},
        )

        click.echo(
            click.style(
                f"Successfully deployed {platform} client for {agent}", fg="green"
            )
        )

    except Exception as e:
        click.echo(click.style("Failed to deploy client:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)


# @click.command()
# @click.argument("agent", nargs=1, required=False)
# @click.option(
#     "--platform",
#     type=click.Choice([t.value for t in ClientType], case_sensitive=False),
#     help="Platform to redeploy (all platforms if not specified)",
# )
# @click.option(
#     "--db",
#     type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
#     default="STAGE",
#     help="DB to save against",
# )
# def redeploy(agent: str | None, platform: str | None, db: str):
#     """Redeploy Modal clients. If agent is not specified, redeploys all agents."""
#     try:
#         load_env(db)

#         if agent:
#             if platform:
#                 api_request(
#                     "POST",
#                     "/deployments/create",
#                     {"agent_username": agent, "platform": platform},
#                 )
#             else:
#                 # Deploy all platforms for this agent
#                 for p in ClientType:
#                     try:
#                         api_request(
#                             "POST",
#                             "/deployments/create",
#                             {"agent_username": agent, "platform": p.value},
#                         )
#                     except requests.exceptions.HTTPError as e:
#                         if (
#                             e.response.status_code != 404
#                         ):  # Ignore if deployment doesn't exist
#                             raise

#         click.echo(click.style("Successfully redeployed clients", fg="green"))

#     except Exception as e:
#         click.echo(click.style("Failed to redeploy clients:", fg="red"))
#         click.echo(click.style(f"Error: {str(e)}", fg="red"))
#         traceback.print_exc(file=sys.stdout)


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def configure(agent: str, db: str):
    """Configure agent deployment from .env file (both secrets and config)."""
    try:
        load_env(db)
        env = "stage" if db == "STAGE" else "prod"

        env_file = root_dir / "eve" / "agents" / env / agent / ".env"
        if env_file.exists():
            env_vars = dotenv_values(env_file)

            if not env_vars:
                click.echo(
                    click.style(f"No configuration found for {agent}", fg="yellow")
                )
                return

            # Extract secrets
            secrets = {
                "eden_api_key": env_vars.get("EDEN_API_KEY"),
                "client_discord_token": env_vars.get("CLIENT_DISCORD_TOKEN"),
                "client_telegram_token": env_vars.get("CLIENT_TELEGRAM_TOKEN"),
                "client_farcaster_mnemonic": env_vars.get("CLIENT_FARCASTER_MNEMONIC"),
                "client_farcaster_neynar_webhook_secret": env_vars.get(
                    "CLIENT_FARCASTER_NEYNAR_WEBHOOK_SECRET"
                ),
            }

            # Extract config
            config = {}
            if "DISCORD_CHANNEL_ALLOWLIST" in env_vars:
                channels = env_vars["DISCORD_CHANNEL_ALLOWLIST"].split(",")
                config["discord_channel_allowlist"] = channels
            if "TELEGRAM_TOPIC_ALLOWLIST" in env_vars:
                topics = env_vars["TELEGRAM_TOPIC_ALLOWLIST"].split(",")
                config["telegram_topic_allowlist"] = topics

            # Send configuration request
            api_request(
                "POST",
                "/deployments/configure",
                {
                    "agent_username": agent,
                    "secrets": {k: v for k, v in secrets.items() if v is not None},
                    "deployment_config": config if config else None,
                },
            )

            click.echo(click.style("Successfully configured deployment", fg="green"))
        else:
            click.echo(click.style(f"No .env file found at {env_file}", fg="yellow"))

    except Exception as e:
        click.echo(click.style("Failed to configure deployment:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.argument("platform", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def stop(agent: str, platform: str, db: str):
    """Stop a Modal client for an agent."""
    try:
        load_env(db)

        api_request(
            "POST",
            "/deployments/delete",
            {"agent_username": agent, "platform": platform},
        )

        click.echo(
            click.style(
                f"Successfully stopped {platform} client for {agent}", fg="green"
            )
        )

    except Exception as e:
        click.echo(click.style("Failed to stop client:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)
