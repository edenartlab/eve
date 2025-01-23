import sys
import click
import traceback
import subprocess
from pathlib import Path
from dotenv import dotenv_values
import shutil

from eve.deploy import DEPLOYMENT_ENV_NAME, prepare_client_file
from eve.agent import Agent
from .. import load_env

root_dir = Path(__file__).parent.parent.parent


def ensure_modal_env_exists():
    """Create the Modal environment if it doesn't exist"""
    # List existing environments
    result = subprocess.run(
        ["rye", "run", "modal", "environment", "list"],
        capture_output=True,
        text=True,
    )

    # Check if our environment exists
    if DEPLOYMENT_ENV_NAME not in result.stdout:
        click.echo(
            click.style(
                f"Creating Modal environment: {DEPLOYMENT_ENV_NAME}", fg="green"
            )
        )
        subprocess.run(
            ["rye", "run", "modal", "environment", "create", DEPLOYMENT_ENV_NAME]
        )
    else:
        click.echo(
            click.style(
                f"Using existing Modal environment: {DEPLOYMENT_ENV_NAME}", fg="blue"
            )
        )


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
        ensure_modal_env_exists()
        load_env(db)
        env = "stage" if db == "STAGE" else "prod"

        # Get agent info from DB
        agent_obj = Agent.load(agent)
        if not agent_obj:
            click.echo(click.style(f"Agent not found in DB: {agent}", fg="yellow"))
            return

        # Deploy the specified client
        client_path = root_dir / f"eve/clients/{platform}/modal_client.py"
        if client_path.exists():
            try:
                temp_file = prepare_client_file(str(client_path), agent, env)
                app_name = f"{agent}-{platform}-{env}"

                subprocess.run(
                    [
                        "rye",
                        "run",
                        "modal",
                        "deploy",
                        "--name",
                        app_name,
                        temp_file,
                        "-e",
                        DEPLOYMENT_ENV_NAME,
                    ]
                )
            finally:
                if temp_file:
                    shutil.rmtree(Path(temp_file).parent)
        else:
            click.echo(
                click.style(
                    f"Warning: Client modal file not found: {client_path}", fg="yellow"
                )
            )

    except Exception as e:
        click.echo(click.style("Failed to deploy client:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def secrets(agent: str, db: str):
    """Create Modal secrets for an agent."""
    try:
        ensure_modal_env_exists()
        load_env(db)
        env = "stage" if db == "STAGE" else "prod"

        # Get agent info from DB
        agent_obj = Agent.load(agent)
        if not agent_obj:
            click.echo(click.style(f"Agent not found in DB: {agent}", fg="yellow"))
            return

        # Create secrets if .env exists
        env_file = root_dir / "eve" / "agents" / env / agent / ".env"
        if env_file.exists():
            click.echo(click.style(f"Creating secrets for: {agent}", fg="green"))
            client_secrets = dotenv_values(env_file)

            if not client_secrets:
                click.echo(click.style(f"No secrets found for {agent}", fg="yellow"))
                return

            cmd_parts = [
                "rye",
                "run",
                "modal",
                "secret",
                "create",
                f"{agent}-secrets-{env}",
            ]
            for key, value in client_secrets.items():
                if value is not None:
                    value = str(value).strip().strip("'\"")
                    cmd_parts.append(f"{key}={value}")
            cmd_parts.extend(["-e", DEPLOYMENT_ENV_NAME, "--force"])

            subprocess.run(cmd_parts)
        else:
            click.echo(click.style(f"No .env file found at {env_file}", fg="yellow"))

    except Exception as e:
        click.echo(click.style("Failed to create secrets:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)
