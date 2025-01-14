import sys
import yaml
import click
import traceback
import subprocess
from pathlib import Path
from dotenv import dotenv_values
import shutil

from eve.deploy import DEPLOYMENT_ENV_NAME, prepare_client_file

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


def create_secrets(agent_key: str, secrets_dict: dict, env: str):
    if not secrets_dict:
        click.echo(click.style(f"No secrets found for {agent_key}", fg="yellow"))
        return

    cmd_parts = [
        "rye",
        "run",
        "modal",
        "secret",
        "create",
        f"{agent_key}-secrets-{env}",
    ]
    for key, value in secrets_dict.items():
        if value is not None:
            value = str(value).strip().strip("'\"")
            cmd_parts.append(f"{key}={value}")
    cmd_parts.extend(["-e", DEPLOYMENT_ENV_NAME, "--force"])

    subprocess.run(cmd_parts)


def deploy_client(agent_key: str, client_name: str, env: str):
    client_path = root_dir / f"eve/clients/{client_name}/modal_client.py"
    if client_path.exists():
        try:
            # Create a temporary modified version of the client file
            temp_file = prepare_client_file(str(client_path), agent_key, env)
            app_name = f"{agent_key}-{client_name}-{env}"

            # Deploy using the temporary file
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
            # Clean up temporary directory
            if temp_file:
                shutil.rmtree(Path(temp_file).parent)
    else:
        click.echo(
            click.style(
                f"Warning: Client modal file not found: {client_path}", fg="yellow"
            )
        )


def get_deployable_agents(env: str):
    """Find all agents that have both .env and deployments configured"""
    agents_dir = root_dir / "eve" / "agents" / env
    deployable = []

    for agent_dir in agents_dir.glob("*"):
        if not agent_dir.is_dir():
            continue

        yaml_path = agent_dir / "api.yaml"
        env_path = agent_dir / ".env"

        if not yaml_path.exists() or not env_path.exists():
            continue

        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
                if config.get("deployments"):
                    deployable.append(agent_dir.name)
        except Exception as e:
            click.echo(
                click.style(
                    f"Error reading config for {agent_dir.name}: {str(e)}", fg="yellow"
                )
            )

    return deployable


def process_agent(agent_path: Path, env: str):
    with open(agent_path) as f:
        agent_config = yaml.safe_load(f)

    if not agent_config.get("deployments"):
        click.echo(click.style(f"No deployments found in {agent_path}", fg="yellow"))
        return

    agent_key = agent_path.parent.name
    click.echo(click.style(f"Processing agent: {agent_key}", fg="blue"))

    # Create secrets if .env exists
    env_file = agent_path.parent / ".env"
    if env_file.exists():
        click.echo(click.style(f"Creating secrets for: {agent_key}", fg="green"))
        client_secrets = dotenv_values(env_file)
        create_secrets(agent_key, client_secrets, env)

    # Deploy each client
    for deployment in agent_config["deployments"]:
        click.echo(click.style(f"Deploying client: {deployment}", fg="green"))
        deploy_client(agent_key, deployment, env)


@click.command()
@click.argument("agent", nargs=1, required=False)
@click.option("--all", is_flag=True, help="Deploy all configured agents")
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def deploy(agent: str, all: bool, db: str):
    """Deploy Modal agents. Use --all to deploy all configured agents."""
    try:
        # Ensure Modal environment exists
        ensure_modal_env_exists()
        load_env(db)
        
        env_map = {
            "STAGE": "stage",
            "PROD": "prod",
            "WEB3-STAGE": "web3-stage",
            "WEB3-PROD": "web3-prod"
        }
        env = env_map.get(db, "stage")
        
        if all:
            agents = get_deployable_agents(env)
            if not agents:
                click.echo(
                    click.style(
                        f"No deployable agents found in {env} environment",
                        fg="yellow",
                    )
                )
                return

            click.echo(
                click.style(
                    f"Found {len(agents)} deployable agents: {', '.join(agents)}",
                    fg="green",
                )
            )

            for agent_name in agents:
                click.echo(click.style(f"\nProcessing agent: {agent_name}", fg="blue"))
                agent_path = root_dir / "eve" / "agents" / env / agent_name / "api.yaml"
                process_agent(agent_path, env)

        else:
            if not agent:
                raise click.UsageError("Please provide an agent name or use --all")

            agent_path = root_dir / "eve" / "agents" / env / agent / "api.yaml"
            if agent_path.exists():
                process_agent(agent_path, env)
            else:
                click.echo(
                    click.style(
                        f"Warning: Agent file not found: {agent_path}", fg="yellow"
                    )
                )

    except Exception as e:
        click.echo(click.style("Failed to deploy agents:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    deploy()
