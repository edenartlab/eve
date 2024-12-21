import sys
import yaml
import click
import traceback
import subprocess
from pathlib import Path
from dotenv import dotenv_values

root_dir = Path(__file__).parent.parent.parent
ENV_NAME = "deployments"


def modify_client_file(file_path: str, agent_key: str) -> None:
    """Modify the client file to use correct secret name and fix pyproject path"""
    with open(file_path, "r") as f:
        content = f.read()

    # Get the repo root directory
    repo_root = root_dir.absolute()
    pyproject_path = repo_root / "pyproject.toml"

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


def create_secrets(agent_key: str, secrets_dict: dict):
    if not secrets_dict:
        click.echo(click.style(f"No secrets found for {agent_key}", fg="yellow"))
        return

    cmd_parts = [
        "rye",
        "run",
        "modal",
        "secret",
        "create",
        f"{agent_key}-client-secrets",
    ]
    for key, value in secrets_dict.items():
        if value is not None:
            value = str(value).strip().strip("'\"")
            cmd_parts.append(f"{key}={value}")
    cmd_parts.extend(["-e", ENV_NAME, "--force"])

    subprocess.run(cmd_parts)


def deploy_client(agent_key: str, client_name: str):
    client_path = root_dir / f"eve/clients/{client_name}/modal_client.py"
    if client_path.exists():
        # Modify the client file to use the correct secret name
        modify_client_file(str(client_path), agent_key)
        app_name = f"{agent_key}-client-{client_name}"
        subprocess.run(
            [
                "rye",
                "run",
                "modal",
                "deploy",
                "--name",
                app_name,
                str(client_path),
                "-e",
                ENV_NAME,
            ]
        )
    else:
        click.echo(
            click.style(
                f"Warning: Client modal file not found: {client_path}", fg="yellow"
            )
        )


def process_agent(agent_path: Path):
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
        create_secrets(agent_key, client_secrets)

    # Deploy each client
    for deployment in agent_config["deployments"]:
        click.echo(click.style(f"Deploying client: {deployment}", fg="green"))
        deploy_client(agent_key, deployment)


@click.command()
@click.argument("agent", nargs=1, required=True)
def deploy(agent: str):
    """Deploy Modal agents"""
    try:
        agents_dir = root_dir / "eve/agents"
        agent_path = agents_dir / agent / "api.yaml"
        if agent_path.exists():
            process_agent(agent_path)
        else:
            click.echo(
                click.style(f"Warning: Agent file not found: {agent_path}", fg="yellow")
            )

    except Exception as e:
        click.echo(click.style("Failed to deploy agents:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    deploy()
