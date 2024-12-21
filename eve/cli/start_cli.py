import sys
import yaml
import click
import traceback
import multiprocessing
from pathlib import Path

from ..models import ClientType
from ..clients.discord.client import start as start_discord
from ..clients.telegram.client import start as start_telegram
from ..clients.farcaster.client import start as start_farcaster


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.option(
    "--env",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to environment file",
)
@click.option(
    "--platforms",
    type=click.Choice(
        [
            ClientType.DISCORD.value,
            ClientType.TELEGRAM.value,
            ClientType.FARCASTER.value,
            ClientType.LOCAL.value,
        ]
    ),
    multiple=True,
    help="Platforms to start",
)
def start(agent: str, db: str, env: str, platforms: tuple):
    """Start one or more clients from yaml files"""
    try:
        agent_dir = Path(__file__).parent.parent / "agents" / agent
        env_path = agent_dir / ".env"
        yaml_path = agent_dir / "api.yaml"

        db = db.upper()
        env_path = env or env_path
        # Set the environment variable

        clients_to_start = {}

        if platforms:
            clients_to_start = {
                ClientType(platform): yaml_path for platform in platforms
            }
        else:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            if "clients" in config:
                for client_type, settings in config["clients"].items():
                    if settings.get("enabled", False):
                        clients_to_start[ClientType(client_type)] = yaml_path

        if not clients_to_start:
            click.echo(click.style("No enabled clients found in yaml files", fg="red"))
            return

        click.echo(
            click.style(f"Starting {len(clients_to_start)} clients...", fg="blue")
        )

        # Start discord and telegram first, local client last
        processes = []
        for client_type, yaml_path in clients_to_start.items():
            try:
                if client_type == ClientType.DISCORD:
                    p = multiprocessing.Process(
                        target=start_discord, args=(env_path, db)
                    )
                elif client_type == ClientType.TELEGRAM:
                    p = multiprocessing.Process(
                        target=start_telegram, args=(env_path, db)
                    )
                elif client_type == ClientType.FARCASTER:
                    p = multiprocessing.Process(
                        target=start_farcaster, args=(env_path, db)
                    )

                p.start()
                processes.append(p)
                click.echo(
                    click.style(f"Started {client_type.value} client", fg="green")
                )
            except Exception as e:
                click.echo(
                    click.style(
                        f"Failed to start {client_type.value} client:", fg="red"
                    )
                )
                click.echo(click.style(f"Error: {str(e)}", fg="red"))
                traceback.print_exc(file=sys.stdout)

        # Wait for other processes
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            click.echo(click.style("\nShutting down clients...", fg="yellow"))
            for p in processes:
                p.terminate()
                p.join()

    except Exception as e:
        click.echo(click.style("Failed to start clients:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)
