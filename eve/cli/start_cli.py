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
# @click.option(
#     "--env",
#     type=click.Path(exists=True, resolve_path=True),
#     help="Path to environment file",
# )
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
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Run locally",
)
def start(agent: str, db: str, platforms: tuple, local: bool):
    """Start one or more clients from yaml files"""
    try:
        db = db.upper()
        agent_dir = Path(__file__).parent.parent / "agents" / db.lower() / agent
        env_path = agent_dir / ".env"
        yaml_path = agent_dir / "api.yaml"

        print("ENV PATH", env_path)
        print("YAML PATH", yaml_path)

        
        # env_path = env or env_path

        print("ENV PATH", env_path)
        print("YAML PATH", yaml_path)

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

        print("CLIENTS TO START", clients_to_start)

        if not clients_to_start:
            click.echo(click.style("No enabled clients found in yaml files", fg="red"))
            return

        click.echo(
            click.style(f"Starting {len(clients_to_start)} clients...", fg="blue")
        )

        # Start discord and telegram first, local client last
        processes = []
        for client_type, yaml_path in clients_to_start.items():
            print("CLIENT TYPE", client_type)
            print("YAML PATH", yaml_path)
            try:
                print("stat discord", env_path, db, local)
                if client_type == ClientType.DISCORD:
                    p = multiprocessing.Process(
                        target=start_discord, args=(env_path, db, local)
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


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to run the server on",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload on code changes",
)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
def api(host: str, port: int, reload: bool, db: str):
    """Start the Eve API server"""
    import uvicorn
    import os

    # Set the DB environment variable
    os.environ["DB"] = db.upper()
    
    click.echo(click.style(f"Starting API server on {host}:{port} with DB={db}...", fg="blue"))

    # Adjusted the import path to look one directory up
    uvicorn.run(
        "eve.api:web_app",
        host=host,
        port=port,
        reload=reload,
        app_dir=str(Path(__file__).parent.parent.parent),
    )
