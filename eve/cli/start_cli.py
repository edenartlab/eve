import sys
import traceback
from pathlib import Path

import click

from eve.agent.session.models import ClientType

from ..agent import Agent


@click.command()
@click.argument("agent", nargs=1, required=True)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.option(
    "--platforms",
    type=click.Choice(
        [
            ClientType.DISCORD.value,
            ClientType.TELEGRAM.value,
            ClientType.FARCASTER.value,
        ]
    ),
    multiple=True,
    required=True,
    help="Platforms to start",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Run locally",
)
def start(agent: str, db: str, platforms: tuple, local: bool):
    """Start one or more clients from database configuration"""
    try:
        env = db.lower()

        # Get agent from DB
        agent_obj = Agent.load(agent)
        if not agent_obj:
            click.echo(click.style(f"Agent not found in DB: {agent}", fg="red"))
            return

        env_path = Path(__file__).parent.parent / "agents" / env / agent / ".env"
        if not env_path.exists():
            click.echo(click.style(f"No .env file found at {env_path}", fg="yellow"))
            return

        clients_to_start = {}
        for platform in platforms:
            clients_to_start[ClientType(platform)] = True

        click.echo(
            click.style(f"Starting {len(clients_to_start)} clients...", fg="blue")
        )

        # Start discord and telegram first, local client last
        processes = []

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
    default="127.0.0.1",
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
    default=True,
    help="Enable auto-reload on code changes",
)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.option(
    "--remote_debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
def api(host: str, port: int, reload: bool, db: str, remote_debug: bool):
    """Start the Eve API server"""
    import os

    import uvicorn

    # Set the LOCAL_DEBUG environment variable if the flag is set
    if not remote_debug:
        os.environ["LOCAL_DEBUG"] = "True"
    else:
        os.environ["LOCAL_DEBUG"] = "False"

    click.echo(
        click.style(f"Starting API server on {host}:{port} with DB={db}...", fg="blue")
    )

    uvicorn.run(
        "eve.api.api:web_app",
        host=host,
        port=port,
        reload=reload,
        app_dir=str(Path(__file__).parent.parent.parent),
    )
