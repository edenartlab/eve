import sys
import click
import traceback
import multiprocessing
from pathlib import Path

from eve.agent.deployments import ClientType
from ..agent import Agent
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
        for client_type in clients_to_start:
            try:
                if client_type == ClientType.DISCORD:
                    p = multiprocessing.Process(
                        target=start_discord, args=(env_path, local)
                    )
                elif client_type == ClientType.TELEGRAM:
                    p = multiprocessing.Process(
                        target=start_telegram, args=(env_path, local)
                    )
                elif client_type == ClientType.FARCASTER:
                    p = multiprocessing.Process(
                        target=start_farcaster, args=(env_path, local)
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
def api(host: str, port: int, reload: bool, db: str):
    """Start the Eve API server"""
    import uvicorn

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
