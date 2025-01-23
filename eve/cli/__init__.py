import click
from .tool_cli import tool
from .agent_cli import agent
from .chat_cli import chat
from .start_cli import start, api
from .upload_cli import upload
from .deploy_cli import deploy, secrets


@click.group()
def cli():
    """Eve CLI"""
    pass


cli.add_command(tool)
cli.add_command(agent)
cli.add_command(chat)
cli.add_command(start)
cli.add_command(api)
cli.add_command(upload)
cli.add_command(deploy)
cli.add_command(secrets)

__all__ = ["deploy", "start"]  # Add other CLI commands as needed
