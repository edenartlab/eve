import click
from .tool_cli import tool
from .chat_cli import chat
from .start_cli import start, api
from .upload_cli import upload
from .deploy_cli import deploy, configure, stop


@click.group()
def cli():
    """Eve CLI"""
    pass


cli.add_command(tool)
cli.add_command(chat)
cli.add_command(start)
cli.add_command(api)
cli.add_command(upload)
cli.add_command(deploy)
# cli.add_command(redeploy)
cli.add_command(stop)
cli.add_command(configure)

__all__ = ["deploy", "start"]  # Add other CLI commands as needed
