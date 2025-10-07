import click
from .tool_cli import tool
from .chat_cli import chat
from .start_cli import start, api
from .upload_cli import upload
from .export_cli import export
from .deploy_cli import deploy, configure, stop
from .lookup_cli import lookup


@click.group()
def cli():
    """Eve CLI"""
    pass


cli.add_command(tool)
cli.add_command(chat)
cli.add_command(start)
cli.add_command(api)
cli.add_command(upload)
cli.add_command(export)
cli.add_command(deploy)
cli.add_command(stop)
cli.add_command(configure)
cli.add_command(lookup)

__all__ = ["deploy", "start"]  # Add other CLI commands as needed
