import click
from .tool_cli import tool
from .start_cli import start, api
from .upload_cli import upload
from .export_cli import export
from .lookup_cli import lookup


@click.group()
def cli():
    """Eve CLI"""
    pass


cli.add_command(tool)
cli.add_command(start)
cli.add_command(api)
cli.add_command(upload)
cli.add_command(export)
cli.add_command(lookup)

__all__ = ["start"]  # Add other CLI commands as needed
