import click
from ..s3 import upload_file
from .. import load_env

@click.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.argument("files", nargs=-1, required=False)
def upload(db: str, files: tuple):
    """Upload agents to mongo"""
    
    load_env(db)
    
    for file in files:
        try:
            result = upload_file(file)
            url = result[0]
            click.echo(
                click.style(
                    f"\nUploaded: {file.split('/')[-1]} to {url}", fg="green", bold=True
                )
            )
        except Exception as e:
            click.echo(
                click.style(
                    f"Failed to upload file {file}: {e}", fg="red"
                )
            )
