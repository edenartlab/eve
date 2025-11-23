from pathlib import Path

import click


@click.command()
@click.option(
    "--agent",
    help="Agent name to filter sessions by (only export sessions containing this agent)",
)
@click.option(
    "--resume-dir",
    help="Resume directory",
    required=False,
    default=None,
)
@click.argument(
    "username",
    required=False,
)
def export(agent: str, username: str, resume_dir: Path = None):
    """Export user data to JSON and HTML files"""

    try:
        from ..utils.export_utils import export_agent_creations, export_user_data

        export_dir = export_user_data(
            username=username, agentname=agent, export_dir=Path(resume_dir)
        )

        export_agent_creations(
            username=username,
            agentname=agent,
            export_dir=Path(export_dir) / f"{agent}_creations",
        )

        # Get the actual username used (in case it was auto-detected)
        display_user = username if username else "current user"

        click.echo(
            click.style(
                f"\nExport completed successfully for {display_user}",
                fg="green",
                bold=True,
            )
        )
        if agent:
            click.echo(click.style(f"Filtered by agent: {agent}", fg="blue"))
        click.echo(click.style(f"Export directory: {export_dir}", fg="cyan"))
    except Exception as e:
        click.echo(click.style(f"Failed to export data: {e}", fg="red"))
