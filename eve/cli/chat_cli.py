import os
import sys
import json
import time
import re
import logging
import click
import asyncio
import traceback
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .. import load_env
from ..llm import async_prompt_thread, UserMessage, UpdateType
from ..eden_utils import prepare_result, dump_json
from ..agent import Agent
from ..auth import get_my_eden_user


async def async_chat(agent_name, new_thread=True, debug=False):
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    user = get_my_eden_user()
    agent = Agent.load(agent_name)

    key = f"cli_{str(agent.name)}_{str(user.id)}"
    if not new_thread:
        key += f"_{int(time.time())}"

    thread = agent.request_thread(key=key)
    tools = agent.get_tools()

    chat_string = f"Chat with {agent.name}".center(36)
    console = Console()
    console.print("\n[bold blue]╭────────────────────────────────────╮")
    console.print(f"[bold blue]│{chat_string}│")
    console.print("[bold blue]╰────────────────────────────────────╯\n")
    # console.print("[dim]Type 'escape' to exit the chat[/dim]\n")

    while True:
        try:
            console.print("[bold yellow]You [dim]→[/dim] ", end="")
            message_input = input("\033[93m")

            print()

            metadata_pattern = r"\{.*?\}"
            attachments_pattern = r"\[.*?\]"
            attachments_match = re.search(attachments_pattern, message_input)
            attachments = json.loads(attachments_match.group(0)) if attachments_match else []
            content = re.sub(metadata_pattern, "", message_input)
            content = re.sub(attachments_pattern, "", content).strip()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=None)

                with open(os.devnull, "w") as devnull:
                    original_stdout = sys.stdout
                    if not debug:                        
                        sys.stdout = devnull

                    async for update in async_prompt_thread(
                        user=user,
                        agent=agent,
                        thread=thread,
                        user_messages=UserMessage(
                            content=content, 
                            attachments=attachments
                        ),
                        tools=tools,
                        force_reply=True,
                    ):
                        sys.stdout = original_stdout

                        progress.update(task)
                        if update.type == UpdateType.ASSISTANT_MESSAGE:
                            console.print(
                                "[bold green]Eve [dim]→[/dim] [green]"
                                + update.message.content
                            )
                            print()
                        elif update.type == UpdateType.TOOL_COMPLETE:
                            result = prepare_result(update.result.get("result"))
                            console.print(
                                "[bold cyan]🔧 [dim]" + update.tool_name + "[/dim]"
                            )
                            # formatted_result = json.dumps(result, indent=2)
                            formatted_result = dump_json(result, indent=2)
                            formatted_result = re.sub(
                                r'(https?://[^\s"]+)',
                                lambda m: f"[link={m.group(1)}]{m.group(1)}[/link]",
                                formatted_result,
                            )
                            console.print("[cyan]" + formatted_result)
                            print()
                        elif update.type == UpdateType.ERROR:
                            print(update)
                            console.print(
                                f"[bold red]❌ Error: [red]{str(update.error)}[/red]"
                            )
                            print()

                        if not debug:
                            sys.stdout = devnull

                    sys.stdout = original_stdout

        except KeyboardInterrupt:
            console.print("\n[dim]Chat interrupted. Goodbye! 👋[/dim]\n")
            break


@click.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.option(
    "--thread", 
    type=str, 
    help="Thread id"
)
@click.option(
    "--debug", 
    is_flag=True, 
    default=False, 
    help="Debug mode"
)
@click.argument("agent", required=True, default="eve")
def chat(db: str, thread: str, agent: str, debug: bool):
    """Chat with an agent"""

    load_env(db)

    try:
        asyncio.run(async_chat(agent, thread, debug))
    except Exception as e:
        click.echo(click.style(f"Failed to chat with {agent}:", fg="red"))
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        traceback.print_exc(file=sys.stdout)
