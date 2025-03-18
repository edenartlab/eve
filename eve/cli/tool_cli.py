import click
import random
import asyncio
import traceback

from .. import load_env
from ..eden_utils import save_test_results, prepare_result, dump_json, CLICK_COLORS
from ..auth import get_my_eden_user
from ..tool import Tool, get_tools_from_mongo, get_tools_from_api_files, get_api_files


api_tools_order = [
    "txt2img",
    "flux_dev",
    "flux_schnell",
    "layer_diffusion",
    "remix_flux_schnell",
    "remix",
    "inpaint",
    "flux_inpainting",
    "outpaint",
    "face_styler",
    "upscaler",
    "background_removal",
    "style_transfer",
    "storydiffusion",
    "xhibit_vton",
    "xhibit_remix",
    "beeple_ai",
    "txt2img_test",
    "sd3_txt2img",
    "HelloMeme_image",
    "HelloMeme_video",
    "flux_redux",
    "mars-id",
    "background_removal_video",
    "animate_3D",
    "style_mixing",
    "txt2vid",
    "vid2vid_sdxl",
    "img2vid",
    "video_upscaler",
    "frame_interpolation",
    "reel",
    "story",
    "texture_flow",
    "runway",
    "animate_3D_new",
    "mochi_preview",
    "lora_trainer",
    "flux_trainer",
    "news",
    "moodmix",
    "stable_audio",
    "musicgen",
    "legacy/create",
]


@click.group()
def tool():
    """Tool management commands"""
    pass


@tool.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.argument("names", nargs=-1, required=False)
def update(db: str, names: tuple):
    """Upload tools to mongo"""

    load_env(db)

    api_files = get_api_files()
    tools_order = {t: index for index, t in enumerate(api_tools_order)}

    if names:
        api_files = {k: v for k, v in api_files.items() if k in names}
    else:
        confirm = click.confirm(
            f"Update all {len(api_files)} tools on {db}?", default=False
        )
        if not confirm:
            return

    updated = 0
    for key, api_file in api_files.items():
        try:
            order = tools_order.get(key, len(api_tools_order))
            tool_ = Tool.from_yaml(api_file)
            tool_.save(order=order)
            click.echo(
                click.style(f"Updated tool {db}:{key} (order={order})", fg="green")
            )
            updated += 1
        except Exception as e:
            traceback.print_exc()
            click.echo(click.style(f"Failed to update tool {db}:{key}: {e}", fg="red"))

    click.echo(
        click.style(
            f"\nUpdated {updated} of {len(api_files)} tools", fg="blue", bold=True
        )
    )
    click.echo(
        click.style(
            f"\nUpdated {updated} of {len(api_files)} tools", fg="blue", bold=True
        )
    )


@tool.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to save against",
)
@click.argument("names", nargs=-1, required=True)
def remove(db: str, names: tuple):
    """Upload tools to mongo"""

    load_env(db)

    confirm = click.confirm(
        f"Are you sure you want to remove {len(names)} following tools from {db}?",
        default=False,
    )
    if not confirm:
        return

    deleted = 0
    for key in names:
        try:
            tool = Tool.load(key=key)
            tool.delete()
            click.echo(click.style(f"Deleted tool {db}:{key})", fg="red"))
            deleted += 1
        except Exception as e:
            traceback.print_exc()
            click.echo(click.style(f"Failed to delete tool {db}:{key}: {e}", fg="red"))

    click.echo(
        click.style(f"Deleted {deleted} of {len(names)} tools", fg="red", bold=True)
    )


@tool.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to load tools from if from mongo",
)
@click.argument("tool", required=False)
@click.pass_context
def run(ctx, tool: str, db: str):
    """Create with a tool. Args are passed as --key=value or --key value"""

    load_env(db)

    tool = Tool.load(key=tool)

    # Parse args
    args = dict()
    i = 0
    while i < len(ctx.args):
        arg = ctx.args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                args[key] = value
            elif i + 1 < len(ctx.args) and not ctx.args[i + 1].startswith("--"):
                value = ctx.args[i + 1]
                args[key] = value
                i += 1
            else:
                args[key] = True
        i += 1

    result = tool.run(args)
    color = random.choice(CLICK_COLORS)
    if result.get("error"):
        click.echo(
            click.style(
                f"\nFailed to test {tool.key}: {result['error']}",
                fg="red",
                bold=True,
            )
        )
    else:
        result = prepare_result(result)
        click.echo(
            click.style(f"\nResult for {tool.key}: {dump_json(result)}", fg=color)
        )

    return result


@tool.command()
@click.option(
    "--yaml",
    is_flag=True,
    default=False,
    help="Whether to load tools from yaml folders (default is from mongo)",
)
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to load tools from if from mongo",
)
@click.option(
    "--api",
    is_flag=True,
    help="Run tasks against API (If not set, will run tools directly)",
)
@click.option("--parallel", is_flag=True, help="Run tests in parallel threads")
@click.option("--save", is_flag=True, default=True, help="Save test results")
@click.option("--mock", is_flag=True, default=False, help="Mock test results")
@click.argument("tools", nargs=-1, required=False)
def test(
    tools: tuple, yaml: bool, db: str, api: bool, parallel: bool, save: bool, mock: bool
):
    """Test multiple tools with their test args"""

    load_env(db)

    async def async_test_tool(tool, api):
        color = random.choice(CLICK_COLORS)
        click.echo(click.style(f"\n\nTesting {tool.key}:", fg=color, bold=True))
        click.echo(click.style(f"Args: {dump_json(tool.test_args)}", fg=color))

        if api:
            user = get_my_eden_user()
            task = await tool.async_start_task(
                user_id=user.id, agent_id=None, args=tool.test_args, mock=mock
            )
            result = await tool.async_wait(task)
        else:
            result = await tool.async_run(tool.test_args, mock=mock)

        if isinstance(result, dict) and result.get("error"):
            click.echo(
                click.style(
                    f"\nFailed to test {tool.key}: {result['error']}",
                    fg="red",
                    bold=True,
                )
            )
        else:
            result = prepare_result(result)
            click.echo(
                click.style(f"\nResult for {tool.key}: {dump_json(result)}", fg=color)
            )

        return result

    async def async_run_tests(tools, api, parallel):
        tasks = [async_test_tool(tool, api) for tool in tools.values()]
        if parallel:
            results = await asyncio.gather(*tasks)
        else:
            results = [await task for task in tasks]
        return results

    if yaml:
        all_tools = get_tools_from_api_files(tools=tools)
    else:
        all_tools = get_tools_from_mongo(tools=tools)

    if not tools:
        confirm = click.confirm(
            f"Run tests for all {len(all_tools)} tools?", default=False
        )
        if not confirm:
            return

    if "flux_trainer" in all_tools:
        confirm = click.confirm(
            "Include flux_trainer test? This will take a long time.", default=False
        )
        if not confirm:
            all_tools.pop("flux_trainer", None)

    results = asyncio.run(async_run_tests(all_tools, api, parallel))

    if save and results:
        save_test_results(all_tools, results)

    errors = [
        f"{tool}: {result['error']}"
        for tool, result in zip(all_tools.keys(), results)
        if result.get("error")
    ]
    error_list = "\n\t".join(errors)
    click.echo(
        click.style(
            f"\n\nTested {len(all_tools)} tools with {len(errors)} errors:\n{error_list}",
            fg="blue",
            bold=True,
        )
    )
