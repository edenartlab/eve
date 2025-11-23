import asyncio
import glob
import json
import os
import random
import traceback

import click

from .. import load_env
from ..agent import Agent
from ..auth import get_my_eden_user
from ..tool import Tool, get_api_files, get_tools_from_api_files, get_tools_from_mongo
from ..utils import CLICK_COLORS, dumps_json, prepare_result, save_test_results

api_tools_order = [
    "txt2img",
    "flux_dev",
    "layer_diffusion",
    "remix",
    "inpaint",
    "outpaint",
    "face_styler",
    "upscaler",
    "background_removal",
    "style_transfer",
    "storydiffusion",
    "beeple_ai",
    "txt2img_test",
    "flux_redux",
    "flux_kontext",
    "mars-id",
    "background_removal_video",
    "animate_3D",
    "style_mixing",
    "txt2vid",
    "transcription",
    "vid2vid_sdxl",
    "img2vid",
    "video_upscaler",
    "frame_interpolation",
    "reel",
    "story",
    "texture_flow",
    "runway",
    "animate_3D_new",
    "lora_trainer",
    "flux_trainer",
    # "news",
    "moodmix",
    "stable_audio",
    "musicgen",
    "elevenlabs_music",
    "elevenlabs_fx",
    "legacy/create",
]


def extract_result_urls(result):
    """Extract URLs from a test result"""
    if not result or not isinstance(result, dict):
        return []

    urls = []

    # Check for direct output URLs
    if "output" in result:
        output = result["output"]
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and "url" in item:
                    urls.append(item["url"])
                elif isinstance(item, str) and (
                    item.startswith("http://") or item.startswith("https://")
                ):
                    urls.append(item)
        elif isinstance(output, str) and (
            output.startswith("http://") or output.startswith("https://")
        ):
            urls.append(output)

    # Check for URLs in nested structures
    if "subtool_calls" in result:
        for subtool in result["subtool_calls"]:
            if isinstance(subtool, dict) and "output" in subtool:
                output = subtool["output"]
                if isinstance(output, str) and (
                    output.startswith("http://") or output.startswith("https://")
                ):
                    urls.append(output)

    return list(set(urls))


def get_tool_test_files(tool_key: str, test_filter: list = None) -> list:
    """Get all test*.json files for a specific tool from its directory"""
    # Get api files to find the tool's directory
    api_files = get_api_files()

    if tool_key not in api_files:
        return []

    # Get the directory path from the api.yaml file path
    api_file_path = api_files[tool_key]
    tool_dir = os.path.dirname(api_file_path)

    # Find all test*.json files in the tool's directory
    test_pattern = os.path.join(tool_dir, "test*.json")
    test_files = glob.glob(test_pattern)

    # Filter test files if specific tests are requested
    if test_filter:
        # Normalize the filter names to include .json extension if not present
        normalized_filter = []
        for test_name in test_filter:
            if not test_name.endswith(".json"):
                test_name += ".json"
            normalized_filter.append(test_name)

        # Only keep files that match the filter
        test_files = [f for f in test_files if os.path.basename(f) in normalized_filter]

    # Load and return the test data from each file
    test_data_list = []
    for test_file in sorted(test_files):
        try:
            with open(test_file, "r") as f:
                test_data = json.load(f)
                # Add metadata about which file this test came from
                test_data["_test_file"] = os.path.basename(test_file)
                test_data_list.append(test_data)
        except Exception as e:
            click.echo(click.style(f"Failed to load {test_file}: {e}", fg="yellow"))

    return test_data_list


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

    # Exit with error code if any updates failed
    if updated < len(api_files):
        raise click.ClickException(
            f"Failed to update {len(api_files) - updated} tool(s)"
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

    os.environ["LOCAL_DEBUG"] = "True"

    tool = Tool.load(key=tool)

    # Parse remaining args into dict, excluding user and agent
    args = {}
    i = 0
    while i < len(ctx.args):
        arg = ctx.args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                # Check if parameter is an array type and wrap single value
                if (
                    key in tool.parameters
                    and tool.parameters[key].get("type") == "array"
                ):
                    args[key] = [value]
                else:
                    args[key] = value
            elif i + 1 < len(ctx.args) and not ctx.args[i + 1].startswith("--"):
                value = ctx.args[i + 1]
                # Check if parameter is an array type and wrap single value
                if (
                    key in tool.parameters
                    and tool.parameters[key].get("type") == "array"
                ):
                    args[key] = [value]
                else:
                    args[key] = value
                i += 1
            else:
                args[key] = True
        i += 1

    # inject
    if args.get("agent"):
        args["agent_id"] = str(Agent.from_mongo(args["agent"]).id)

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
            click.style(f"\nResult for {tool.key}: {dumps_json(result)}", fg=color)
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
@click.option("--save", is_flag=True, default=False, help="Save test results")
@click.option("--mock", is_flag=True, default=False, help="Mock test results")
@click.option(
    "--tests",
    multiple=True,
    help="Specific test files to run (e.g. test1, test2). Runs all tests if not specified.",
)
@click.argument("tools", nargs=-1, required=False)
def test(
    tools: tuple,
    yaml: bool,
    db: str,
    api: bool,
    parallel: bool,
    save: bool,
    mock: bool,
    tests: tuple,
):
    """Test multiple tools with their test args"""

    os.environ["LOCAL_DEBUG"] = "True"

    async def async_test_tool(tool, api, test_args=None, test_name=None):
        color = random.choice(CLICK_COLORS)

        # Use provided test_args or fall back to tool's default test_args
        args_to_test = test_args if test_args is not None else tool.test_args

        test_id = f"{tool.key}[{test_name}]" if test_name else tool.key

        if test_name:
            click.echo(
                click.style(
                    f"\n\nTesting {tool.key} [{test_name}]:", fg=color, bold=True
                )
            )
        else:
            click.echo(click.style(f"\n\nTesting {tool.key}:", fg=color, bold=True))
        click.echo(click.style(f"Args: {dumps_json(args_to_test)}", fg=color))

        try:
            if api:
                user = get_my_eden_user()
                agent_id = args_to_test.pop("agent_id", None) if args_to_test else None
                task = await tool.async_start_task(
                    user_id=user.id,
                    agent_id=agent_id,
                    args=args_to_test,
                    mock=mock,
                    public=False,
                )
                result = await tool.async_wait(task)
            else:
                result = await tool.async_run(args_to_test, mock=mock)

            if isinstance(result, dict) and result.get("error"):
                test_info = f" [{test_name}]" if test_name else ""
                click.echo(
                    click.style(
                        f"\nFailed to test {tool.key}{test_info}: {result['error']}",
                        fg="red",
                        bold=True,
                    )
                )
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": result["error"],
                    "result": None,
                }
            else:
                result = prepare_result(result)
                test_info = f" [{test_name}]" if test_name else ""
                click.echo(
                    click.style(
                        f"\nResult for {tool.key}{test_info}: {dumps_json(result)}",
                        fg=color,
                    )
                )
                return {
                    "test_id": test_id,
                    "success": True,
                    "error": None,
                    "result": result,
                }

        except Exception as e:
            error_msg = str(e)
            test_info = f" [{test_name}]" if test_name else ""
            click.echo(
                click.style(
                    f"\nFailed to test {tool.key}{test_info}: {error_msg}",
                    fg="red",
                    bold=True,
                )
            )
            return {
                "test_id": test_id,
                "success": False,
                "error": error_msg,
                "result": None,
            }

    async def async_run_tests(tools, api, parallel, test_filter):
        all_tasks = []

        for tool_key, tool in tools.items():
            # Try to load test files from the tool's directory
            test_files_data = get_tool_test_files(tool_key, test_filter)

            if test_files_data:
                # Use test files from directory if found
                filter_info = (
                    f" (filtered to {list(test_filter)})" if test_filter else ""
                )
                click.echo(
                    click.style(
                        f"Found {len(test_files_data)} test file(s) for {tool_key}{filter_info}",
                        fg="cyan",
                    )
                )
                for test_data in test_files_data:
                    # Remove the metadata field before passing as test args
                    test_name = test_data.pop("_test_file", "test.json")
                    all_tasks.append(
                        async_test_tool(
                            tool, api, test_args=test_data, test_name=test_name
                        )
                    )
            elif not test_filter and tool.test_args:
                # Fall back to MongoDB test_args if no test files found and no filter specified
                click.echo(
                    click.style(f"Using MongoDB test_args for {tool_key}", fg="cyan")
                )
                all_tasks.append(async_test_tool(tool, api))
            else:
                if test_filter:
                    click.echo(
                        click.style(
                            f"No matching test files found for {tool_key} with filter {list(test_filter)}",
                            fg="yellow",
                        )
                    )
                else:
                    click.echo(
                        click.style(f"No test args found for {tool_key}", fg="yellow")
                    )

        if parallel:
            results = await asyncio.gather(*all_tasks)
        else:
            results = [await task for task in all_tasks]
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

    results = asyncio.run(async_run_tests(all_tools, api, parallel, tests))

    if save and results:
        save_test_results(all_tools, results)

    # Count errors from all test results
    errors = []
    success_count = 0
    for result in results:
        if isinstance(result, dict):
            if result.get("success"):
                success_count += 1
            elif result.get("error"):
                errors.append(result.get("error"))

    total_tests = len(results)
    error_count = len(errors)

    error_list = "\n\t".join(errors) if errors else "None"
    click.echo(
        click.style(
            f"\n\nRan {total_tests} tests across {len(all_tools)} tools:\n"
            f"✓ {success_count} passed\n"
            f"✗ {error_count} failed\n"
            f"Errors: {error_list}",
            fg="blue",
            bold=True,
        )
    )

    # Print detailed execution summary
    click.echo(click.style("\n" + "=" * 60, fg="cyan", bold=True))
    click.echo(click.style("EXECUTION SUMMARY", fg="cyan", bold=True))
    click.echo(click.style("=" * 60, fg="cyan", bold=True))

    for result in results:
        if isinstance(result, dict) and "test_id" in result:
            test_id = result["test_id"]

            if result["success"]:
                # Extract URLs from successful results
                urls = extract_result_urls(result["result"])
                if urls:
                    click.echo(click.style(f"✓ {test_id}:", fg="green", bold=True))
                    for url in urls:
                        click.echo(click.style(f"  → {url}", fg="green"))
                else:
                    click.echo(
                        click.style(
                            f"✓ {test_id}: Success (no URLs found)",
                            fg="green",
                            bold=True,
                        )
                    )
            else:
                # Show error for failed results
                error_msg = result["error"] or "Unknown error"
                click.echo(click.style(f"✗ {test_id}:", fg="red", bold=True))
                click.echo(click.style(f"  → Error: {error_msg}", fg="red"))

    click.echo(click.style("=" * 60, fg="cyan", bold=True))
