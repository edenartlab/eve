import os
from pathlib import Path
import datetime
import functools
from line_profiler import LineProfiler
from typing import Callable, Any

from eve.tool import Tool, get_api_files

# Profiling setup
PROFILE_PERF = os.getenv("PROFILE_PERF", "false").lower() == "true"
PROFILE_DIR = Path("profiling")

# Initialize profiler if enabled
if PROFILE_PERF:
    PROFILE_DIR.mkdir(exist_ok=True)
    profiler = LineProfiler()

    # Add base classes to profile
    from eve.agent import Agent
    from eve.user import User
    from eve.thread import Thread

    # Add methods to profile
    profiler.add_function(Agent.from_mongo)
    profiler.add_function(Agent.get_tools)
    profiler.add_function(User.from_mongo)
    profiler.add_function(Thread.from_mongo)
    profiler.add_function(Agent.request_thread)
    profiler.add_function(Tool.from_raw_yaml)
    profiler.add_function(Tool.get_sub_class)
    profiler.add_function(Tool.convert_from_yaml)
    profiler.add_function(Tool._get_schema)
    profiler.add_function(get_api_files)


def profile_async(func: Callable) -> Callable:
    """Decorator to profile async functions when PROFILE_PERF is enabled"""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not PROFILE_PERF:
            return await func(*args, **kwargs)

        # Only wrap the top-level function (handle_chat)
        if func.__name__ != "handle_chat":
            return await func(*args, **kwargs)

        # Add function to profiler if not already added
        if func not in profiler.functions:
            profiler.add_function(func)

        # Add setup_chat to profiler if not already added
        from eve.api import setup_chat

        if setup_chat not in profiler.functions:
            profiler.add_function(setup_chat)

        # Enable profiling
        profiler.enable()

        try:
            result = await func(*args, **kwargs)

            # If this is handle_chat, wait for background tasks
            if "background_tasks" in kwargs:
                background_tasks = kwargs["background_tasks"]
                for task in background_tasks.tasks:
                    await task.func(*task.args, **task.kwargs)

            return result
        finally:
            # Disable profiling and write results
            profiler.disable()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_path = PROFILE_DIR / f"profile_{timestamp}.txt"
            with open(profile_path, "w") as f:
                profiler.print_stats(stream=f)
            print(f"\nProfile saved to: {profile_path}")

    return wrapper if PROFILE_PERF else func
