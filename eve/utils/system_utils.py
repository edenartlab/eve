from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat

from loguru import logger


def log_memory_info():
    """
    Log basic GPU, RAM, and disk usage percentages using nvidia-smi for GPU metrics.
    """
    import shutil
    import subprocess

    import psutil

    # GPU VRAM using nvidia-smi
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ]
        )
        total_mem, used_mem = map(int, result.decode("utf-8").strip().split(","))
        gpu_percent = (used_mem / total_mem) * 100
        logger.info(f"GPU Memory: {gpu_percent:.1f}% of {total_mem / 1024:.1f}GB")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("GPU info not available")

    # System RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM Usage: {ram.percent}% of {ram.total / (1024**3):.1f}GB")

    # Disk usage (root directory)
    usage = shutil.disk_usage("/root")
    disk_percent = (usage.used / usage.total) * 100
    logger.info(f"Disk Usage: {disk_percent:.1f}% of {usage.total / (1024**3):.1f}GB")


def exponential_backoff(
    func,
    max_attempts=5,
    initial_delay=1,
    max_jitter=1,
):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(-max_jitter, max_jitter)
            logger.warning(
                f"Attempt {attempt} failed because: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay + jitter)
            delay = delay * 2


async def async_exponential_backoff(
    func,
    max_attempts=5,
    initial_delay=1,
    max_jitter=1,
    timeout_seconds: float | None = 300,  # 5 minute default timeout per attempt
):
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries (doubles each retry)
        max_jitter: Random jitter added to delay
        timeout_seconds: Timeout per attempt in seconds (default 5 min, None to disable)
    """
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            if timeout_seconds is not None:
                return await asyncio.wait_for(func(), timeout=timeout_seconds)
            else:
                return await func()
        except asyncio.TimeoutError:
            if attempt == max_attempts:
                raise TimeoutError(
                    f"Operation timed out after {max_attempts} attempts "
                    f"({timeout_seconds}s timeout per attempt)"
                )
            logger.warning(
                f"Attempt {attempt} timed out after {timeout_seconds}s. "
                f"Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)
            delay = delay * 2
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(-max_jitter, max_jitter)
            logger.warning(
                f"Attempt {attempt} failed because: {e}. Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay + jitter)
            delay = delay * 2


def process_in_parallel(array, func, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(func, item, index): index
            for index, item in enumerate(array)
        }
        results = [None] * len(array)
        for future in as_completed(futures):
            try:
                index = futures[future]
                results[index] = future.result()
            except Exception as e:
                logger.error(f"Task error: {e}")
                for f in futures:
                    f.cancel()
                raise e
    return results


def pprint(*args, color=None, indent=4):
    colors = {
        "red": "\033[38;2;255;100;100m",
        "green": "\033[38;2;100;255;100m",
        "blue": "\033[38;2;100;100;255m",
        "yellow": "\033[38;2;255;255;100m",
        "magenta": "\033[38;2;255;100;255m",
        "cyan": "\033[38;2;100;255;255m",
    }
    if not color:
        color = random.choice(list(colors.keys()))
    if color not in colors:
        raise ValueError(f"Invalid color: {color}")
    for arg in args:
        string = pformat(arg, indent=indent)
        colored_output = f"{colors[color]}{string}\033[0m"
        logger.info(colored_output)
