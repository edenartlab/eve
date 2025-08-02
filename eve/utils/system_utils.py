from __future__ import annotations
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat


def log_memory_info():
    """
    Log basic GPU, RAM, and disk usage percentages using nvidia-smi for GPU metrics.
    """
    import psutil
    import shutil
    import subprocess

    print("\n=== Memory Usage ===")

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
        print(f"GPU Memory: {gpu_percent:.1f}% of {total_mem / 1024:.1f}GB")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GPU info not available")

    # System RAM
    ram = psutil.virtual_memory()
    print(f"RAM Usage: {ram.percent}% of {ram.total / (1024**3):.1f}GB")

    # Disk usage (root directory)
    usage = shutil.disk_usage("/root")
    disk_percent = (usage.used / usage.total) * 100
    print(f"Disk Usage: {disk_percent:.1f}% of {usage.total / (1024**3):.1f}GB")
    print("==================\n")


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
            print(
                f"Attempt {attempt} failed because: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay + jitter)
            delay = delay * 2


async def async_exponential_backoff(
    func,
    max_attempts=5,
    initial_delay=1,
    max_jitter=1,
):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(-max_jitter, max_jitter)
            print(
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
                print(f"Task error: {e}")
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
        print(colored_output)