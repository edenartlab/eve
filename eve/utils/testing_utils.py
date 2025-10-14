from __future__ import annotations
import os
import random
import requests
from datetime import datetime

from loguru import logger


def save_test_results(tools, results):
    if not results:
        return

    results_dir = os.path.join(
        "tests", "out", f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(results_dir, exist_ok=True)

    for tool, tool_result in zip(tools.keys(), results):
        if isinstance(tool_result, dict) and tool_result.get("error"):
            file_path = os.path.join(results_dir, f"{tool}_ERROR.txt")
            with open(file_path, "w") as f:
                f.write(tool_result["error"])
        else:
            outputs = tool_result.get("output", [])
            outputs = outputs if isinstance(outputs, list) else [outputs]
            intermediate_outputs = tool_result.get("intermediate_outputs", {})

            for o, output in enumerate(outputs):
                if "url" in output:
                    ext = output.get("url").split(".")[-1]
                    filename = (
                        f"{tool}_{o}.{ext}" if len(outputs) > 1 else f"{tool}.{ext}"
                    )
                    file_path = os.path.join(results_dir, filename)
                    response = requests.get(output.get("url"))
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                else:
                    filename = f"{tool}_{o}.txt" if len(outputs) > 1 else f"{tool}.txt"
                    file_path = os.path.join(results_dir, filename)
                    with open(file_path, "w") as f:
                        f.write(output)

            for k, v in intermediate_outputs.items():
                if "url" in v:
                    ext = v.get("url").split(".")[-1]
                    filename = f"{tool}_{k}.{ext}"
                    file_path = os.path.join(results_dir, filename)
                    response = requests.get(v.get("url"))
                    with open(file_path, "wb") as f:
                        f.write(response.content)
    logger.info(f"Test results saved to {results_dir}")


def random_string(length=28):
    # modeled after Replicate id
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
