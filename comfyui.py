"""
DB=STAGE SKIP_TESTS=1 WORKSPACE=audio modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=batch_tools modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=flux modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=img_tools modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=mars_exclusive modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=txt2img modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=video modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=video2 modal deploy comfyui.py

DB=PROD WORKSPACE=audio modal deploy comfyui.py
DB=PROD WORKSPACE=batch_tools modal deploy comfyui.py
DB=PROD WORKSPACE=flux modal deploy comfyui.py
DB=PROD WORKSPACE=img_tools modal deploy comfyui.py
DB=PROD WORKSPACE=mars_exclusive modal deploy comfyui.py
DB=PROD WORKSPACE=txt2img modal deploy comfyui.py
DB=PROD WORKSPACE=video modal deploy comfyui.py
DB=PROD WORKSPACE=video2 modal deploy comfyui.py
"""

import copy
import glob
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import traceback
import urllib
from pathlib import Path

import git
import modal
from bson import ObjectId
from loguru import logger

import eve.utils as eden_utils
from eve.mongo import get_collection
from eve.s3 import get_full_url
from eve.task import task_handler_method
from eve.tool import Tool

GPUs = {
    "A100": "A100-40GB",  # Changed from modal.gpu.A100()
    "A100-80GB": "A100-80GB",  # Changed from modal.gpu.A100(size="80GB")
}

if not os.getenv("WORKSPACE"):
    raise Exception("No workspace selected")

db = os.getenv("DB", "STAGE").upper()
workspace_name = os.getenv("WORKSPACE")
app_name = f"comfyui-{workspace_name}-{db}"
test_workflows = os.getenv("WORKFLOWS")
root_workflows_folder = (
    "../private_workflows" if os.getenv("PRIVATE") else "../workflows"
)
test_all = True if os.getenv("TEST_ALL") else False
specific_tests = (
    os.getenv("SPECIFIC_TEST").split(",") if os.getenv("SPECIFIC_TEST") else []
)
skip_tests = os.getenv("SKIP_TESTS")
test_inactive = True if os.getenv("TEST_INACTIVE") else False

# Run a bunch of checks to verify input args:
if test_all and specific_tests:
    logger.info(
        "WARNING: can't have both TEST_ALL and SPECIFIC_TEST at the same time..."
    )
    logger.info("Running TEST_ALL instead")
    specific_tests = []

logger.info("========================================")
logger.info(f"db: {db}")
logger.info(f"workspace: {workspace_name}")
logger.info(f"test_workflows: {test_workflows}")
logger.info(f"test_all: {test_all}")
logger.info(f"specific_tests: {specific_tests}")
logger.info(f"skip_tests: {skip_tests}")
logger.info(f"test_inactive: {test_inactive}")
logger.info("========================================")

if not test_workflows and workspace_name and not test_all:
    logger.info("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info("!!!! WARNING: You are deploying a workspace without TEST_ALL !!!!")
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")


def install_comfyui():
    os.chdir("/root")
    snapshot = json.load(open("/root/workspace/snapshot.json", "r"))
    comfyui_commit_sha = snapshot["comfyui"]

    logger.info(f"Initializing git repository in {os.getcwd()}")
    result = subprocess.run(["git", "init", "."], check=True, capture_output=True)
    logger.info(f"Git init output: {result.stdout.decode()}")

    logger.info("Adding ComfyUI remote and fetching")
    result = subprocess.run(
        [
            "git",
            "remote",
            "add",
            "--fetch",
            "origin",
            "https://github.com/comfyanonymous/ComfyUI",
        ],
        check=True,
        capture_output=True,
    )
    logger.info(f"Git remote add output: {result.stdout.decode()}")

    logger.info(f"Checking out commit: {comfyui_commit_sha}")
    result = subprocess.run(
        ["git", "checkout", comfyui_commit_sha], check=True, capture_output=True
    )
    logger.info(f"Git checkout output: {result.stdout.decode()}")

    subprocess.run(
        [
            "pip",
            "install",
            "xformers!=0.0.18",
            "sageattention",
            "-r",
            "requirements.txt",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    # List all files and directories in the current directory:
    logger.info("Current directory structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(os.getcwd(), "").count(os.sep)
        if level <= 1:
            indent = " " * 4 * (level)
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                logger.info(f"{subindent}{f}")
        if level >= 1:
            dirs.clear()  # This prevents os.walk from recursing deeper
    logger.info("ComfyUI installation completed successfully")


def install_custom_nodes():
    os.chdir("/root")

    snapshot = json.load(open("/root/workspace/snapshot.json", "r"))
    custom_nodes = snapshot["git_custom_nodes"]
    for url, node in custom_nodes.items():
        logger.info(f"Installing custom node {url} with hash {node['hash']}")
        install_custom_node_with_retries(url, node["hash"])
    post_install_commands = snapshot.get("post_install_commands", [])
    for cmd in post_install_commands:
        os.system(cmd)

    # Check for ControlFlowUtils helper.py and set DEBUG_MODE to False
    try:
        helper_path = os.path.join(
            os.environ.get("COMFYUI_PATH", "/root"),
            "custom_nodes/ControlFlowUtils/helper.py",
        )
        if os.path.exists(helper_path):
            logger.info(
                f"Found ControlFlowUtils helper.py at {helper_path}, checking for DEBUG_MODE"
            )
            with open(helper_path, "r") as f:
                content = f.read()

            if "DEBUG_MODE = True" in content:
                logger.info("Setting DEBUG_MODE to False in ControlFlowUtils helper.py")
                content = content.replace("DEBUG_MODE = True", "DEBUG_MODE = False")
                with open(helper_path, "w") as f:
                    f.write(content)
            else:
                logger.info(
                    "DEBUG_MODE is already set to False or not found in ControlFlowUtils helper.py"
                )
    except Exception:
        pass


def install_custom_node_with_retries(url, hash, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}: Installing {url}")
            install_custom_node(url, hash)
            logger.info(f"Successfully installed {url}")
            return
        except Exception as e:
            if attempt < max_retries:
                logger.info(f"Attempt {attempt + 1} failed because: {str(e)}")
                logger.info(f"Exception type: {type(e)}")
                traceback.print_exc()  # This will print the full stack trace
                logger.info("Retrying...")
                time.sleep(5)
            else:
                logger.info(f"All attempts failed. Final error: {str(e)}")
                traceback.print_exc()
                raise


def install_custom_node(url, hash):
    repo_name = url.split("/")[-1].split(".")[0]
    repo_path = os.path.join("/root", "custom_nodes", repo_name)
    if os.path.exists(repo_path):
        return
    repo = git.Repo.clone_from(url, repo_path)
    repo.git.checkout(hash)
    repo.submodule_update(recursive=True)
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.startswith("requirements") and file.endswith((".txt", ".pip")):
                try:
                    requirements_path = os.path.join(root, file)
                    if (
                        "with-cupy" in requirements_path
                    ):  # hack for ComfyUI-Frame-Interpolation, don't use CuPy
                        continue
                    subprocess.run(
                        ["pip", "install", "-r", requirements_path], check=True
                    )
                except Exception as e:
                    logger.info(f"Error installing requirements: {e}")


def create_symlink(source_path, target_path, is_directory=False, force=False):
    """Create a symlink, ensuring parent directories exist."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if force:
            target_path.unlink()
        else:
            return
    target_path.symlink_to(source_path, target_is_directory=is_directory)


def clone_repo(repo_url, target_path, force=False):
    """Clone a git repository to the specified target path."""
    if target_path.exists():
        if force:
            logger.info(f"Removing existing repository at {target_path}")
            shutil.rmtree(target_path)
        else:
            logger.info(f"Repository already exists at {target_path}, skipping clone")
            return

    logger.info(f"Cloning repository {repo_url} to {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Clone directly to the specified target path
        subprocess.run(
            ["git", "clone", repo_url, str(target_path)],
            check=True,
            capture_output=True,
        )
        downloads_vol.commit()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error cloning repository {repo_url}: {e.stderr.decode()}")


def download_file(url, target_path, force=False):
    """Download a single file to the target path."""
    if target_path.is_file() and not force:
        logger.info(f"Skipping download, getting {target_path} from cache")
        return

    logger.info(f"Downloading {url} to {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    eden_utils.download_file(url, target_path)
    downloads_vol.commit()


def handle_repo_download(repo_url, vol_path, comfy_path, force=False):
    """Handle downloading and linking a git repository."""
    # Clone directly to the volume path that matches the specified comfy path
    clone_repo(repo_url, vol_path, force=force)

    # Create symlink to the exact specified location
    create_symlink(vol_path, comfy_path, is_directory=True, force=force)


def handle_file_download(url, vol_path, comfy_path, force=False):
    """Handle downloading and linking a single file."""
    download_file(url, vol_path, force=force)
    create_symlink(vol_path, comfy_path, force=force)


def sync_final_state_to_volume():
    """
    Sync the final state of ComfyUI after all tests complete to the persistent volume.
    This captures any files downloaded during testing that aren't already in the volume.
    """
    logger.info("Starting sync of final ComfyUI state to persistent volume...")

    # Key directories that might contain downloaded files from testing
    sync_paths = ["models", "custom_nodes", "output", "input"]

    volume_root = "/data/comfyui_root"
    os.makedirs(volume_root, exist_ok=True)

    total_synced = 0
    for path in sync_paths:
        root_path = f"/root/{path}"
        volume_path = f"{volume_root}/{path}"

        if not os.path.exists(root_path):
            logger.info(f"Skipping {root_path} - does not exist")
            continue

        logger.info(f"Syncing {root_path} to {volume_path}...")

        # Use rsync-like logic to sync only new/changed files
        if os.path.exists(volume_path):
            # Update existing directory
            synced_files = sync_directory_changes(root_path, volume_path)
        else:
            # Copy entire directory for first time
            shutil.copytree(root_path, volume_path, dirs_exist_ok=True)
            synced_files = count_files_recursive(root_path)

        total_synced += synced_files
        logger.info(f"Synced {synced_files} files from {path}")

    # Commit changes to volume
    downloads_vol.commit()
    logger.info(
        f"Successfully synced {total_synced} files to persistent volume and committed changes"
    )


def sync_directory_changes(source_dir, dest_dir):
    """
    Sync only new or modified files from source to destination directory.
    Returns the number of files synced.
    """
    synced_count = 0

    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path from source root
        rel_path = os.path.relpath(root, source_dir)
        dest_root = os.path.join(dest_dir, rel_path) if rel_path != "." else dest_dir

        # Create destination directory if it doesn't exist
        os.makedirs(dest_root, exist_ok=True)

        for file in files:
            source_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)

            # Check if file needs to be synced (new or modified)
            should_sync = False
            if not os.path.exists(dest_file):
                should_sync = True
            else:
                # Compare modification times and sizes
                source_stat = os.stat(source_file)
                dest_stat = os.stat(dest_file)
                if (
                    source_stat.st_mtime > dest_stat.st_mtime
                    or source_stat.st_size != dest_stat.st_size
                ):
                    should_sync = True

            if should_sync:
                shutil.copy2(source_file, dest_file)  # copy2 preserves metadata
                synced_count += 1

    return synced_count


def count_files_recursive(directory):
    """Count total files in a directory recursively."""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count


def restore_state_from_volume():
    """
    Restore the cached ComfyUI state from the persistent volume.
    This is used when SKIP_TESTS=1 to populate /root with previously cached downloads.
    """
    logger.info("Restoring ComfyUI state from persistent volume...")

    volume_root = "/data/comfyui_root"
    if not os.path.exists(volume_root):
        logger.info(
            "No cached state found in volume. This appears to be the first deployment."
        )
        return

    # Restore key directories
    restore_paths = ["models", "custom_nodes", "output", "input"]

    total_restored = 0
    for path in restore_paths:
        volume_path = f"{volume_root}/{path}"
        root_path = f"/root/{path}"

        if not os.path.exists(volume_path):
            logger.info(f"Skipping {path} - no cached version found")
            continue

        logger.info(f"Restoring {path} from volume...")

        # Remove existing directory and replace with cached version
        if os.path.exists(root_path):
            shutil.rmtree(root_path)

        shutil.copytree(volume_path, root_path)
        restored_files = count_files_recursive(root_path)
        total_restored += restored_files
        logger.info(f"Restored {restored_files} files to {path}")

    logger.info(
        f"Successfully restored {total_restored} files from persistent volume cache"
    )


def download_files(force_redownload=False):
    """
    Main function to process downloads from downloads.json.
    If a source URL appears multiple times, it's downloaded once to the persistent
    volume path corresponding to its first encountered key in downloads.json.
    Subsequent entries for the same URL will have their comfy_path symlinked
    to this single downloaded copy.
    The force_redownload flag refreshes the content of this first download.

    Args:
        force_redownload (bool): If True, force redownload and overwrite existing files.
    """
    downloads = json.load(open("/root/workspace/downloads.json", "r"))
    # Maps actual_source_url to its persistent_vol_path where it was first downloaded/cloned.
    downloaded_source_registry = {}

    for path_key, source_identifier in downloads.items():
        comfy_path = (
            pathlib.Path("/root") / path_key
        )  # Target path in ComfyUI's view (e.g., /root/models/...)
        # persistent_vol_path is the path on /data IF this entry causes a download.
        # It's derived from path_key, implying a unique storage location per downloads.json key if downloaded directly by this entry.
        current_entry_persistent_vol_path = pathlib.Path("/data") / path_key

        # If comfy_path already exists (e.g., from a previous run) and we are not forcing a redownload for this entry,
        # we can skip processing it. The optimization is primarily for downloads within a single fresh run.
        if comfy_path.exists() and not force_redownload:
            logger.info(
                f"Comfy path {comfy_path} already exists and not forcing redownload, skipping processing for this entry."
            )
            # Note: If this comfy_path was for a URL that *will* appear again, and this skip prevents
            # populating downloaded_source_registry, the later occurrence might download it.
            # This is acceptable; the main goal is to avoid duplicate downloads for URLs processed *within the same run*.
            # For a truly clean state ensuring all symlinks, clear /root and /data before run.
            continue

        try:
            is_git_clone = source_identifier.startswith("git clone ")
            actual_source_url = (
                source_identifier[10:].strip() if is_git_clone else source_identifier
            )

            if actual_source_url in downloaded_source_registry:
                # This source URL has already been processed (downloaded/cloned) in this run.
                original_persistent_path = downloaded_source_registry[actual_source_url]

                logger.info(
                    f"Source '{actual_source_url}' already processed to persistent path '{original_persistent_path}'."
                )
                if force_redownload:
                    # The content at original_persistent_path should have been refreshed by its first encounter if force_redownload is true.
                    logger.info(
                        f"Force_redownload is True; ensuring symlink from '{comfy_path}' to refreshed content at '{original_persistent_path}'."
                    )
                else:
                    logger.info(
                        f"Creating symlink from '{comfy_path}' to '{original_persistent_path}'."
                    )

                # Ensure the comfy_path for *this* entry symlinks to the one true persistent copy.
                # Force symlink creation to overwrite existing file/symlink at comfy_path if necessary.
                create_symlink(
                    original_persistent_path,
                    comfy_path,
                    is_directory=original_persistent_path.is_dir(),
                    force=True,
                )
            else:
                # This is the first time this actual_source_url is being processed in this run.
                # It needs to be downloaded/cloned. It will be stored at current_entry_persistent_vol_path.
                # This path then becomes the canonical persistent path for this actual_source_url in the registry.
                logger.info(
                    f"Processing source '{actual_source_url}' for the first time in this run."
                )
                logger.info(
                    f"It will be stored at persistent path: '{current_entry_persistent_vol_path}'."
                )
                logger.info(f"ComfyUI path will be: '{comfy_path}'.")

                if is_git_clone:
                    # handle_repo_download clones to current_entry_persistent_vol_path and symlinks comfy_path to it.
                    handle_repo_download(
                        actual_source_url,
                        current_entry_persistent_vol_path,
                        comfy_path,
                        force=force_redownload,
                    )
                else:
                    # handle_file_download downloads to current_entry_persistent_vol_path and symlinks comfy_path to it.
                    handle_file_download(
                        actual_source_url,
                        current_entry_persistent_vol_path,
                        comfy_path,
                        force=force_redownload,
                    )

                # Register this source URL and its actual persistent storage path.
                downloaded_source_registry[actual_source_url] = (
                    current_entry_persistent_vol_path
                )
                logger.info(
                    f"Registered '{actual_source_url}' to persistent path '{current_entry_persistent_vol_path}'."
                )

            # Final check: the comfy_path must exist after processing.
            if not comfy_path.exists():
                # This could happen if create_symlink failed silently or handle_x_download didn't create the symlink.
                raise Exception(
                    f"ComfyUI path {comfy_path} was not found after processing source {source_identifier}. Expected it to be a symlink or file."
                )

        except Exception as e:
            # Provide context for which item failed.
            detailed_error = traceback.format_exc()
            raise Exception(
                f"Error processing download item with key '{path_key}' (source: '{source_identifier}'): {e}\nFull Traceback:\n{detailed_error}"
            )


def get_workflows():
    """Get list of workflows to test based on environment variables."""
    workflows_dir = pathlib.Path("/root/workspace/workflows")
    if not workflows_dir.exists():
        raise Exception(f"Workflows directory not found: {workflows_dir}")

    # First get all workflows with workflow_api.json
    workflow_names = [
        f.name
        for f in workflows_dir.iterdir()
        if f.is_dir() and (f / "workflow_api.json").exists()
    ]

    # First filter by WORKFLOWS env var if specified
    test_workflows = os.getenv("WORKFLOWS")
    if test_workflows:
        test_workflows = test_workflows.split(",")
        if not all([w in workflow_names for w in test_workflows]):
            invalid_workflows = [w for w in test_workflows if w not in workflow_names]
            raise Exception(
                f"One or more invalid workflows found: {', '.join(invalid_workflows)}. Available workflows: {', '.join(workflow_names)}"
            )
        workflow_names = test_workflows
        logger.info(
            f"====> Running tests for subset of workflows: {' | '.join(workflow_names)}"
        )
    else:
        logger.info(
            f"====> Running tests for all workflows: {' | '.join(workflow_names)}"
        )

    # Then filter based on status if TEST_INACTIVE is not set
    if not os.getenv("TEST_INACTIVE"):
        filtered_names = []
        for name in workflow_names:
            tool = Tool.from_yaml(f"/root/workspace/workflows/{name}/api.yaml")
            if tool.active:
                filtered_names.append(name)
            else:
                logger.info(f"Workflow {name} is inactive, skipping")

        workflow_names = filtered_names

    if not workflow_names:
        raise Exception("No workflows found!")

    # Sort workflow names alphabetically to ensure consistent test order
    workflow_names.sort()
    return workflow_names


def get_test_files(workflow):
    """Get list of test files for a workflow."""
    # First check if this workflow is in the WORKFLOWS list if specified
    if os.getenv("WORKFLOWS"):
        allowed_workflows = os.getenv("WORKFLOWS").split(",")
        if workflow not in allowed_workflows:
            return []  # Skip workflows not in the WORKFLOWS list

    # Now handle TEST_ALL and specific tests for allowed workflows
    if os.getenv("TEST_ALL"):
        return glob.glob(f"/root/workspace/workflows/{workflow}/test*.json")
    elif specific_tests:
        return [
            f"/root/workspace/workflows/{workflow}/{test}" for test in specific_tests
        ]
    else:
        return [f"/root/workspace/workflows/{workflow}/test.json"]


def run_tests_or_restore():
    """
    Either run all tests for the current workspace OR restore cached state from volume.
    The behavior is controlled by the SKIP_TESTS environment variable.
    """
    if os.getenv("SKIP_TESTS"):
        logger.info(
            "SKIP_TESTS is set - restoring cached state from volume instead of running tests"
        )
        restore_state_from_volume()
        logger.info("Cached state restoration completed")
        return
    else:
        logger.info("SKIP_TESTS not set - running full test suite")
        test_workflows()
        return


def test_workflows():
    """Run all tests for the current workspace."""
    overall_start_time = time.time()
    logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting test execution")

    if os.getenv("SKIP_TESTS"):
        logger.info("Skipping tests")
        return

    # Only run tests if TEST_ALL or specific tests are specified
    if not (
        os.getenv("TEST_ALL") or os.getenv("WORKFLOWS") or os.getenv("SPECIFIC_TEST")
    ):
        logger.info("No tests specified, skipping tests")
        return

    test_summary = []
    failed_tests = []
    succinct_summary = []

    # Get list of workflows
    workflow_start = time.time()
    logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finding workflows...")
    workflows = get_workflows()
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found workflows: {workflows}")
    logger.info(f"Workflow discovery took {time.time() - workflow_start:.2f}s")

    # Print overview of all tests that will be run
    logger.info("\n========== TEST EXECUTION PLAN ==========")
    total_tests = 0
    for workflow in workflows:
        test_files = get_test_files(workflow)
        if test_files:
            # Check if workflow is inactive before counting it
            tool = Tool.from_yaml(f"/root/workspace/workflows/{workflow}/api.yaml")
            if not tool.active and not test_inactive:
                logger.info(f"\nWorkflow: {workflow} (inactive, will be skipped)")
                continue
            logger.info(f"\nWorkflow: {workflow}")
            for test in test_files:
                logger.info(f"  • {os.path.basename(test)}")
                total_tests += 1
    logger.info(f"\nTotal tests to run: {total_tests}")
    logger.info("========================================\n")

    # Check if we have AWS credentials
    has_aws_creds = all(
        [
            os.getenv("AWS_ACCESS_KEY_ID"),
            os.getenv("AWS_SECRET_ACCESS_KEY"),
            os.getenv("AWS_REGION_NAME"),
        ]
    )
    logger.info(f"AWS credentials available: {has_aws_creds}")

    # Create a single ComfyUI instance for all tests
    server_start = time.time()
    logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting ComfyUI server...")
    comfyui = ComfyUI()
    try:
        comfyui._start()  # This will set server_address and start the server
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server started successfully"
        )
        logger.info(f"Server startup took {time.time() - server_start:.2f}s")
    except Exception as e:
        logger.info(f"Error starting ComfyUI server: {e}")
        raise

    current_test = 0
    # For each workflow
    for workflow_idx, workflow in enumerate(workflows, 1):
        workflow_test_start = time.time()
        logger.info(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing workflow: {workflow}"
        )

        # Get list of test files
        test_files = get_test_files(workflow)
        logger.info(f"Found test files: {test_files}")

        # For each test file
        for test_idx, test in enumerate(test_files, 1):
            test_start = time.time()

            try:
                tool = Tool.from_yaml(f"/root/workspace/workflows/{workflow}/api.yaml")
                if not tool.active and not test_inactive:
                    logger.info(f"{workflow} is inactive, skipping test")
                    continue

                current_test += 1
                successful_tests = current_test - 1 - len(failed_tests)
                logger.info(
                    f"\n\n\n------------------ Test ({current_test}/{total_tests}) - Workflow {workflow_idx}/{len(workflows)} - {workflow} ({test_idx}/{len(test_files)}) ------------------"
                )
                logger.info(
                    f"Progress: {successful_tests} successful, {len(failed_tests)} failed tests so far"
                )

                test_args = json.loads(open(test, "r").read())
                test_args = tool.prepare_args(test_args)
                test_name = f"{workflow}_{os.path.basename(test)}"
                logger.info(f"====> Running test: {test_name}")

                result = comfyui._execute(workflow, test_args)

                if has_aws_creds:
                    try:
                        logger.info(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting S3 upload..."
                        )
                        s3_start = time.time()
                        result = eden_utils.upload_result(result)
                        logger.info(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] S3 upload successful"
                        )
                        logger.info(f"S3 upload took {time.time() - s3_start:.2f}s")

                        # Print individual test summary
                        logger.info("\n========== TEST SUMMARY ==========")
                        test_duration = time.time() - test_start
                        logger.info(f"Test: {test_name}")
                        logger.info(f"Duration: {test_duration:.2f}s")
                        logger.info("\nGenerated Files:")
                        bucket_prefix = (
                            "edenartlab-stage-data"
                            if db == "STAGE"
                            else "edenartlab-data"
                        )
                        region = os.getenv("AWS_REGION_NAME", "us-east-1")

                        # Store primary output URL for succinct summary
                        primary_url = None
                        for output_key, output_files in result.items():
                            if isinstance(output_files, list):
                                for output in output_files:
                                    if isinstance(output, dict):
                                        filename = output.get("filename")
                                        if filename:
                                            s3_url = f"https://{bucket_prefix}.s3.{region}.amazonaws.com/{filename}"
                                            if (
                                                output_key == "output"
                                                and not primary_url
                                            ):
                                                primary_url = s3_url
                                            logger.info(f"\n{output_key}:")
                                            logger.info(f"  URL: {s3_url}")
                                            if "mediaAttributes" in output:
                                                for attr, value in output[
                                                    "mediaAttributes"
                                                ].items():
                                                    logger.info(f"  {attr}: {value}")
                                    else:
                                        logger.info(f"\n{output_key}: {output}")
                        logger.info("================================\n")

                        # Add to succinct summary
                        succinct_summary.append(
                            {
                                "test": test_name,
                                "status": "SUCCESS",
                                "duration": test_duration,
                                "url": primary_url,
                            }
                        )

                    except Exception as e:
                        logger.info(f"S3 upload failed: {e}")
                        logger.info("Falling back to local file verification")
                        # Fall back to local verification
                        for output_files in result.values():
                            if isinstance(output_files, list):
                                for file_path in output_files:
                                    full_path = os.path.join("/root", file_path)
                                    if not os.path.exists(full_path):
                                        raise Exception(
                                            f"Output file not found: {full_path}"
                                        )
                                    logger.info(
                                        f"====> Verified output file exists: {full_path}"
                                    )
                else:
                    logger.info("No AWS credentials available, verifying files locally")
                    # Local verification only
                    for output_files in result.values():
                        if isinstance(output_files, list):
                            for file_path in output_files:
                                full_path = os.path.join("/root", file_path)
                                if not os.path.exists(full_path):
                                    raise Exception(
                                        f"Output file not found: {full_path}"
                                    )
                                logger.info(
                                    f"====> Verified output file exists: {full_path}"
                                )

                test_duration = time.time() - test_start
                logger.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test completed successfully in {test_duration:.2f}s"
                )

                # Add to summary
                summary_entry = {
                    "workflow": workflow,
                    "test": os.path.basename(test),
                    "duration": f"{test_duration:.2f}s",
                    "outputs": {},
                }

                # Add URLs or local paths to summary
                for output_key, output_files in result.items():
                    if isinstance(output_files, list):
                        summary_entry["outputs"][output_key] = output_files
                    else:
                        summary_entry["outputs"][output_key] = [output_files]

                test_summary.append(summary_entry)
            except Exception as e:
                error_trace = traceback.format_exc()
                failed_tests.append(
                    {
                        "workflow": workflow,
                        "test": os.path.basename(test),
                        "error": error_trace,
                        "test_path": test.replace("/root/workspace/workflows/", ""),
                    }
                )
                logger.info(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test failed: {test_name}"
                )
                logger.info(error_trace)

                # Add to succinct summary
                succinct_summary.append(
                    {
                        "test": test_name,
                        "status": "FAILED",
                        "duration": time.time() - test_start,
                        "error": str(e),
                    }
                )
                # Continue with next test instead of raising exception here
                continue

        workflow_duration = time.time() - workflow_test_start
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Workflow {workflow} completed in {workflow_duration:.2f}s"
        )

    # Print final summary
    total_duration = time.time() - overall_start_time
    logger.info("\n\n========== FINAL TEST EXECUTION SUMMARY ==========")
    logger.info(f"Total execution time: {total_duration:.2f}s")
    logger.info(f"Number of tests executed: {len(test_summary)}")
    logger.info(f"Number of failed tests: {len(failed_tests)}")

    logger.info("\nDetailed test results:")
    logger.info("----------------------------------------")

    for entry in test_summary:
        logger.info(f"\nWorkflow: {entry['workflow']}")
        logger.info(f"Test: {entry['test']}")
        logger.info(f"Duration: {entry['duration']}")
        logger.info("Outputs:")
        for output_key, outputs in entry["outputs"].items():
            for output in outputs:
                if isinstance(output, dict):
                    filename = output.get("filename")
                    if filename:
                        bucket_prefix = (
                            "edenartlab-stage-data"
                            if db == "STAGE"
                            else "edenartlab-data"
                        )
                        region = os.getenv("AWS_REGION_NAME", "us-east-1")
                        s3_url = f"https://{bucket_prefix}.s3.{region}.amazonaws.com/{filename}"
                        logger.info(f"  {output_key}: {s3_url}")

                        if "mediaAttributes" in output:
                            for attr, value in output["mediaAttributes"].items():
                                logger.info(f"    {attr}: {value}")
                else:
                    logger.info(f"  {output_key}: {output}")
        logger.info("----------------------------------------")

    # Print failed tests summary if any
    if failed_tests:
        logger.info("\n========== FAILED TESTS SUMMARY ==========")
        logger.info("\nTests failed:")
        for failed_test in failed_tests:
            logger.info(f"    {failed_test['test_path']}")

        logger.info("\nDetailed failure information:")
        logger.info("----------------------------------------")
        for failed_test in failed_tests:
            logger.info(f"\nWorkflow: {failed_test['workflow']}")
            logger.info(f"Test: {failed_test['test']}")
            logger.info("Error Stack Trace:")
            logger.info(failed_test["error"])
            logger.info("----------------------------------------")

    # Print ComfyUI reinit timing information
    if len(test_summary) > 1:
        logger.info("\n========== COMFYUI REINIT TIMING ==========")
        logger.info(
            "Reinitializing ComfyUI to check startup time after all files are downloaded..."
        )
        reinit_start = time.time()

        # Stop the existing ComfyUI server first
        try:
            comfyui._stop_server()
        except Exception as e:
            logger.info(f"Warning: Error stopping existing server: {e}")

        # Create new instance and start
        comfyui = ComfyUI()
        try:
            comfyui._start()
            reinit_duration = time.time() - reinit_start
            logger.info(f"ComfyUI reinitialization completed in {reinit_duration:.2f}s")
        except Exception as e:
            logger.info(f"Error during ComfyUI reinitialization: {e}")
        logger.info("----------------------------------------")

    # Print succinct test results at the very end
    logger.info("\nSuccinct Test Results:")
    logger.info("----------------------------------------")
    for result in succinct_summary:
        if result["status"] == "SUCCESS":
            logger.info(f"✓ {result['test']} - {result['duration']:.2f}s")
            logger.info(f"    {result['url']}")
        else:
            logger.info(f"✗ {result['test']} - {result['duration']:.2f}s")
            logger.info(f"    Error: {result['error']}")
    logger.info("----------------------------------------")

    # Sync the final ComfyUI state to persistent volume after all tests
    logger.info("\n========== SYNCING FINAL STATE TO VOLUME ==========")
    try:
        sync_final_state_to_volume()
        logger.info("Successfully synced final state to persistent volume")
    except Exception as e:
        logger.info(f"Warning: Failed to sync final state to volume: {e}")
        # Don't fail the entire deployment if sync fails
    logger.info("========================================")

    # Only raise the exception at the end after running all tests
    if failed_tests:
        raise Exception(
            f"{len(failed_tests)} test(s) failed. See above for detailed error traces."
        )


root_dir = Path(__file__).parent

downloads_vol = modal.Volume.from_name("comfy-downloads", create_if_missing=True)


# you can add "force_build=True" to any of these functions to force a full rebuild of the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"COMFYUI_PATH": "/root", "COMFYUI_MODEL_PATH": "/root/models"})
    .env({"TEST_ALL": str(os.getenv("TEST_ALL", ""))})
    .env({"SPECIFIC_TEST": str(os.getenv("SPECIFIC_TEST", ""))})
    .env({"WORKFLOWS": str(os.getenv("WORKFLOWS", ""))})
    .apt_install(
        "git",
        "git-lfs",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libmagic1",
        "ffmpeg",
        "libegl1",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("diffusers==0.31.0", "psutil", "flet==0.27.6")
    .env({"WORKSPACE": workspace_name})
    .add_local_python_source("eve", copy=True)
    # First copy of workflow files
    .add_local_dir(
        f"{root_workflows_folder}/workspaces/{workspace_name}",
        "/root/workspace",
        copy=True,
    )
    .run_function(install_comfyui)
    .run_function(install_custom_nodes, gpu="A100")
    .pip_install(
        "moviepy==1.0.3",
        "accelerate==1.4.0",
        "peft==0.14.0",
        "transformers==4.49.0",
        "flet==0.27.6",
        "safetensors==0.5.3",
        "imgui-bundle==1.6.3",
    )
    .run_function(
        download_files,
        volumes={"/data": downloads_vol},
        secrets=[
            modal.Secret.from_name("eve-secrets"),
            modal.Secret.from_name(f"eve-secrets-{db}"),
        ],
    )
    # Second copy of workflow files after downloads
    .add_local_dir(
        f"{root_workflows_folder}/workspaces/{workspace_name}",
        "/root/workspace",
        copy=True,
    )
    .run_function(
        run_tests_or_restore,
        gpu="A100",
        volumes={"/data": downloads_vol},
        secrets=[
            modal.Secret.from_name("eve-secrets"),
            modal.Secret.from_name(f"eve-secrets-{db}"),
        ],
    )
    .env({"SKIP_TESTS": str(skip_tests or "")})
)

gpu = "A100"

app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)


class ComfyUI:
    server_address: str = modal.parameter(default="127.0.0.1:8188")

    def _start(self, port=8188):
        logger.info("DEBUG: Starting ComfyUI server...")
        t1 = time.time()
        self.server_address = f"127.0.0.1:{port}"
        cmd = f"python /root/main.py --dont-print-server --listen --port {port}"
        subprocess.Popen(cmd, shell=True)
        while not self._is_server_running():
            time.sleep(1.0)
        t2 = time.time()
        self.launch_time = t2 - t1
        logger.info(f"DEBUG: ComfyUI server started in {self.launch_time:.2f}s")

    def _execute(
        self,
        workflow_name: str,
        args: dict,
        user: str = None,
        agent: str = None,
        session: str = None,
    ):
        try:
            logger.info("\n" + "=" * 60)
            logger.info(f"{' ' * 10}STARTING NEW TASK: {workflow_name}{' ' * 10}")
            logger.info("=" * 60 + "\n")

            # Debug: Get detailed server stats
            logger.info("DEBUG: Getting detailed ComfyUI server stats...")
            server_check_start = time.time()
            server_stats = self.get_server_stats()
            server_check_time = time.time() - server_check_start

            # Check if we got valid stats without error
            if "error" not in server_stats:
                logger.info(
                    f"DEBUG: ComfyUI server is running at {self.server_address} (stats check took {server_check_time:.3f}s)"
                )

                # Print memory usage if available
                if server_stats.get("memory_usage"):
                    mem = server_stats["memory_usage"]
                    logger.info(
                        f"DEBUG: VRAM Usage: {mem['vram_used_gb']:.2f}GB / {mem['vram_total_gb']:.2f}GB ({mem['vram_utilization']:.1f}%)"
                    )

                # Print queue information if available
                if server_stats.get("queue"):
                    queue = server_stats["queue"]
                    logger.info(
                        f"DEBUG: Queue status - Running: {queue.get('running_size', 'N/A')}, Pending: {queue.get('pending_size', 'N/A')}"
                    )

                # Print processing times if available
                if server_stats.get("processing_times"):
                    times = server_stats["processing_times"]
                    logger.info(
                        f"DEBUG: Recent processing times - Avg: {times['average']:.2f}s, Min: {times['min']:.2f}s, Max: {times['max']:.2f}s"
                    )
            else:
                logger.info(
                    f"DEBUG: WARNING - ComfyUI server is NOT running at {self.server_address}!"
                )
                logger.info(f"DEBUG: Stats check error: {server_stats['error']}")

                logger.info("DEBUG: Attempting to restart server...")
                try:
                    self._start()
                    logger.info("DEBUG: Server restarted successfully")
                    # Verify server stats after restart
                    server_stats = self.get_server_stats()
                    if "error" in server_stats:
                        raise Exception("Server restart failed - still not responding")
                except Exception as e:
                    logger.info(f"DEBUG: Failed to restart server: {e}")
                    raise Exception(
                        f"ComfyUI server is not running and restart failed: {e}"
                    )

            eden_utils.log_memory_info()
            tool_path = f"/root/workspace/workflows/{workflow_name}"
            tool = Tool.from_yaml(f"{tool_path}/api.yaml")
            workflow = json.load(open(f"{tool_path}/workflow_api.json", "r"))
            self._validate_comfyui_args(workflow, tool)
            workflow = self._inject_args_into_workflow(workflow, tool, args)

            # Debug: Check server again before queuing prompt
            server_stats = self.get_server_stats()
            if "error" in server_stats:
                logger.info(
                    "DEBUG: ERROR - Server not responding before queuing prompt!"
                )
                raise Exception(
                    "ComfyUI server stopped responding before queuing prompt"
                )

            logger.info("DEBUG: Queuing prompt to ComfyUI server...")
            queue_start = time.time()
            prompt_id = self._queue_prompt(workflow)["prompt_id"]
            queue_time = time.time() - queue_start
            logger.info(
                f"DEBUG: Prompt queued successfully (took {queue_time:.3f}s), prompt_id: {prompt_id}"
            )

            outputs = self._get_outputs(prompt_id)
            output = outputs[str(tool.comfyui_output_node_id)]
            if not output:
                raise Exception(
                    f"No output found for {workflow_name} at output node {tool.comfyui_output_node_id}"
                )
            logger.info("---- comfyui output ----")
            result = {"output": output}
            if tool.comfyui_intermediate_outputs:
                result["intermediate_outputs"] = {
                    key: outputs[str(node_id)]
                    for key, node_id in tool.comfyui_intermediate_outputs.items()
                }
            logger.info(result)
            return result
        except modal.exception.InputCancellation:
            logger.info("Modal Task Cancelled")
            logger.info("Interrupting ComfyUI")
            self._interrupt()
            logger.info("ComfyUI interrupted")
        except Exception as error:
            logger.info("ComfyUI pipeline error: ", error)
            raise

    @modal.method()
    def run(self, tool_key: str, args: dict):
        result = self._execute(tool_key, args)
        return eden_utils.upload_result(result)

    @modal.method()
    @task_handler_method
    async def run_task(
        self,
        tool_key: str,
        args: dict,
        user: str = None,
        agent: str = None,
        session: str = None,
    ):
        return self._execute(tool_key, args, user, agent, session)

    @modal.enter()
    def enter(self):
        self._start()

    def _is_server_running(self):
        try:
            start_time = time.time()
            url = f"http://{self.server_address}/history/123"

            with urllib.request.urlopen(
                urllib.request.Request(url), timeout=5
            ) as response:
                response_time = time.time() - start_time
                is_running = response.status == 200

                if is_running:
                    logger.info(
                        f"DEBUG: Server check successful - response time: {response_time:.3f}s"
                    )
                else:
                    logger.info(
                        f"DEBUG: Server responded with unexpected status: {response.status}"
                    )

                return is_running

        except urllib.error.URLError as e:
            response_time = time.time() - start_time
            if hasattr(e, "reason"):
                logger.info(f"DEBUG: Failure reason: {e.reason}, time: {response_time}")
            return False
        except socket.timeout:
            response_time = time.time() - start_time
            logger.info(f"DEBUG: Server check timed out after {response_time:.3f}s")
            return False
        except Exception as e:
            response_time = time.time() - start_time
            logger.info(
                f"DEBUG: Server check failed with unexpected error after {response_time:.3f}s: {e}"
            )
            logger.info(f"DEBUG: Error type: {type(e).__name__}")
            return False

    def _queue_prompt(self, prompt):
        data = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            "http://{}/prompt".format(self.server_address), data=data
        )
        return json.loads(urllib.request.urlopen(req).read())

    def _get_history(self, prompt_id):
        with urllib.request.urlopen(
            "http://{}/history/{}".format(self.server_address, prompt_id)
        ) as response:
            return json.loads(response.read())

    def _interrupt(self):
        try:
            logger.info("Interrupting ComfyUI ...")
            with urllib.request.urlopen(
                f"http://{self.server_address}/interrupt"
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to interrupt ComfyUI: {response.status}")
        except Exception as e:
            logger.info(f"Error interrupting ComfyUI: {e}")
            raise

    def _stop_server(self):
        """Stop the ComfyUI server by killing the process."""
        try:
            logger.info("DEBUG: Stopping ComfyUI server...")
            # Kill any existing ComfyUI processes
            result = subprocess.run(
                ["pkill", "-f", "main.py"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("DEBUG: ComfyUI server process killed")
            else:
                logger.info("DEBUG: No ComfyUI server process found to kill")

            # Wait a moment for the process to fully terminate
            time.sleep(2)
        except Exception as e:
            logger.info(f"Error stopping ComfyUI server: {e}")
            raise

    def _get_history(self, prompt_id):
        """
        Get history for a specific prompt ID.

        Args:
            prompt_id: The ID of the prompt to check

        Returns:
            dict: The history data for the prompt

        Raises:
            urllib.error.URLError: If there's a connection error
            json.JSONDecodeError: If the response isn't valid JSON
            Exception: For other unexpected errors
        """
        try:
            url = f"http://{self.server_address}/history/{prompt_id}"

            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    logger.info(f"Warning: Unexpected status code {response.status}")

                response_data = response.read()
                try:
                    history_data = json.loads(response_data)

                    if not history_data or (prompt_id not in history_data):
                        return {}

                    return history_data

                except json.JSONDecodeError as e:
                    logger.info(f"Failed to decode JSON response from {url}")
                    logger.info(f"Error decoding JSON response: {e}")
                    logger.info(
                        f"Raw response data: {response_data[:200]}..."
                    )  # Print first 200 chars
                    raise

        except urllib.error.URLError as e:
            logger.info(f"Connection error while fetching history: {e}")
            if hasattr(e, "reason"):
                logger.info(f"Failure reason: {e.reason}")
            raise

        except Exception as e:
            logger.info(f"Unexpected error in _get_history: {str(e)}")
            logger.info(f"Error type: {type(e).__name__}")
            raise

    def _get_outputs(self, prompt_id, poll_interval=1):
        while True:
            outputs = {}
            history = self._get_history(prompt_id)
            if not history or not history.get(prompt_id):
                time.sleep(poll_interval)
                continue

            history = history[prompt_id]
            status = history["status"]
            status_str = status.get("status_str")
            if status_str == "error":
                messages = status.get("messages")
                errors = [
                    f"ComfyUI Error: {v.get('node_type')} {v.get('exception_type')}, {v.get('exception_message')}"
                    for k, v in messages
                    if k == "execution_error"
                ]
                error_str = ", ".join(errors)
                logger.info("error", error_str)
                raise Exception(error_str)

            for _ in history["outputs"]:
                for node_id in history["outputs"]:
                    node_output = history["outputs"][node_id]
                    if "images" in node_output:
                        outputs[node_id] = [
                            os.path.join(
                                "output", image["subfolder"], image["filename"]
                            )
                            for image in node_output["images"]
                        ]
                    elif "gifs" in node_output:
                        outputs[node_id] = [
                            os.path.join(
                                "output", video["subfolder"], video["filename"]
                            )
                            for video in node_output["gifs"]
                        ]
                    elif "audio" in node_output:
                        outputs[node_id] = [
                            os.path.join(
                                "output", audio["subfolder"], audio["filename"]
                            )
                            for audio in node_output["audio"]
                        ]

            logger.info("comfy outputs", outputs)
            if not outputs:
                raise Exception("No outputs found")

            return outputs

    def _inject_embedding_mentions_sdxl(
        self, text, embedding_trigger, embeddings_filename, lora_mode, lora_strength
    ):
        reference = f"(embedding:{embeddings_filename})"

        # Make two deep copies of the input text:
        user_prompt = copy.deepcopy(text)
        lora_prompt = copy.deepcopy(text)

        if lora_mode == "face" or lora_mode == "object" or lora_mode == "concept":
            # Match all variations of the embedding_trigger:
            pattern = r"(<{0}>|<{1}>|{0}|{1})".format(
                re.escape(embedding_trigger), re.escape(embedding_trigger.lower())
            )
            lora_prompt = re.sub(pattern, reference, lora_prompt, flags=re.IGNORECASE)
            lora_prompt = re.sub(
                r"(<concept>)", reference, lora_prompt, flags=re.IGNORECASE
            )
            if lora_mode == "face":
                base_word = "person"
            else:
                base_word = "object"

            user_prompt = re.sub(pattern, base_word, user_prompt, flags=re.IGNORECASE)
            user_prompt = re.sub(
                r"(<concept>)", base_word, user_prompt, flags=re.IGNORECASE
            )

        if reference not in lora_prompt:  # Make sure the concept is always triggered:
            if lora_mode == "style":
                lora_prompt = f"in the style of {reference}, {lora_prompt}"
            else:
                lora_prompt = f"{reference}, {lora_prompt}"

        return user_prompt, lora_prompt

    def _inject_embedding_mentions_flux(
        self, text, embedding_trigger, lora_trigger_text
    ):
        orig_text = orig_text = str(text)
        if not embedding_trigger:  # Handles both None and empty string
            if lora_trigger_text:
                text = re.sub(
                    r"(<concept>)", lora_trigger_text, text, flags=re.IGNORECASE
                )
        else:
            pattern = r"(<{0}>|<{1}>|{0}|{1})".format(
                re.escape(embedding_trigger), re.escape(embedding_trigger.lower())
            )
            text = re.sub(pattern, lora_trigger_text, text, flags=re.IGNORECASE)
            text = re.sub(r"(<concept>)", lora_trigger_text, text, flags=re.IGNORECASE)

        if lora_trigger_text:
            if lora_trigger_text not in text:
                text = f"{lora_trigger_text}, {text}"

        logger.info("Performed FLUX LoRA trigger injection:")
        logger.info(orig_text)
        logger.info("Changed to:")
        logger.info(text)

        return text

    def _transport_lora_flux(self, lora_url: str):
        loras_folder = "/root/models/loras"

        logger.info("tl download lora", lora_url)
        if not re.match(r"^https?://", lora_url):
            raise ValueError(f"Lora URL Invalid: {lora_url}")

        lora_filename = lora_url.split("/")[-1]
        lora_path = os.path.join(loras_folder, lora_filename)
        logger.info("tl destination folder", loras_folder)

        if os.path.exists(lora_path):
            logger.info("Lora safetensors file already extracted. Skipping.")
        else:
            eden_utils.download_file(lora_url, lora_path)
            if not os.path.exists(lora_path):
                raise FileNotFoundError(
                    f"The LoRA tar file {lora_path} does not exist."
                )

        logger.info("destination path", lora_path)
        logger.info("lora filename", lora_filename)

        return lora_filename

    def _transport_lora_sdxl(self, lora_url: str):
        downloads_folder = "/root/downloads"
        loras_folder = "/root/models/loras"
        embeddings_folder = "/root/models/embeddings"

        logger.info("tl download lora", lora_url)
        if not re.match(r"^https?://", lora_url):
            raise ValueError(f"Lora URL Invalid: {lora_url}")

        lora_tar_filename = lora_url.split("/")[-1]
        name = lora_tar_filename.split(".")[0]  # e.g., 'hotmale' from 'hotmale.tar.gz'
        destination_folder = os.path.join(downloads_folder, name)
        logger.info("tl destination folder", destination_folder)

        if os.path.exists(destination_folder):
            logger.info("LORA bundle already extracted, skipping download")
        else:
            try:
                lora_tarfile_path = eden_utils.download_file(
                    lora_url, f"/root/downloads/{lora_tar_filename}"
                )
                if not os.path.exists(lora_tarfile_path):
                    raise FileNotFoundError(
                        f"The LoRA tar file {lora_tarfile_path} does not exist."
                    )
                with tarfile.open(lora_tarfile_path, "r:*") as tar:
                    tar.extractall(path=destination_folder)
                    logger.info("Extraction complete.")
            except Exception as e:
                raise IOError(f"Failed to extract tar file: {e}")

        extracted_files = os.listdir(destination_folder)
        logger.info(
            f"Found {len(extracted_files)} files in LORA bundle: {extracted_files}"
        )

        lora_filename = None
        embeddings_filename = None

        # 1. Standard patterns first
        lora_pattern = re.compile(r".*_lora\.safetensors$")
        lora_filename = next(
            (f for f in extracted_files if lora_pattern.match(f)), None
        )

        embeddings_pattern = re.compile(r".*_embeddings\.safetensors$")
        embeddings_filename = next(
            (f for f in extracted_files if embeddings_pattern.match(f)), None
        )

        if lora_filename:
            logger.info(f"Found standard LoRA file: {lora_filename}")
        if embeddings_filename:
            logger.info(f"Found standard embeddings file: {embeddings_filename}")

        # 2. Fallbacks for lora_filename if not found by standard pattern
        if not lora_filename:
            logger.info(
                "Standard LoRA file pattern not matched. Trying fallbacks for LoRA file..."
            )
            # Fallback 2a: General *.safetensors that isn't an embeddings file
            lora_filename = next(
                (
                    f
                    for f in extracted_files
                    if f.endswith(".safetensors")
                    and not embeddings_pattern.match(f)  # Not a standard embedding
                    and (
                        not embeddings_filename or f != embeddings_filename
                    )  # Not the one already found as standard embedding
                    and "embedding"
                    not in f.lower()  # General check against "embedding" in name
                    and "embeddings" not in f.lower()
                ),
                None,
            )
            if lora_filename:
                logger.info(
                    f"Found LoRA file by general .safetensors fallback: {lora_filename}"
                )
            else:
                # Fallback 2b: Specific 'lora.safetensors' for very old format
                if "lora.safetensors" in extracted_files:
                    lora_filename = "lora.safetensors"
                    logger.info(
                        f"Found LoRA file by specific name 'lora.safetensors': {lora_filename}"
                    )

        # 3. Fallbacks for embeddings_filename if not found by standard pattern
        if not embeddings_filename:
            logger.info(
                "Standard embeddings file pattern not matched. Trying fallbacks for embeddings file..."
            )
            # Fallback 3a: Specific 'embeddings.pti'
            if "embeddings.pti" in extracted_files:
                embeddings_filename = "embeddings.pti"
                logger.info(
                    f"Found embeddings file by specific name 'embeddings.pti': {embeddings_filename}"
                )
            elif "embedding.pti" in extracted_files:  # Also check singular version
                embeddings_filename = "embedding.pti"
                logger.info(
                    f"Found embeddings file by specific name 'embedding.pti': {embeddings_filename}"
                )
            # Fallback 3b: (No generic .safetensors fallback for embeddings to avoid ambiguity with other files)

        # Convert .pti to .safetensors if necessary
        if embeddings_filename and embeddings_filename.endswith(".pti"):
            logger.info(
                f"Found .pti embeddings file: {embeddings_filename}. Attempting conversion."
            )
            base_embedding_name = embeddings_filename[:-4]  # Remove .pti
            converted_embeddings_filename = base_embedding_name + ".safetensors"

            input_pti_path = os.path.join(destination_folder, embeddings_filename)
            output_safetensors_path = os.path.join(
                destination_folder, converted_embeddings_filename
            )

            conversion_success = eden_utils.convert_pti_to_safetensors(
                input_pti_path, output_safetensors_path
            )
            if conversion_success:
                logger.info(
                    f"Successfully converted {embeddings_filename} to {converted_embeddings_filename}"
                )
                if converted_embeddings_filename not in extracted_files:
                    extracted_files.append(converted_embeddings_filename)
                embeddings_filename = converted_embeddings_filename
            else:
                logger.info(f"Conversion of {embeddings_filename} failed. Check logs.")

        # --- Determine lora_mode and embedding_trigger ---
        training_args_filename = next(
            (f for f in extracted_files if f == "training_args.json"), None
        )
        lora_mode = "style"
        embedding_trigger = None

        if training_args_filename:
            logger.info(f"Found training_args.json: {training_args_filename}")
            with open(
                os.path.join(destination_folder, training_args_filename), "r"
            ) as f:
                training_args = json.load(f)
                lora_mode = training_args.get(
                    "concept_mode", training_args.get("mode", "style")
                )
                embedding_trigger = training_args.get("name")
                if not embedding_trigger:
                    logger.info(
                        f"Warning: 'name' for embedding_trigger not found in training_args.json. Will use archive name '{name}'."
                    )
                    embedding_trigger = name
        else:
            logger.info("training_args.json not found.")
            lora_mode = "style"  # Default lora_mode if no training_args.json
            embedding_trigger = name  # Default trigger to archive name
            logger.info(
                f"Defaulting lora_mode to '{lora_mode}' and embedding_trigger to archive name '{embedding_trigger}' due to missing training_args.json."
            )

        # --- Final checks for file existence ---
        if not lora_filename:
            raise FileNotFoundError(
                f"Unable to find a suitable LoRA file (e.g., *_lora.safetensors or lora.safetensors) in extracted files: {extracted_files}"
            )

        if (
            lora_filename not in extracted_files
        ):  # Should not happen if logic above is correct
            raise FileNotFoundError(
                f"LoRA file '{lora_filename}' was determined but not in extracted files: {extracted_files}"
            )

        if embeddings_filename and embeddings_filename not in extracted_files:
            raise FileNotFoundError(
                f"Embeddings file '{embeddings_filename}' was determined but not in extracted files: {extracted_files}"
            )

        # If no embeddings file was found, but LoRA mode suggests it's needed (not 'style'), raise an error.
        if not embeddings_filename and lora_mode and lora_mode != "style":
            raise FileNotFoundError(
                f"LoRA mode is '{lora_mode}', which typically requires an embeddings file, but none was found in package {extracted_files}. "
                f"Searched for standard patterns (e.g. *_embeddings.safetensors) and fallbacks (e.g. embeddings.pti)."
            )

        logger.info(f"LORA mode: {lora_mode}")
        logger.info(f"Using LORA file: {lora_filename}")
        if embeddings_filename:
            logger.info(f"Using embeddings file: {embeddings_filename}")
        else:
            logger.info(
                "No embeddings file will be used (e.g., for style LoRA or if not found and mode allows)."
            )
        logger.info(f"Embedding trigger: {embedding_trigger}")

        # --- File operations ---
        if not os.path.exists(loras_folder):
            os.makedirs(loras_folder)
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

        # Copy lora file to loras folder
        lora_path = os.path.join(destination_folder, lora_filename)
        lora_copy_path = os.path.join(loras_folder, lora_filename)
        shutil.copy(lora_path, lora_copy_path)
        logger.info(f"LoRA {lora_path} has been copied to {lora_copy_path}")

        # Copy embedding file to embeddings folder (if it exists and was found)
        if embeddings_filename:
            embeddings_path = os.path.join(destination_folder, embeddings_filename)
            embeddings_copy_path = os.path.join(embeddings_folder, embeddings_filename)
            shutil.copy(embeddings_path, embeddings_copy_path)
            logger.info(
                f"Embeddings {embeddings_path} has been copied to {embeddings_copy_path}"
            )

        return lora_filename, embeddings_filename, embedding_trigger, lora_mode

    def _url_to_filename(self, url):
        filename = url.split("/")[-1]
        filename = re.sub(r"\?.*$", "", filename)
        max_length = 255
        if len(filename) > max_length:  # ensure filename is not too long
            name, ext = os.path.splitext(filename)
            filename = name[: max_length - len(ext)] + ext
        return filename

    def _validate_comfyui_args(self, workflow, tool):
        for key, comfy_param in tool.comfyui_map.items():
            node_id, field, subfield, remaps = (
                str(comfy_param.node_id),
                str(comfy_param.field),
                str(comfy_param.subfield),
                comfy_param.remap,
            )
            subfields = [s.strip() for s in subfield.split(",")]
            for subfield in subfields:
                if (
                    node_id not in workflow
                    or field not in workflow[node_id]
                    or subfield not in workflow[node_id][field]
                ):
                    raise Exception(
                        f"Node ID {node_id}, field {field}, subfield {subfield} not found in workflow"
                    )
            for remap in remaps or []:
                subfields = [s.strip() for s in str(remap.subfield).split(",")]
                for subfield in subfields:
                    if (
                        str(remap.node_id) not in workflow
                        or str(remap.field) not in workflow[str(remap.node_id)]
                        or subfield
                        not in workflow[str(remap.node_id)][str(remap.field)]
                    ):
                        raise Exception(
                            f"Node ID {remap.node_id}, field {remap.field}, subfield {subfield} not found in workflow"
                        )
                param = tool.model.model_fields[key]
                # has_choices = isinstance(param.annotation, type) and issubclass(param.annotation, Enum)
                # if not has_choices:
                #     raise Exception(f"Remap parameter {key} has no original choices")
                # choices = [e.value for e in param.annotation]
                choices = param.json_schema_extra.get("choices")
                if not all(choice in choices for choice in remap.map.keys()):
                    raise Exception(
                        f"Remap parameter {key} has invalid choices: {remap.map}"
                    )
                if not all(choice in remap.map.keys() for choice in choices):
                    raise Exception(
                        f"Remap parameter {key} is missing original choices: {choices}"
                    )

    def _inject_args_into_workflow(self, workflow, tool, args):
        base_model = "unknown"

        # Helper function to validate and normalize URLs
        def validate_url(url):
            if not isinstance(url, str):
                raise ValueError(f"Invalid URL type: {type(url)}. Expected string.")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            return url

        logger.info("===== Injecting comfyui args into workflow =====")
        logger.info(args)

        embedding_triggers = {"lora": None, "lora2": None}
        lora_trigger_texts = {"lora": None, "lora2": None}

        # First pass: Download and process all files
        for key, param in tool.model.model_fields.items():
            metadata = param.json_schema_extra or {}
            file_type = metadata.get("file_type")
            is_array = metadata.get("is_array")
            if file_type and any(
                t in ["image", "video", "audio"] for t in file_type.split("|")
            ):
                if not args.get(key):
                    continue
                if is_array:
                    urls = [validate_url(url) for url in args.get(key)]
                    args[key] = (
                        [
                            eden_utils.download_file(
                                url, f"/root/input/{self._url_to_filename(url)}"
                            )
                            if url
                            else None
                            for url in urls
                        ]
                        if urls
                        else None
                    )
                else:
                    url = validate_url(args.get(key))
                    local_path = (
                        eden_utils.download_file(
                            url, f"/root/input/{self._url_to_filename(url)}"
                        )
                        if url
                        else None
                    )
                    logger.info(f"Downloaded {url} to {local_path}")
                    args[key] = local_path

            elif file_type == "lora":
                lora_id = args.get(key)

                if not lora_id:
                    args[key] = None
                    args[f"{key}_strength"] = 0
                    logger.info(f"DISABLING {key}")
                    continue

                logger.info(f"Found {key} LORA ID: {lora_id}")

                models = get_collection("models3")
                lora = models.find_one({"_id": ObjectId(lora_id)})

                if not lora:
                    raise Exception(
                        f"Lora {key} with id: {lora_id} not found in DB {db}!"
                    )

                base_model = lora.get("base_model")
                lora_url = lora.get("checkpoint")

                if not lora_url:
                    raise Exception(f"Lora {lora_id} has no checkpoint")
                else:
                    logger.info("LORA URL", lora_url)

                lora_url = get_full_url(lora_url)
                logger.info("lora url", lora_url)
                logger.info("base model", base_model)

                if base_model == "sdxl":
                    lora_filename, embeddings_filename, embedding_trigger, lora_mode = (
                        self._transport_lora_sdxl(lora_url)
                    )
                elif base_model == "flux-dev":
                    lora_filename = self._transport_lora_flux(lora_url)
                    embedding_triggers[key] = lora.get("args", {}).get("name")
                    try:
                        lora_trigger_texts[key] = lora.get("lora_trigger_text")
                    except Exception:  # old flux LoRA's:
                        lora_trigger_texts[key] = lora.get("args", {}).get(
                            "caption_prefix"
                        )

                args[key] = lora_filename

        # Second pass: Inject the downloaded files and other parameters into workflow
        for key, comfyui in tool.comfyui_map.items():
            value = args.get(key)
            if value is None:
                continue

            if key == "no_token_prompt":
                continue

            # if there's a lora, replace mentions with embedding name
            if key == "prompt":
                if "flux" in base_model:
                    if not (
                        ("subj_1" in value) and ("subj_2" in value)
                    ):  # Skip trigger injection
                        for lora_key in ["lora", "lora2"]:
                            if args.get(f"use_{lora_key}", False):
                                lora_strength = args.get(f"{lora_key}_strength", 0.7)
                                value = self._inject_embedding_mentions_flux(
                                    value,
                                    embedding_triggers[lora_key],
                                    lora_trigger_texts[lora_key],
                                )
                elif base_model == "sdxl":
                    if embedding_trigger:
                        lora_strength = args.get("lora_strength", 0.7)
                        no_token_prompt, value = self._inject_embedding_mentions_sdxl(
                            value,
                            embedding_trigger,
                            embeddings_filename,
                            lora_mode,
                            lora_strength,
                        )

                        if "no_token_prompt" in args:
                            no_token_mapping = next(
                                (
                                    comfy_param
                                    for key, comfy_param in tool.comfyui_map.items()
                                    if key == "no_token_prompt"
                                ),
                                None,
                            )
                            if no_token_mapping:
                                logger.info(
                                    "Updating no_token_prompt for SDXL: ",
                                    no_token_prompt,
                                )
                                workflow[str(no_token_mapping.node_id)][
                                    no_token_mapping.field
                                ][no_token_mapping.subfield] = no_token_prompt

                logger.info("====> Final updated prompt for workflow: ", value)

            # Handle preprocessing
            if comfyui.preprocessing is not None:
                if comfyui.preprocessing == "csv":
                    value = ",".join(value)
                elif comfyui.preprocessing == "concat":
                    value = ";\n".join(value)
                elif comfyui.preprocessing == "folder":
                    temp_subfolder = tempfile.mkdtemp(dir="/root/input")
                    if isinstance(value, list):
                        for i, file in enumerate(value):
                            filename = f"{i:06d}_{os.path.basename(file)}"
                            new_path = os.path.join(temp_subfolder, filename)
                            shutil.copy(file, new_path)
                    else:
                        shutil.copy(value, temp_subfolder)
                    value = temp_subfolder

            logger.info(f"Injecting {key} = {value}")
            node_id, field, subfield = (
                str(comfyui.node_id),
                str(comfyui.field),
                str(comfyui.subfield),
            )
            subfields = [s.strip() for s in subfield.split(",")]
            for subfield in subfields:
                logger.info("inject", node_id, field, subfield, " = ", value)
                workflow[node_id][field][subfield] = value

            # Handle remaps
            for remap in comfyui.remap or []:
                subfields = [s.strip() for s in str(remap.subfield).split(",")]
                for subfield in subfields:
                    output_value = remap.map.get(value)
                    if output_value is not None:
                        logger.info(
                            f"Remapping {key}={value} to {remap.node_id}.{remap.field}.{subfield}={output_value}"
                        )
                        workflow[str(remap.node_id)][remap.field][subfield] = (
                            output_value
                        )

        return workflow

    def get_server_stats(self):
        """
        Get comprehensive operational statistics from the ComfyUI server
        """
        base_url = f"http://{self.server_address}"
        stats = {
            "system": None,
            "queue": None,
            "history": None,
            "memory_usage": None,
            "processing_times": None,
        }

        try:
            # Get system statistics
            response = urllib.request.urlopen(f"{base_url}/system_stats", timeout=5)
            if response.status == 200:
                stats["system"] = json.loads(response.read())

                # Extract useful memory metrics
                if stats["system"] and "devices" in stats["system"]:
                    for device in stats["system"]["devices"]:
                        if device["type"] == "cuda":
                            vram_total = device.get("vram_total", 0)
                            vram_free = device.get("vram_free", 0)
                            stats["memory_usage"] = {
                                "vram_total_gb": round(vram_total / (1024**3), 2),
                                "vram_used_gb": round(
                                    (vram_total - vram_free) / (1024**3), 2
                                ),
                                "vram_utilization": round(
                                    (vram_total - vram_free) / vram_total * 100, 2
                                )
                                if vram_total > 0
                                else 0,
                            }

            # Get queue information
            response = urllib.request.urlopen(f"{base_url}/queue", timeout=5)
            if response.status == 200:
                stats["queue"] = json.loads(response.read())

            # Get recent history (last 5 items)
            response = urllib.request.urlopen(
                f"{base_url}/history?max_items=5", timeout=5
            )
            if response.status == 200:
                history_data = json.loads(response.read())
                stats["history"] = history_data

                # Calculate average processing times if history exists
                if history_data and len(history_data) > 0:
                    processing_times = []
                    for item in history_data:
                        if (
                            "exec_info" in item
                            and "execution_time" in item["exec_info"]
                        ):
                            processing_times.append(item["exec_info"]["execution_time"])

                    if processing_times:
                        stats["processing_times"] = {
                            "average": sum(processing_times) / len(processing_times),
                            "min": min(processing_times),
                            "max": max(processing_times),
                            "samples": len(processing_times),
                        }

            return stats

        except Exception as e:
            return {"error": str(e)}


@app.cls(
    image=image,
    gpu=gpu,
    cpu=8.0,
    volumes={"/data": downloads_vol},
    max_containers=10,
    scaledown_window=60,
    min_containers=0,
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
class ComfyUIPremium(ComfyUI):
    pass


@app.cls(
    image=image,
    gpu=gpu,
    cpu=8.0,
    volumes={"/data": downloads_vol},
    max_containers=1,
    scaledown_window=60,
    min_containers=0,
    timeout=3600,
)
class ComfyUIBasic(ComfyUI):
    pass


@app.cls(
    image=image,
    gpu=gpu,
    cpu=8.0,
    volumes={"/data": downloads_vol},
    max_containers=50,
    scaledown_window=60,
    min_containers=0,
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
class ComfyUITempleAbyss(ComfyUI):
    pass


@app.local_entrypoint()
def run():
    comfyui = ComfyUI()
    return comfyui
