"""
DB=STAGE SKIP_TESTS=1 WORKSPACE=audio modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=batch_tools modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=flux modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=img_tools modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=mars_exclusive modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=sd3 modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=txt2img modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=video modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=video2 modal deploy comfyui.py
DB=STAGE SKIP_TESTS=1 WORKSPACE=video_mochi modal deploy comfyui.py

DB=PROD WORKSPACE=audio modal deploy comfyui.py
DB=PROD WORKSPACE=batch_tools modal deploy comfyui.py
DB=PROD WORKSPACE=flux modal deploy comfyui.py
DB=PROD WORKSPACE=img_tools modal deploy comfyui.py
DB=PROD WORKSPACE=mars_exclusive modal deploy comfyui.py
DB=PROD WORKSPACE=sd3 modal deploy comfyui.py
DB=PROD WORKSPACE=txt2img modal deploy comfyui.py
DB=PROD WORKSPACE=video modal deploy comfyui.py
DB=PROD WORKSPACE=video2 modal deploy comfyui.py
DB=PROD WORKSPACE=video_mochi modal deploy comfyui.py

"""

from urllib.error import URLError
from bson import ObjectId
from pprint import pprint
from pathlib import Path
import os
import re
import git
import time
import json
import glob
import copy
import modal
import shutil
import urllib
import tarfile
import pathlib
import tempfile
import subprocess
import traceback

import eve.eden_utils as eden_utils
from eve.tool import Tool
from eve.mongo import get_collection
from eve.task import task_handler_method
from eve.s3 import get_full_url

GPUs = {
    "A100": modal.gpu.A100(),
    "A100-80GB": modal.gpu.A100(size="80GB")
}

if not os.getenv("WORKSPACE"):
    raise Exception("No workspace selected")

db = os.getenv("DB", "STAGE").upper()
workspace_name = os.getenv("WORKSPACE")
app_name = f"comfyui-{workspace_name}-{db}"
test_workflows = os.getenv("WORKFLOWS")
root_workflows_folder = "../private_workflows" if os.getenv("PRIVATE") else "../workflows"
test_all = True if os.getenv("TEST_ALL") else False
specific_test = os.getenv("SPECIFIC_TEST") if os.getenv("SPECIFIC_TEST") else ""
skip_tests = os.getenv("SKIP_TESTS")

# Run a bunch of checks to verify input args:
if test_all and specific_test:
    print(f"WARNING: can't have both TEST_ALL and SPECIFIC_TEST at the same time...")
    print(f"Running TEST_ALL instead")
    specific_test = ""

print("========================================")
print(f"db: {db}")
print(f"workspace: {workspace_name}")
print(f"test_workflows: {test_workflows}")
print(f"test_all: {test_all}")
print(f"specific_test: {specific_test}")
print(f"skip_tests: {skip_tests}")
print("========================================")

if not test_workflows and workspace_name and not test_all:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!! WARNING: You are deploying a workspace without TEST_ALL !!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

def install_comfyui():
    snapshot = json.load(open("/root/workspace/snapshot.json", 'r'))
    comfyui_commit_sha = snapshot["comfyui"]
    subprocess.run(["git", "init", "."], check=True)
    subprocess.run(["git", "remote", "add", "--fetch", "origin", "https://github.com/comfyanonymous/ComfyUI"], check=True)
    subprocess.run(["git", "checkout", comfyui_commit_sha], check=True)
    subprocess.run(["pip", "install", "xformers!=0.0.18", "-r", "requirements.txt", "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=True)


def install_custom_nodes():
    snapshot = json.load(open("/root/workspace/snapshot.json", 'r'))
    custom_nodes = snapshot["git_custom_nodes"]
    for url, node in custom_nodes.items():
        print(f"Installing custom node {url} with hash {node['hash']}") 
        install_custom_node_with_retries(url, node['hash'])
    post_install_commands = snapshot.get("post_install_commands", [])
    for cmd in post_install_commands:
        os.system(cmd)

def install_custom_node_with_retries(url, hash, max_retries=3): 
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempt {attempt + 1}: Installing {url}")
            install_custom_node(url, hash)
            print(f"Successfully installed {url}")
            return
        except Exception as e:
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed because: {str(e)}")
                print(f"Exception type: {type(e)}")
                traceback.print_exc()  # This will print the full stack trace
                print("Retrying...")
                time.sleep(5)
            else:
                print(f"All attempts failed. Final error: {str(e)}")
                traceback.print_exc()
                raise

def install_custom_node(url, hash):
    repo_name = url.split("/")[-1].split(".")[0]
    repo_path = f"custom_nodes/{repo_name}"
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
                    if "with-cupy" in requirements_path: # hack for ComfyUI-Frame-Interpolation, don't use CuPy
                        continue
                    subprocess.run(["pip", "install", "-r", requirements_path], check=True)
                except Exception as e:
                    print(f"Error installing requirements: {e}")
                   
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
            print(f"Removing existing repository at {target_path}")
            shutil.rmtree(target_path)
        else:
            print(f"Repository already exists at {target_path}, skipping clone")
            return
            
    print(f"Cloning repository {repo_url} to {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Clone directly to the specified target path
        subprocess.run(['git', 'clone', repo_url, str(target_path)], 
                     check=True, 
                     capture_output=True)
        downloads_vol.commit()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error cloning repository {repo_url}: {e.stderr.decode()}")

def download_file(url, target_path, force=False):
    """Download a single file to the target path."""
    if target_path.is_file() and not force:
        print(f"Skipping download, getting {target_path} from cache")
        return
        
    print(f"Downloading {url} to {target_path}")
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

def download_files(force_redownload=False):
    """
    Main function to process downloads from downloads.json.
    
    Args:
        force_redownload (bool): If True, force redownload and overwrite existing files.
    """
    downloads = json.load(open("/root/workspace/downloads.json", 'r'))
    
    for path, source in downloads.items():
        comfy_path = pathlib.Path("/root") / path
        vol_path = pathlib.Path("/data") / path
        
        # Skip if target already exists and force_redownload is False
        if (comfy_path.is_file() or comfy_path.is_dir()) and not force_redownload:
            print(f"Path already exists at {comfy_path}, skipping")
            continue
        
        try:
            if source.startswith("git clone "):
                # Extract the repository URL after "git clone "
                repo_url = source[10:].strip()
                handle_repo_download(repo_url, vol_path, comfy_path, force=force_redownload)
            else:
                handle_file_download(source, vol_path, comfy_path, force=force_redownload)
                
            if not comfy_path.exists():
                raise Exception(f"No file/directory found at {comfy_path}")
                
        except Exception as e:
            raise Exception(f"Error processing {path}: {e}")

root_dir = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"COMFYUI_PATH": "/root", "COMFYUI_MODEL_PATH": "/root/models"}) 
    .env({"TEST_ALL": os.getenv("TEST_ALL")})
    .env({"SPECIFIC_TEST": os.getenv("SPECIFIC_TEST")})
    .apt_install("git", "git-lfs", "libgl1-mesa-glx", "libglib2.0-0", "libmagic1", "ffmpeg", "libegl1")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("diffusers==0.31.0", "psutil")
    .env({"WORKSPACE": workspace_name}) 
    .copy_local_file(f"{root_workflows_folder}/workspaces/{workspace_name}/snapshot.json", "/root/workspace/snapshot.json")
    .copy_local_file(f"{root_workflows_folder}/workspaces/{workspace_name}/downloads.json", "/root/workspace/downloads.json")
    .run_function(install_comfyui) #, force_build=True)
    .run_function(install_custom_nodes, gpu=modal.gpu.A100())
    .pip_install("moviepy==1.0.3")
    .copy_local_dir(f"{root_workflows_folder}/workspaces/{workspace_name}", "/root/workspace")
    .env({"WORKFLOWS": test_workflows, "SKIP_TESTS": skip_tests})
)

gpu = modal.gpu.A100()

downloads_vol = modal.Volume.from_name(
    "comfy-downloads", 
    create_if_missing=True
)

app = modal.App(
    name=app_name, 
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ]
)

@app.cls(
    image=image,
    gpu=gpu,
    cpu=8.0,
    volumes={"/data": downloads_vol},
    concurrency_limit=3,
    container_idle_timeout=60,
    keep_warm=0,
    timeout=3600,
)
class ComfyUI:
    
    def _start(self, port=8188):
        print("Start server")
        t1 = time.time()
        self.server_address = f"127.0.0.1:{port}"
        cmd = f"python main.py --dont-print-server --listen --port {port}"
        subprocess.Popen(cmd, shell=True)
        while not self._is_server_running():
            time.sleep(1)
        t2 = time.time()
        self.launch_time = t2 - t1

    def _execute(self, workflow_name: str, args: dict, user: str = None, requester: str = None):
        try:
            print("\n----------->  Starting new task execution: ", workflow_name)
            eden_utils.log_memory_info()
            tool_path = f"/root/workspace/workflows/{workflow_name}"
            tool = Tool.from_yaml(f"{tool_path}/api.yaml")
            workflow = json.load(open(f"{tool_path}/workflow_api.json", 'r'))
            self._validate_comfyui_args(workflow, tool)
            workflow = self._inject_args_into_workflow(workflow, tool, args)
            prompt_id = self._queue_prompt(workflow)['prompt_id']
            outputs = self._get_outputs(prompt_id)
            output = outputs[str(tool.comfyui_output_node_id)]
            if not output:
                raise Exception(f"No output found for {workflow_name} at output node {tool.comfyui_output_node_id}") 
            print("---- comfyui output ----")
            result = {"output": output}
            if tool.comfyui_intermediate_outputs:
                result["intermediate_outputs"] = {
                    key: outputs[str(node_id)]
                    for key, node_id in tool.comfyui_intermediate_outputs.items()
                } 
            print(result)
            return result
        except modal.exception.InputCancellation:
            print("Modal Task Cancelled")
            print("Interrupting ComfyUI")
            self._interrupt()
            print("ComfyUI interrupted")
        except Exception as error:
            print("ComfyUI pipeline error: ", error)
            raise

    @modal.method()
    def run(self, tool_key: str, args: dict):
        result = self._execute(tool_key, args)
        return eden_utils.upload_result(result)

    @modal.method()
    @task_handler_method
    async def run_task(self, tool_key: str, args: dict, user: str = None, requester: str = None):
        return self._execute(tool_key, args, user, requester)
        
    @modal.enter()
    def enter(self):
        self._start()

    @modal.build()
    def downloads(self):
        download_files()
            
    @modal.build()
    def test_workflows(self):
        if os.getenv("SKIP_TESTS"):
            print("Skipping tests")
            return
            
        print(" ==== TESTING WORKFLOWS ====")
        
        t1 = time.time()
        self._start()
        t2 = time.time()
        
        results = {"_performance": {"launch": t2 - t1}}
        workflows_dir = pathlib.Path("/root/workspace/workflows")
        workflow_names = [f.name for f in workflows_dir.iterdir() if f.is_dir()]
        test_workflows = os.getenv("WORKFLOWS")
        if test_workflows:
            test_workflows = test_workflows.split(",")
            if not all([w in workflow_names for w in test_workflows]):
                raise Exception(f"One or more invalid workflows found: {', '.join(test_workflows)}")
            workflow_names = test_workflows
            print(f"====> Running tests for subset of workflows: {' | '.join(workflow_names)}")
        else:
            print(f"====> Running tests for all workflows: {' | '.join(workflow_names)}")

        if not workflow_names:
            raise Exception("No workflows found!")

        for workflow in workflow_names:
            test_all = os.getenv("TEST_ALL", False)
            if test_all:
                tests = glob.glob(f"/root/workspace/workflows/{workflow}/test*.json")
            elif specific_test:
                tests = [f"/root/workspace/workflows/{workflow}/{specific_test}"]
            else:
                tests = [f"/root/workspace/workflows/{workflow}/test.json"]
            print(f"====> Running tests for {workflow}: ", tests)
            for i, test in enumerate(tests):
                print(f"\n\n\n------------------ Running test {i+1} of {len(tests)} ------------------")
                tool = Tool.from_yaml(f"/root/workspace/workflows/{workflow}/api.yaml")
                if tool.status == "inactive":
                    print(f"{workflow} is inactive, skipping test")
                    continue
                test_args = json.loads(open(test, "r").read())
                test_args = tool.prepare_args(test_args)
                test_name = f"{workflow}_{os.path.basename(test)}"
                print(f"====> Running test: {test_name}")
                t1 = time.time()
                result = self._execute(workflow, test_args)
                result = eden_utils.upload_result(result)
                result = eden_utils.prepare_result(result)
                print(f"====> Final media url: {result}")
                t2 = time.time()       
                results[test_name] = result
                results["_performance"][test_name] = t2 - t1

        with open("_test_results_.json", "w") as f:
            json.dump(results, f, indent=4)

    @modal.method()
    def print_test_results(self):
        with open("_test_results_.json", "r") as f:
            results = json.load(f)
        print("\n\n\n============ Test Results ============")
        print(json.dumps(results, indent=4))

    def _is_server_running(self):
        try:
            url = f"http://{self.server_address}/history/123"
            with urllib.request.urlopen(url) as response:
                return response.status == 200
        except URLError:
            return False

    def _queue_prompt(self, prompt):
        data = json.dumps({"prompt": prompt}).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def _interrupt(self):
        try:
            print("Interrupting ComfyUI ...")
            with urllib.request.urlopen(f"http://{self.server_address}/interrupt") as response:
                if response.status != 200:
                    raise Exception(f"Failed to interrupt ComfyUI: {response.status}")
        except Exception as e:
            print(f"Error interrupting ComfyUI: {e}")
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
                    print(f"Warning: Unexpected status code {response.status}")
                
                response_data = response.read()
                try:
                    history_data = json.loads(response_data)
                    
                    if not history_data or (prompt_id not in history_data):
                        return {}
                    
                    return history_data
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON response from {url}")
                    print(f"Error decoding JSON response: {e}")
                    print(f"Raw response data: {response_data[:200]}...")  # Print first 200 chars
                    raise
                    
        except urllib.error.URLError as e:
            print(f"Connection error while fetching history: {e}")
            if hasattr(e, 'reason'):
                print(f"Failure reason: {e.reason}")
            raise
            
        except Exception as e:
            print(f"Unexpected error in _get_history: {str(e)}")
            print(f"Error type: {type(e).__name__}")
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
                    for k, v in messages if k == "execution_error"
                ]
                error_str = ", ".join(errors)
                print("error", error_str)
                raise Exception(error_str)

            for _ in history['outputs']:
                for node_id in history['outputs']:
                    node_output = history['outputs'][node_id]
                    if 'images' in node_output:
                        outputs[node_id] = [
                            os.path.join("output", image['subfolder'], image['filename'])
                            for image in node_output['images']
                        ]
                    elif 'gifs' in node_output:
                        outputs[node_id] = [
                            os.path.join("output", video['subfolder'], video['filename'])
                            for video in node_output['gifs']
                        ]
                    elif 'audio' in node_output:
                        outputs[node_id] = [
                            os.path.join("output", audio['subfolder'], audio['filename'])
                            for audio in node_output['audio']
                        ]
            
            print("comfy outputs", outputs)
            if not outputs:
                raise Exception("No outputs found")
            
            return outputs

    def _inject_embedding_mentions_sdxl(self, text, embedding_trigger, embeddings_filename, lora_mode, lora_strength):
        # Hardcoded computation of the token_strength for the embedding trigger:
        token_strength = 0.5 + lora_strength / 2

        reference = f'(embedding:{embeddings_filename})'
        
        # Make two deep copies of the input text:
        user_prompt = copy.deepcopy(text)
        lora_prompt = copy.deepcopy(text)

        if lora_mode == "face" or lora_mode == "object" or lora_mode == "concept":
            # Match all variations of the embedding_trigger:
            pattern = r'(<{0}>|<{1}>|{0}|{1})'.format(
                re.escape(embedding_trigger),
                re.escape(embedding_trigger.lower())
            )
            lora_prompt = re.sub(pattern, reference, lora_prompt, flags=re.IGNORECASE)
            lora_prompt = re.sub(r'(<concept>)', reference, lora_prompt, flags=re.IGNORECASE)
            if lora_mode == "face":
                base_word = "person"
            else:
                base_word = "object"

            user_prompt = re.sub(pattern, base_word, user_prompt, flags=re.IGNORECASE)
            user_prompt = re.sub(r'(<concept>)', base_word, user_prompt, flags=re.IGNORECASE)

        if reference not in lora_prompt: # Make sure the concept is always triggered:
            if lora_mode == "style":
                lora_prompt = f"in the style of {reference}, {lora_prompt}"
            else:
                lora_prompt = f"{reference}, {lora_prompt}"

        return user_prompt, lora_prompt
    
    def _inject_embedding_mentions_flux(self, text, embedding_trigger, lora_trigger_text):
        if not embedding_trigger:  # Handles both None and empty string
            if lora_trigger_text:
                text = re.sub(r'(<concept>)', lora_trigger_text, text, flags=re.IGNORECASE)
        else:
            pattern = r'(<{0}>|<{1}>|{0}|{1})'.format(
                re.escape(embedding_trigger),
                re.escape(embedding_trigger.lower())
            )
            text = re.sub(pattern, lora_trigger_text, text, flags=re.IGNORECASE)
            text = re.sub(r'(<concept>)', lora_trigger_text, text, flags=re.IGNORECASE)

        if lora_trigger_text:
            if lora_trigger_text not in text:
                text = f"{lora_trigger_text}, {text}"

        return text

    def _transport_lora_flux(self, lora_url: str):
        loras_folder = "/root/models/loras"

        print("tl download lora", lora_url)
        if not re.match(r'^https?://', lora_url):
            raise ValueError(f"Lora URL Invalid: {lora_url}")
        
        lora_filename = lora_url.split("/")[-1]    
        lora_path = os.path.join(loras_folder, lora_filename)
        print("tl destination folder", loras_folder)

        if os.path.exists(lora_path):
            print("Lora safetensors file already extracted. Skipping.")
        else:
            eden_utils.download_file(lora_url, lora_path)
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"The LoRA tar file {lora_path} does not exist.")
        
        print("destination path", lora_path)
        print("lora filename", lora_filename)

        return lora_filename

    def _transport_lora_sdxl(self, lora_url: str):
        downloads_folder = "/root/downloads"
        loras_folder = "/root/models/loras"
        embeddings_folder = "/root/models/embeddings"

        print("tl download lora", lora_url)
        if not re.match(r'^https?://', lora_url):
            raise ValueError(f"Lora URL Invalid: {lora_url}")
        
        lora_filename = lora_url.split("/")[-1]    
        name = lora_filename.split(".")[0]
        destination_folder = os.path.join(downloads_folder, name)
        print("tl destination folder", destination_folder)

        if os.path.exists(destination_folder):
            print("Lora bundle already extracted. Skipping.")
        else:
            try:
                lora_tarfile = eden_utils.download_file(lora_url, f"/root/downloads/{lora_filename}")
                if not os.path.exists(lora_tarfile):
                    raise FileNotFoundError(f"The LoRA tar file {lora_tarfile} does not exist.")
                with tarfile.open(lora_tarfile, "r:*") as tar:
                    tar.extractall(path=destination_folder)
                    print("Extraction complete.")
            except Exception as e:
                raise IOError(f"Failed to extract tar file: {e}")

        extracted_files = os.listdir(destination_folder)
        print("tl, extracted files", extracted_files)

        # Find lora and embeddings files using regex
        lora_pattern = re.compile(r'.*_lora\.safetensors$')
        embeddings_pattern = re.compile(r'.*_embeddings\.safetensors$')

        lora_filename = next((f for f in extracted_files if lora_pattern.match(f)), None)
        embeddings_filename = next((f for f in extracted_files if embeddings_pattern.match(f)), None)
        training_args_filename = next((f for f in extracted_files if f == "training_args.json"), None)

        if training_args_filename:
            with open(os.path.join(destination_folder, training_args_filename), "r") as f:
                training_args = json.load(f)
                lora_mode = training_args["concept_mode"]
                embedding_trigger = training_args["name"]
        else:
            lora_mode = None
            embedding_trigger = embeddings_filename.split('_embeddings.safetensors')[0]

        # hack to correct for older lora naming convention
        if not lora_filename:
            print("Lora file not found with standard naming convention. Searching for alternative...")
            lora_filename = next((f for f in extracted_files if f.endswith('.safetensors') and 'embedding' not in f.lower()), None)
            if not lora_filename:
                raise FileNotFoundError(f"Unable to find a lora *.safetensors file in {extracted_files}")
            
        print("tl, lora mode:", lora_mode)
        print("tl, lora filename:", lora_filename)
        print("tl, embeddings filename:", embeddings_filename)
        print("tl, embedding_trigger:", embedding_trigger)

        for file in [lora_filename, embeddings_filename]:
            if str(file) not in extracted_files:
                raise FileNotFoundError(f"Required file {file} does not exist in the extracted files: {extracted_files}")

        if not os.path.exists(loras_folder):
            os.makedirs(loras_folder)
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

        # copy lora file to loras folder
        lora_path = os.path.join(destination_folder, lora_filename)
        lora_copy_path = os.path.join(loras_folder, lora_filename)
        shutil.copy(lora_path, lora_copy_path)
        print(f"LoRA {lora_path} has been moved to {lora_copy_path}")

        # copy embedding file to embeddings folder
        embeddings_path = os.path.join(destination_folder, embeddings_filename)
        embeddings_copy_path = os.path.join(embeddings_folder, embeddings_filename)
        shutil.copy(embeddings_path, embeddings_copy_path)
        print(f"Embeddings {embeddings_path} has been moved to {embeddings_copy_path}")
        
        return lora_filename, embeddings_filename, embedding_trigger, lora_mode

    def _url_to_filename(self, url):
        filename = url.split('/')[-1]
        filename = re.sub(r'\?.*$', '', filename)
        max_length = 255
        if len(filename) > max_length: # ensure filename is not too long
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext
        return filename    

    def _validate_comfyui_args(self, workflow, tool):
        for key, comfy_param in tool.comfyui_map.items():
            node_id, field, subfield, remaps = str(comfy_param.node_id), str(comfy_param.field), str(comfy_param.subfield), comfy_param.remap
            subfields = [s.strip() for s in subfield.split(",")]
            for subfield in subfields:
                if node_id not in workflow or field not in workflow[node_id] or subfield not in workflow[node_id][field]:
                    raise Exception(f"Node ID {node_id}, field {field}, subfield {subfield} not found in workflow")
            for remap in remaps or []:
                subfields = [s.strip() for s in str(remap.subfield).split(",")]
                for subfield in subfields:
                    if str(remap.node_id) not in workflow or str(remap.field) not in workflow[str(remap.node_id)] or subfield not in workflow[str(remap.node_id)][str(remap.field)]:
                        raise Exception(f"Node ID {remap.node_id}, field {remap.field}, subfield {subfield} not found in workflow")
                param = tool.model.model_fields[key]
                # has_choices = isinstance(param.annotation, type) and issubclass(param.annotation, Enum)
                # if not has_choices:
                #     raise Exception(f"Remap parameter {key} has no original choices")
                # choices = [e.value for e in param.annotation]
                choices = param.json_schema_extra.get("choices")
                if not all(choice in choices for choice in remap.map.keys()):
                    raise Exception(f"Remap parameter {key} has invalid choices: {remap.map}")
                if not all(choice in remap.map.keys() for choice in choices):
                    raise Exception(f"Remap parameter {key} is missing original choices: {choices}")
                                
    def _inject_args_into_workflow(self, workflow, tool, args):
        base_model = "unknown"
        # Helper function to validate and normalize URLs
        def validate_url(url):
            if not isinstance(url, str):
                raise ValueError(f"Invalid URL type: {type(url)}. Expected string.")
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            return url

        print("===== Injecting comfyui args into workflow =====")
        pprint(args)

        embedding_triggers = {"lora": None, "lora2": None}
        lora_trigger_texts = {"lora": None, "lora2": None}

        # download and transport files        
        for key, param in tool.model.model_fields.items():
            metadata = param.json_schema_extra or {}
            file_type = metadata.get('file_type')
            is_array = metadata.get('is_array')
            print(f"Parsing {key}, param: {param}")

            if file_type and any(t in ["image", "video", "audio"] for t in file_type.split("|")):
                if not args.get(key):
                    continue
                if is_array:
                    urls = [validate_url(url) for url in args.get(key)]
                    args[key] = [
                        eden_utils.download_file(url, f"/root/input/{self._url_to_filename(url)}") if url else None 
                        for url in urls
                    ] if urls else None
                else:
                    url = validate_url(args.get(key))
                    args[key] = eden_utils.download_file(url, f"/root/input/{self._url_to_filename(url)}") if url else None
            
            elif file_type == "lora":
                lora_id = args.get(key)
                
                if not lora_id:
                    args[key] = None
                    args[f"{key}_strength"] = 0
                    print(f"DISABLING {key}")
                    continue
                
                print(f"Found {key} LORA ID: ", lora_id)
                
                models = get_collection("models3")
                lora = models.find_one({"_id": ObjectId(lora_id)})
                #print("found lora:\n", lora)

                if not lora:
                    raise Exception(f"Lora {key} with id: {lora_id} not found in DB {db}!")

                base_model = lora.get("base_model")
                lora_url = lora.get("checkpoint")
                #lora_name = lora.get("name")
                #pretrained_model = lora.get("args").get("sd_model_version")

                if not lora_url:
                    raise Exception(f"Lora {lora_id} has no checkpoint")
                else:
                    print("LORA URL", lora_url)

                lora_url = get_full_url(lora_url)
                print("lora url", lora_url)
                print("base model", base_model)

                if base_model == "sdxl":
                    lora_filename, embeddings_filename, embedding_trigger, lora_mode = self._transport_lora_sdxl(lora_url)
                elif base_model == "flux-dev":
                    lora_filename = self._transport_lora_flux(lora_url)
                    embedding_triggers[key] = lora.get("args", {}).get("name")
                    try:
                        lora_trigger_texts[key] = lora.get("lora_trigger_text")
                    except: # old flux LoRA's:
                        lora_trigger_texts[key] = lora.get("args", {}).get("caption_prefix")

                args[key] = lora_filename

        for key, comfyui in tool.comfyui_map.items():
            
            value = args.get(key)
            if value is None:
                continue

            if key == "no_token_prompt":
                continue

            # if there's a lora, replace mentions with embedding name
            if key == "prompt":
                if "flux" in base_model:
                    for lora_key in ["lora", "lora2"]:
                        if args.get(f"use_{lora_key}", False):
                            lora_strength = args.get(f"{lora_key}_strength", 0.7)
                            value = self._inject_embedding_mentions_flux(
                                value,
                                embedding_triggers[lora_key],
                                lora_trigger_texts[lora_key]
                            )
                            print(f"====> INJECTED {lora_key} TRIGGER TEXT", value)
                elif base_model == "sdxl":  
                    if embedding_trigger:
                        lora_strength = args.get("lora_strength", 0.7)
                        no_token_prompt, value = self._inject_embedding_mentions_sdxl(value, embedding_trigger, embeddings_filename, lora_mode, lora_strength)
                        
                        if "no_token_prompt" in args:
                            no_token_mapping = next((comfy_param for key, comfy_param in tool.comfyui_map.items() if key == "no_token_prompt"), None)
                            if no_token_mapping:
                                print("Updating no_token_prompt for SDXL: ", no_token_prompt)
                                workflow[str(no_token_mapping.node_id)][no_token_mapping.field][no_token_mapping.subfield] = no_token_prompt

                print("====> Final updated prompt for workflow: ", value)

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

            print("comfyui mapping")
            print(comfyui)

            node_id, field, subfield = str(comfyui.node_id), str(comfyui.field), str(comfyui.subfield)
            subfields = [s.strip() for s in subfield.split(",")]
            for subfield in subfields:
                print("inject", node_id, field, subfield, " = ", value)
                workflow[node_id][field][subfield] = value  

            for remap in comfyui.remap or []:
                subfields = [s.strip() for s in str(remap.subfield).split(",")]
                for subfield in subfields:
                    output_value = remap.map.get(value)
                    print("remap", str(remap.node_id), remap.field, subfield, " = ", output_value)
                    workflow[str(remap.node_id)][remap.field][subfield] = output_value

        return workflow


@app.local_entrypoint()
def run():
    comfyui = ComfyUI()
    comfyui.print_test_results.remote()
