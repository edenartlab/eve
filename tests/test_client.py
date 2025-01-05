import os
import json
import time
import subprocess
import requests


EDEN_ADMIN_KEY = os.getenv("EDEN_ADMIN_KEY")
headers = {
    "Authorization": f"Bearer {EDEN_ADMIN_KEY}",
    "Content-Type": "application/json",
}


def run_create(server_url):
    request = {
        "user_id": "65284b18f8bbb9bff13ebe65",
        "tool": "flux_schnell",
        "args": {
            "prompt": "a picture of a kangaroo roller skating in venice beach",
        }
    }
    response = requests.post(server_url+"/create", json=request, headers=headers)
    print("GO!!!")
    print(response)
    print("Status Code:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    print("done...")


def run_chat(server_url):
    request = {
        "user_id": "65284b18f8bbb9bff13ebe65",
        "agent_id": "675fd3c379e00297cdac16fb",
        "user_message": {
            "content": "verdelis make a picture of yourself on the beach. use flux_dev_lora and make sure to mention 'Verdelis' in the prompt",
        }
    }
    response = requests.post(server_url+"/chat", json=request, headers=headers)
    print("Status Code:", response.status_code)
    print(json.dumps(response.json(), indent=2))


def test_client():
    server_url = None  
    try:
        if not server_url:
            print("Starting server...")
            server = subprocess.Popen(
                ["rye", "run", "eve", "api", "--db", "STAGE"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)
            server_url = "http://localhost:8000"

        print("server_url", server_url)
        
        print("\nRunning create test...")
        run_create(server_url)

        print("\nRunning chat test...")
        # run_chat(server_url)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'server' in locals():
            server.terminate()
            server.wait()


if __name__ == "__main__":
    test_client()