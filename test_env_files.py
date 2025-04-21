import os

home_dir = os.path.expanduser("~")  # or set this manually if needed

for db in ["STAGE", "PROD"]:
    os.environ["DB"] = db

    stage = db == "STAGE"
    env_file = ".env.STAGE" if stage else ".env"
    eve_file = ".eve.STAGE" if stage else ".eve"
    eve_path = os.path.join(home_dir, eve_file)
    env_path = os.path.join(home_dir, env_file)

    print(f"\nChecking files for DB={db}:")

    if os.path.isfile(env_path):
        print(f"Found env file: {env_path}")
    else:
        print(f"Missing env file: {env_path}")

    if os.path.isfile(eve_path):
        print(f"Found eve file: {eve_path}")
    else:
        print(f"Missing eve file: {eve_path}")
