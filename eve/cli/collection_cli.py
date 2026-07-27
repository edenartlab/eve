import json
import os
from pathlib import Path

import click
import requests
from bson import ObjectId
from bson.errors import InvalidId
from loguru import logger

from .. import load_env
from ..mongo import get_collection
from ..task import Creation, CreationsCollection, Task
from ..utils import data_utils


@click.group()
def collection():
    """Collection management commands"""
    pass


@collection.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to download from",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./collection_<id>)",
)
@click.argument("collection_id", required=True)
def download(db: str, output: str, collection_id: str):
    """Download all creations from a collection.

    COLLECTION_ID can be a MongoDB ObjectId or a URL like
    https://app.eden.art/collections/69b75676540b89db4429fe45
    """

    load_env(db)

    # Extract ID from URL if needed
    if "/" in collection_id:
        collection_id = collection_id.rstrip("/").split("/")[-1]

    try:
        obj_id = ObjectId(collection_id)
    except InvalidId:
        click.echo(click.style(f"Invalid collection ID: {collection_id}", fg="red"))
        return

    # Fetch the collection
    collections_col = get_collection("collections3")
    col_doc = collections_col.find_one({"_id": obj_id})

    if not col_doc:
        click.echo(click.style(f"Collection not found: {collection_id}", fg="red"))
        return

    col_name = col_doc.get("name", "untitled")
    creation_ids = col_doc.get("creations", [])

    click.echo(
        click.style(f'Collection: "{col_name}" ({len(creation_ids)} creations)', fg="cyan", bold=True)
    )

    if not creation_ids:
        click.echo(click.style("Collection is empty.", fg="yellow"))
        return

    # Set up output directory
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in col_name).strip().replace(" ", "_")
    if output:
        export_dir = Path(output)
    else:
        export_dir = Path(f"collection_{safe_name}_{collection_id[:8]}")

    export_dir.mkdir(parents=True, exist_ok=True)

    # Save collection metadata
    col_meta = {
        "collection_id": str(obj_id),
        "name": col_name,
        "description": col_doc.get("description"),
        "user": str(col_doc.get("user")),
        "public": col_doc.get("public", True),
        "createdAt": str(col_doc.get("createdAt")),
        "updatedAt": str(col_doc.get("updatedAt")),
        "num_creations": len(creation_ids),
    }
    meta_path = export_dir / "collection.json"
    with open(meta_path, "w") as f:
        json.dump(col_meta, f, indent=2, default=str)

    # Download each creation
    creations_col = get_collection("creations3")
    downloaded = 0
    skipped = 0
    failed = 0

    for i, creation_id in enumerate(creation_ids):
        id_str = str(creation_id)
        prefix = f"[{i + 1}/{len(creation_ids)}]"

        # Check if already downloaded (any file matching this id)
        existing = list(export_dir.glob(f"*_{id_str}.*"))
        if existing:
            skipped += 1
            click.echo(click.style(f"  {prefix} Skipping {id_str} (already exists)", fg="yellow"))
            continue

        creation_doc = creations_col.find_one({"_id": creation_id})
        if not creation_doc:
            failed += 1
            click.echo(click.style(f"  {prefix} Creation not found: {id_str}", fg="red"))
            continue

        filename = creation_doc.get("filename")
        if not filename:
            failed += 1
            click.echo(click.style(f"  {prefix} No filename for {id_str}", fg="red"))
            continue

        # Get the download URL
        prepared = data_utils.prepare_result(dict(creation_doc))
        url = prepared.get("url")
        if not url:
            failed += 1
            click.echo(click.style(f"  {prefix} No URL for {id_str}", fg="red"))
            continue

        # Determine file extension
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
        created_at = creation_doc.get("createdAt")
        date_prefix = created_at.strftime("%Y-%m-%d_%H%M") if created_at else "unknown"
        asset_filename = export_dir / f"{date_prefix}_{id_str}.{ext}"
        json_filename = export_dir / f"{date_prefix}_{id_str}.json"

        # Download the media file
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(asset_filename, "wb") as f:
                f.write(response.content)
        except Exception as e:
            failed += 1
            click.echo(click.style(f"  {prefix} Download failed for {id_str}: {e}", fg="red"))
            continue

        # Save creation metadata JSON
        creation_meta = dict(creation_doc)
        # Enrich with task data if available
        try:
            task = Task.from_mongo(creation_doc["task"])
            if task:
                creation_meta["task"] = task.model_dump()
        except Exception:
            pass

        with open(json_filename, "w") as f:
            json.dump(creation_meta, f, indent=2, default=str)

        downloaded += 1
        size_kb = len(response.content) / 1024
        click.echo(
            click.style(f"  {prefix} ", fg="white")
            + click.style(f"{asset_filename.name}", fg="green")
            + click.style(f" ({size_kb:.0f} KB)", fg="white")
        )

    click.echo()
    click.echo(click.style(f"Done! Output: {export_dir}", fg="cyan", bold=True))
    click.echo(
        click.style(f"  Downloaded: {downloaded}", fg="green")
        + click.style(f"  Skipped: {skipped}", fg="yellow")
        + click.style(f"  Failed: {failed}", fg="red")
    )
