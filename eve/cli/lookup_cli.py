import click
from bson import ObjectId
from bson.errors import InvalidId

from ..task import Creation
from ..user import User
from ..utils import dumps_json


def print_document(doc_id, doc_dict, doc_type):
    """Print a document in a nice, readable format"""
    click.echo(click.style(f"\n{'=' * 60}", fg="cyan"))
    click.echo(click.style(f"{doc_type.upper()} FOUND", fg="cyan", bold=True))
    click.echo(click.style(f"{'=' * 60}\n", fg="cyan"))

    # Print ID at the top
    click.echo(click.style(f"ID: {doc_id}", fg="yellow", bold=True))
    click.echo()

    # Print the rest of the document
    # Remove _id from the dict since we already printed it
    doc_dict_copy = {k: v for k, v in doc_dict.items() if k != "_id"}

    # Convert ObjectIds to strings for better display
    for key, value in doc_dict_copy.items():
        if isinstance(value, ObjectId):
            doc_dict_copy[key] = str(value)
        elif isinstance(value, list):
            doc_dict_copy[key] = [
                str(v) if isinstance(v, ObjectId) else v for v in value
            ]

    formatted_doc = dumps_json(doc_dict_copy, indent=2)
    click.echo(click.style(formatted_doc, fg="white"))
    click.echo(click.style(f"\n{'=' * 60}\n", fg="cyan"))


@click.command()
@click.option(
    "--db",
    type=click.Choice(["STAGE", "PROD"], case_sensitive=False),
    default="STAGE",
    help="DB to search in",
)
@click.argument("name", required=True)
def lookup(db: str, name: str):
    """Lookup a user by username or a creation by ID"""

    # First, try to find by username in users3
    try:
        user = User.load(username=name)
        if user:
            # Get the raw document from mongo to display all fields
            users_collection = User.get_collection()
            user_doc = users_collection.find_one({"username": name})
            print_document(user_doc["_id"], user_doc, "user")
            return
    except Exception:
        # User not found, continue to next search
        pass

    # Second, try to find by ObjectId in creations3
    try:
        # Validate if it's a valid ObjectId format
        creation_id = ObjectId(name)
        creations_collection = Creation.get_collection()
        creation_doc = creations_collection.find_one({"_id": creation_id})

        if creation_doc:
            print_document(creation_doc["_id"], creation_doc, "creation")
            return
    except (InvalidId, Exception):
        # Not a valid ObjectId or creation not found
        pass

    # Nothing found
    click.echo(click.style(f"\nNo document found for: {name}", fg="red", bold=True))
    click.echo(click.style("Searched in:", fg="yellow"))
    click.echo(click.style("  - users3 collection (by username)", fg="yellow"))
    click.echo(click.style("  - creations3 collection (by ObjectId)\n", fg="yellow"))
