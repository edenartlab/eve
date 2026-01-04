import asyncio
import base64
import json
import uuid

import requests

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    agent_obj = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="shopify")
    if not deployment:
        raise Exception("No valid shopify deployments found")

    store_name = deployment.secrets.shopify.store_name
    access_token = deployment.secrets.shopify.access_token
    location_id = deployment.secrets.shopify.location_id
    api_version = "2025-07"

    if not all([store_name, access_token, location_id]):
        raise ValueError("Missing required Shopify credentials")

    ENDPOINT = (
        f"https://{store_name}.myshopify.com/admin/api/{api_version}/graphql.json"
    )
    HEADERS = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }

    # ────────────────────────────────────────────────────────────────
    # GraphQL Helper
    # ────────────────────────────────────────────────────────────────
    async def _gql(query: str, variables: dict | None = None):
        r = await asyncio.to_thread(
            requests.post,
            ENDPOINT,
            headers=HEADERS,
            data=json.dumps({"query": query, "variables": variables or {}}),
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        return data["data"]

    # ────────────────────────────────────────────────────────────────
    # Mutations / queries
    # ────────────────────────────────────────────────────────────────
    PRODUCT_CREATE = """
    mutation productCreate($product: ProductCreateInput!, $media: [CreateMediaInput!]) {
        productCreate(product: $product, media: $media) {
            product {
                id title status
                handle
                variants(first: 1) { nodes { id inventoryItem { id sku } } }
            }
            userErrors { field message }
        }
    }
    """

    VARIANTS_UPDATE = """
    mutation productVariantsBulkUpdate($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
        productVariantsBulkUpdate(productId: $productId, variants: $variants) {
            productVariants { id price inventoryItem { id sku } }
            userErrors      { field message }
        }
    }
    """

    SET_QTY = """
    mutation inventorySetQuantities($input: InventorySetQuantitiesInput!) {
        inventorySetQuantities(input: $input) {
            inventoryAdjustmentGroup { createdAt }
            userErrors { field message }
        }
    }
    """

    PUBS = "query { publications(first: 5) { nodes { id name } } }"  # Online Store is usually first

    PUBLISH = """
    mutation publish($id: ID!, $pub: ID!) {
        publishablePublish(
            id:   $id,                     # product GID
            input:{ publicationId: $pub }  # Online‑Store publication GID
        ){
            userErrors { field message }
        }
    }
    """

    # ---- 1. create skeleton product + hero image ----
    draft = await _gql(
        PRODUCT_CREATE,
        {
            "product": {
                "title": context.args.get("title"),
                "descriptionHtml": context.args.get("description"),
                "status": "ACTIVE",
            },
            "media": [
                {
                    "alt": context.args.get("alt_text"),
                    "mediaContentType": "IMAGE",
                    "originalSource": context.args.get("image"),
                }
            ],
        },
    )["productCreate"]

    if draft["userErrors"]:
        raise RuntimeError(draft["userErrors"])

    product_id = draft["product"]["id"]
    variant_id = draft["product"]["variants"]["nodes"][0]["id"]
    item_id = draft["product"]["variants"]["nodes"][0]["inventoryItem"]["id"]
    handle = draft["product"]["handle"]
    store_url = f"https://{store_name}.myshopify.com/products/{handle}"

    # make a sku from the slug
    base = "".join(e for e in context.args.get("title") if e.isalnum() or e == " ")
    base = base.upper().replace(" ", "-")[:10]
    sku = base64.b32encode(uuid.uuid4().bytes)[:4].decode("ascii")
    sku = f"{base}-{sku}"

    # ---- 2. set price + SKU on the default variant ----
    await _gql(
        VARIANTS_UPDATE,
        {
            "productId": product_id,
            "variants": [
                {
                    "id": variant_id,
                    "price": context.args.get("price"),
                    "inventoryItem": {"sku": sku},
                }
            ],
        },
    )

    # ---- 3. set on‑hand inventory at first active location ----
    location_id = f"gid://shopify/Location/{location_id}"
    await _gql(
        SET_QTY,
        {
            "input": {
                "name": "available",  # or "on_hand"
                "reason": "initial_stock",  # any non‑empty string
                "quantities": [
                    {
                        "inventoryItemId": item_id,
                        "locationId": location_id,
                        "quantity": context.args.get("quantity"),
                        # "compareQuantity": 0  # optional: CAS check
                    }
                ],
            }
        },
    )

    # ---- 4. publish to Online Store, so no manual merch action is needed ----
    if context.args.get("auto_publish"):
        online_pub = (await _gql(PUBS))["publications"]["nodes"][0]["id"]
        await _gql(PUBLISH, {"id": product_id, "pub": online_pub})

    return {"output": [{"url": store_url}]}
