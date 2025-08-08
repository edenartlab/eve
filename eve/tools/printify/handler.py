import requests
import uuid
from typing import Dict, Any

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")

    agent_obj = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="printify")
    if not deployment:
        raise Exception("No valid Printify deployments found")

    api_token = deployment.secrets.printify.api_token
    shop_id = deployment.secrets.printify.shop_id

    if not all([api_token, shop_id]):
        raise ValueError("Missing required Printify credentials")

    BASE_URL = "https://api.printify.com/v1"
    HEADERS = {
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "EdenPrintify/1.0.0",
        "Content-Type": "application/json",
    }

    def _api_request(method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make a request to the Printify API"""
        url = f"{BASE_URL}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=HEADERS, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=HEADERS, json=data, timeout=30)
            elif method == "PUT":
                response = requests.put(url, headers=HEADERS, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                error_data = (
                    e.response.json()
                    if e.response.headers.get("content-type") == "application/json"
                    else {}
                )
                raise RuntimeError(f"Printify API error: {error_data}")
            raise

    # Get blueprint details to find available variants
    blueprint_id = args.get("blueprint_id")
    print_provider_id = args.get("print_provider_id")

    # Get variants for the blueprint and provider
    variants_response = _api_request(
        "GET",
        f"/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json",
    )

    # Select default variants (we'll enable the first few common sizes)
    # For t-shirts, typically these are S, M, L, XL, 2XL
    enabled_variants = []
    variant_ids = []

    for i, variant in enumerate(variants_response[:5]):  # Enable first 5 variants
        variant_ids.append(variant["id"])
        enabled_variants.append(
            {
                "id": variant["id"],
                "price": int(args.get("price") * 100),  # Convert to cents
                "is_enabled": True,
                "is_default": i == 0,  # Make first variant default
            }
        )

    # Create print areas configuration
    print_areas = [
        {
            "variant_ids": variant_ids,
            "placeholders": [
                {
                    "position": "front",
                    "images": [
                        {
                            "id": str(uuid.uuid4()),
                            "x": 0.5,
                            "y": 0.5,
                            "scale": 1,
                            "angle": 0,
                            "src": args.get("image"),
                        }
                    ],
                }
            ],
        }
    ]

    # Create the product
    product_data = {
        "title": args.get("title"),
        "description": args.get("description"),
        "blueprint_id": blueprint_id,
        "print_provider_id": print_provider_id,
        "variants": enabled_variants,
        "print_areas": print_areas,
    }

    created_product = _api_request(
        "POST", f"/shops/{shop_id}/products.json", product_data
    )
    product_id = created_product["id"]

    # Auto-publish if requested
    if args.get("auto_publish"):
        # Note: Publishing requires a connected sales channel (e.g., Shopify, Etsy)
        # This would need to be configured in Printify account
        try:
            # Get available sales channels
            shops_response = _api_request("GET", "/shops.json")

            # Find the shop and its sales channels
            current_shop = next(
                (shop for shop in shops_response if shop["id"] == int(shop_id)), None
            )

            if current_shop and current_shop.get("sales_channels"):
                # Publish to first available sales channel
                sales_channel = current_shop["sales_channels"][0]
                publish_data = {"sales_channel_id": sales_channel["id"]}
                _api_request(
                    "POST",
                    f"/shops/{shop_id}/products/{product_id}/publish.json",
                    publish_data,
                )
        except Exception as e:
            print(
                f"Auto-publish failed: {e}. Product created but remains in draft state."
            )

    # Generate product URL
    product_url = f"https://app.printify.com/app/products/{product_id}"

    return {
        "output": [
            {
                "url": product_url,
                "product_id": product_id,
                "status": "published" if args.get("auto_publish") else "draft",
            }
        ]
    }
