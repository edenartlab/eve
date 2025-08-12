import requests
import uuid
import random
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

    # Randomly select 3 variants
    enabled_variants = []
    variant_ids = []

    # Get the variants array from the response
    variants = variants_response.get("variants", [])

    # Pick 3 random variants from the available ones
    num_variants_to_select = min(3, len(variants))
    selected_variants = random.sample(variants, num_variants_to_select)

    for i, variant in enumerate(selected_variants):
        variant_ids.append(variant["id"])
        enabled_variants.append(
            {
                "id": variant["id"],
                "price": int(args.get("price") * 100),  # Convert to cents
                "is_enabled": True,
                "is_default": i == 0,  # Make first variant default
            }
        )

    # Upload image to Printify first
    image_url = args.get("image")
    upload_data = {"file_name": f"design_{uuid.uuid4().hex}.png", "url": image_url}

    uploaded_image = _api_request("POST", "/uploads/images.json", upload_data)

    # Get the uploaded image ID
    image_id = uploaded_image["id"]

    # Create print areas configuration using the uploaded image ID
    print_areas = [
        {
            "variant_ids": variant_ids,
            "placeholders": [
                {
                    "position": "front",
                    "images": [
                        {
                            "id": image_id,  # Use the Printify image ID
                            "x": 0.5,
                            "y": 0.5,
                            "scale": 1,
                            "angle": 0,
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
        try:
            publish_response = _api_request(
                "POST",
                f"/shops/{shop_id}/products/{product_id}/publish.json",
                {
                    "title": True,
                    "description": True,
                    "images": True,
                    "variants": True,
                    "tags": True,
                },
            )
        except Exception as e2:
            print(
                f"Auto-publish failed: {e2}. Product created but remains in draft state."
            )

    # Fetch the product details to get the generated product images
    product_details = _api_request(
        "GET", f"/shops/{shop_id}/products/{product_id}.json"
    )

    # Extract product images from the response
    product_images = []
    if product_details and "images" in product_details:
        # Get the main product images (mockups)
        for image in product_details["images"]:
            if image.get("src"):
                product_images.append(
                    {
                        "url": image["src"],
                        "position": image.get("position", ""),
                        "is_default": image.get("is_default", False),
                    }
                )

    # If no images in main response, check variants for images
    if not product_images and "variants" in product_details:
        for variant in product_details["variants"][
            :3
        ]:  # Get images from first 3 variants
            if variant.get("preview_url"):
                product_images.append(
                    {
                        "url": variant["preview_url"],
                        "variant_id": variant["id"],
                        "variant_title": variant.get("title", ""),
                    }
                )

    # Generate product URL - use the public product details URL format
    product_url = f"https://printify.com/app/product-details/{product_id}"

    return {
        "output": [
            {
                "url": product_url,
                "product_id": product_id,
                "images": product_images,
                "title": product_details.get("title", args.get("title")),
                "status": "published" if args.get("auto_publish") else "draft",
            }
        ]
    }
