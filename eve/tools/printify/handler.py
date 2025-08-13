import requests
import uuid
import traceback
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

    # Hardcoded blueprint and print provider
    blueprint_id = 77
    print_provider_id = 99

    # Hardcoded variants for t-shirt sizes
    variant_ids = [32918, 32919, 32920, 32921]  # S, M, L, XL
    enabled_variants = [
        {
            "id": 32918,  # Small (S)
            "price": int(args.get("price") * 100),  # Convert to cents
            "is_enabled": True,
            "is_default": True,  # Make S the default
        },
        {
            "id": 32919,  # Medium (M)
            "price": int(args.get("price") * 100),  # Convert to cents
            "is_enabled": True,
            "is_default": False,
        },
        {
            "id": 32920,  # Large (L)
            "price": int(args.get("price") * 100),  # Convert to cents
            "is_enabled": True,
            "is_default": False,
        },
        {
            "id": 32921,  # Extra Large (XL)
            "price": int(args.get("price") * 100),  # Convert to cents
            "is_enabled": True,
            "is_default": False,
        },
    ]

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
    published = False
    if args.get("auto_publish"):
        try:
            _api_request(
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
            published = True
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
        for variant in product_details["variants"]:
            if variant.get("preview_url"):
                product_images.append(
                    {
                        "url": variant["preview_url"],
                        "variant_id": variant["id"],
                        "variant_title": variant.get("title", ""),
                    }
                )

    # Generate product URL
    product_url = f"https://printify.com/app/product-details/{product_id}"

    # If published, generate the Shopify URL from the title slug
    if published:
        try:
            # Get shop details to find the Shopify domain
            shop_details = _api_request("GET", "/shops.json")

            # Find the specific shop from the list
            shop_info = None
            for shop in shop_details:
                if str(shop.get("id")) == str(shop_id):
                    shop_info = shop
                    break

            if not shop_info:
                raise RuntimeError(f"Shop with ID {shop_id} not found")

            if shop_info.get("sales_channel") == "shopify":
                # Try to get the Shopify store name from an existing Shopify deployment
                try:
                    shopify_deployment = Deployment.load(
                        agent=agent_obj.id, platform="shopify"
                    )
                    if shopify_deployment and shopify_deployment.secrets.shopify:
                        shopify_store_name = (
                            shopify_deployment.secrets.shopify.store_name
                        )

                        # Generate slug from product title
                        title = product_details.get("title", args.get("title", ""))
                        # Convert title to slug: lowercase, replace spaces with hyphens, remove special chars
                        slug = "".join(
                            c if c.isalnum() or c == " " else "" for c in title.lower()
                        )
                        slug = "-".join(slug.split())

                        # Construct the Shopify URL
                        product_url = f"https://{shopify_store_name}.myshopify.com/products/{slug}"
                    else:
                        print(
                            "No Shopify deployment found for this agent, using Printify URL"
                        )
                except Exception as e:
                    print(f"Could not get Shopify deployment: {e}")
        except Exception as e:
            # If we can't get the Shopify URL, fall back to Printify URL
            print(f"Could not construct Shopify URL: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")

    return {
        "output": [
            {
                "url": product_url,
                "product_id": product_id,
                "images": product_images,
                "title": product_details.get("title", args.get("title")),
                "status": "published" if published else "draft",
            }
        ]
    }
