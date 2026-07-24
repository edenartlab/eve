from bson import ObjectId

from eve.s3 import get_full_url
from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

T2I_ENDPOINT = "fal-ai/flux-lora"
I2I_ENDPOINT = "fal-ai/flux-lora/image-to-image"


def _lora_url(lora_id):
    """Resolve a models3 LoRA id to its public CloudFront URL."""
    from eve.models import Model

    model = Model.from_mongo(ObjectId(str(lora_id)))
    if not model or not model.checkpoint:
        raise ValueError(f"LoRA {lora_id} not found or has no checkpoint")
    return get_full_url(model.checkpoint)


async def handler(context: ToolContext):
    args = context.args

    loras = []
    if args.get("lora"):
        loras.append({"path": _lora_url(args["lora"]),
                      "scale": float(args.get("lora_strength") or 0.8)})
    if args.get("lora2"):
        loras.append({"path": _lora_url(args["lora2"]),
                      "scale": float(args.get("lora2_strength") or 0.8)})

    payload = {
        "prompt": args["prompt"],
        "num_images": int(args.get("n_samples") or 1),
        "image_size": {
            "width": int(args.get("width") or 1024),
            "height": int(args.get("height") or 1024),
        },
        "guidance_scale": float(args.get("flux_guidance") or 3.5),
        "num_inference_steps": int(args.get("steps") or 28),
        "enable_safety_checker": True,
    }
    if loras:
        payload["loras"] = loras
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    if args.get("init_image"):
        payload["image_url"] = args["init_image"]
        payload["strength"] = float(args.get("denoise") or 0.75)
        endpoint = I2I_ENDPOINT
    else:
        endpoint = T2I_ENDPOINT

    result = await call_fal_with_retry(endpoint, payload)
    images = (result or {}).get("images") or []
    urls = [im.get("url") for im in images if im.get("url")]
    if not urls:
        raise ValueError(f"FLUX (fal) returned no images: {result}")
    return {"output": urls}
