from bson import ObjectId

from eve.s3 import get_full_url
from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

T2I_ENDPOINT = "fal-ai/flux-lora"
I2I_ENDPOINT = "fal-ai/flux-lora/image-to-image"


def _load_lora(lora_id):
    """Resolve a models3 LoRA id to (public CloudFront URL, trigger text)."""
    from eve.models import Model

    model = Model.from_mongo(ObjectId(str(lora_id)))
    if not model or not model.checkpoint:
        raise ValueError(f"LoRA {lora_id} not found or has no checkpoint")
    return get_full_url(model.checkpoint), (model.lora_trigger_text or "").strip()


async def handler(context: ToolContext):
    args = context.args

    loras = []
    prompt = args["prompt"]
    for id_key, strength_key in (("lora", "lora_strength"), ("lora2", "lora2_strength")):
        if args.get(id_key):
            url, trigger = _load_lora(args[id_key])
            loras.append({"path": url, "scale": float(args.get(strength_key) or 0.8)})
            # Mirror the comfy pipeline: ensure the trigger token is present so
            # the LoRA actually activates even when the caller forgets it.
            if trigger and trigger.lower() not in prompt.lower():
                prompt = f"{trigger}, {prompt}"

    payload = {
        "prompt": prompt,
        "num_images": int(args.get("n_samples") or 1),
        "image_size": {
            "width": int(args.get("width") or 1024),
            "height": int(args.get("height") or 1024),
        },
        "guidance_scale": float(args.get("flux_guidance") or 3.5),
        "num_inference_steps": int(args.get("steps") or 28),
        # Permissive default, matching Eden's self-hosted Modal FLUX (which has no
        # safety filter): a lot of LoRAs are of real people. Overridable per call.
        # NOTE: fal-ai/flux-lora exposes ONLY enable_safety_checker — it has no
        # safety_tolerance param (verified against the live schema); sending one
        # would risk rejection.
        "enable_safety_checker": bool(args.get("enable_safety_checker", False)),
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
