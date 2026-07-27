"""Validate the args `create` builds against each target tool's REAL schema.

create advertises a superset of options (11 aspect ratios, n_samples to 15) and
forwards them to models that accept narrower sets. Any mismatch is a hard
pydantic validation error at runtime, i.e. a failed generation. This sweeps the
combination space and reports every (route, args) pair that would fail.
"""
import asyncio
import sys
from unittest.mock import patch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

from eve.tool import Tool  # noqa: E402
from eve.tools.media_utils.create import handler as ch  # noqa: E402
from eve.agent.generation import GenerationAccess  # noqa: E402

TARGETS = [
    "kling_v3", "seedance2", "seedance2_reference", "wan_27", "veo_31_lite",
    "veo3", "seedance1", "runway", "hedra", "nano_banana_2_fal",
    "nano_banana_pro", "gpt_image_2", "seedream45", "txt2img", "flux_dev_lora",
    "flux_dev", "gpt_image_15_edit",
]

# Pre-load REAL schemas BEFORE patching Tool.load, or the patch intercepts these too.
REAL = {}
for k in TARGETS:
    try:
        REAL[k] = Tool.load(k)
    except Exception:
        REAL[k] = None

CALLS = []


class ValidatingTool:
    def __init__(self, key):
        self.key = key
        self._t = REAL.get(key)

    @property
    def parameters(self):
        return getattr(self._t, "parameters", {}) or {}

    async def async_run(self, args, **kw):
        err = None
        if self._t is not None:
            try:
                self._t.prepare_args(dict(args))
            except Exception as e:
                err = str(e).replace("\n", " ")[:130]
        CALLS.append((self.key, err))
        return {"output": [{"filename": "x.mp4"}]}


def fake_load(key, **kw):
    return ValidatingTool(key)


def access(q="standard", sub=True, prem=True):
    return GenerationAccess(
        subscriber=sub, premium_entitled=prem, premium_enabled=prem,
        default_quality=q, image_model_preference=None, video_model_preference=None,
    )


RATIOS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4",
          "2:3", "9:16", "9:21"]
PREFS = [None, "kling", "wan", "seedance", "veo", "nano_banana", "seedream", "openai"]

fails = {}
total = 0
for out in ("video", "image"):
    for ar in RATIOS:
        for ns in (1, 4, 15):
            for q in ("standard", "pro"):
                for pref in PREFS:
                    for refs in ([], ["https://x/a.png"]):
                        CALLS.clear()
                        a = {"output": out, "prompt": "p", "aspect_ratio": ar,
                             "n_samples": ns, "quality": q, "duration": 5,
                             "reference_images": refs}
                        if pref:
                            a["model_preference"] = pref
                        with patch.object(ch.Tool, "load", side_effect=fake_load), \
                             patch("eve.agent.generation.resolve_generation_access",
                                   return_value=access(q=q)), \
                             patch.object(ch, "get_loras", return_value=[]), \
                             patch.object(ch, "get_media_attributes",
                                          return_value=({"aspect_ratio": 1.7}, None)), \
                             patch.object(ch, "get_full_url",
                                          side_effect=lambda f: "https://cdn/" + str(f)):
                            try:
                                fn = (ch.handle_video_creation if out == "video"
                                      else ch.handle_image_creation)
                                asyncio.run(fn(dict(a), user="u"))
                            except Exception:
                                pass
                        for key, err in CALLS:
                            total += 1
                            if err:
                                sig = (key, err.split("Input should be")[0][:70])
                                fails.setdefault(sig, []).append(
                                    f"out={out} ar={ar} n={ns} q={q} pref={pref} refs={len(refs)}")

print(f"sub-tool calls exercised: {total}")
print(f"distinct failure signatures: {len(fails)}\n")
for (key, msg), examples in sorted(fails.items()):
    print(f"  [{key}] {msg}")
    print(f"      {len(examples)} combos, e.g. {examples[0]}")
if not fails:
    print("  NONE — every route validates against its target tool's schema.")
