"""Routing matrix tests for the create tool's model selection + premium guardrail.

Mocks Tool.load and resolve_generation_access so no network/DB is touched;
asserts WHICH tool each (quality, preference, entitlement, opt-in) combination
routes to, and that premium tools are unreachable without both keys.
"""

from unittest.mock import patch

import pytest

from eve.agent.generation import GenerationAccess


class RecordingTool:
    """Stands in for any loaded tool; records the key it was loaded as."""

    def __init__(self, key, recorder):
        self.key = key
        self._recorder = recorder

    async def async_run(self, args, **kwargs):
        self._recorder.append((self.key, args))
        if self.key == "create":  # start-image pre-pass
            return {"output": [{"filename": "start.png"}]}
        return {"output": [{"filename": f"{self.key}.mp4", "url": "u"}]}


def make_access(subscriber=False, premium=False, quality=None, video_pref=None,
                image_pref=None):
    return GenerationAccess(
        paying_user=None,
        subscriber=subscriber,
        premium_entitled=premium,
        premium_enabled=premium,
        default_quality=quality or "standard",
        image_model_preference=image_pref,
        video_model_preference=video_pref,
    )


async def route_video(args, access):
    from eve.tools.media_utils.create import handler as create_handler

    calls = []
    with patch.object(create_handler.Tool, "load",
                      side_effect=lambda key, **kw: RecordingTool(key, calls)), \
         patch("eve.agent.generation.resolve_generation_access",
               return_value=access), \
         patch.object(create_handler, "get_loras", return_value=[]), \
         patch.object(create_handler, "get_media_attributes",
                      return_value=({}, None), create=True):
        try:
            await create_handler.handle_video_creation(dict(args), user="u1")
        except Exception:
            if not calls:
                raise
    video_calls = [k for k, _ in calls if k not in ("create", "thinksound")]
    return video_calls[0] if video_calls else None


IMG = ["https://x/img.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("args,access_kw,expected", [
    # img2vid default -> kling_v3 (the broken kling_v25 must be unreachable)
    ({"prompt": "p", "reference_images": IMG}, {}, "kling_v3"),
    # txt2vid standard default -> veo_31_lite (cheap tier)
    ({"prompt": "p"}, {}, "veo_31_lite"),
    # txt2vid pro subscriber (no premium opt-in) -> veo3
    ({"prompt": "p", "quality": "pro"}, {"subscriber": True}, "veo3"),
    # img2vid pro + premium -> seedance2
    ({"prompt": "p", "reference_images": IMG, "quality": "pro"},
     {"subscriber": True, "premium": True}, "seedance2"),
    # pro WITHOUT premium opt-in must NOT reach seedance2
    ({"prompt": "p", "reference_images": IMG, "quality": "pro"},
     {"subscriber": True}, "kling_v3"),
    # seedance preference without premium -> seedance1 downgrade
    ({"prompt": "p", "model_preference": "seedance"}, {}, "seedance1"),
    # seedance preference, pro + premium -> seedance2
    ({"prompt": "p", "model_preference": "seedance", "quality": "pro"},
     {"subscriber": True, "premium": True}, "seedance2"),
    # wan preference
    ({"prompt": "p", "model_preference": "wan"}, {}, "wan_27"),
    # kling preference on txt2vid maps to wan (kling_v3 is i2v-only)
    ({"prompt": "p", "model_preference": "kling"}, {}, "wan_27"),
    # veo preference without subscription -> veo_31_lite
    ({"prompt": "p", "model_preference": "veo"}, {}, "veo_31_lite"),
    # veo preference with subscription -> veo3
    ({"prompt": "p", "model_preference": "veo"}, {"subscriber": True}, "veo3"),
    # stored agent preference applies when request omits it
    ({"prompt": "p"}, {"video_pref": "wan"}, "wan_27"),
    # request arg beats stored preference
    ({"prompt": "p", "model_preference": "veo"},
     {"video_pref": "wan", "subscriber": True}, "veo3"),
])
async def test_video_routing(args, access_kw, expected):
    assert await route_video(args, make_access(**access_kw)) == expected


@pytest.mark.asyncio
async def test_reference_video_requires_premium():
    from eve.tools.media_utils.create import handler as create_handler

    with patch("eve.agent.generation.resolve_generation_access",
               return_value=make_access()), \
         patch.object(create_handler, "get_loras", return_value=[]):
        with pytest.raises(Exception, match="premium"):
            await create_handler.handle_video_creation(
                {"prompt": "p", "reference_video": "https://x/v.mp4"}, user="u1"
            )


@pytest.mark.asyncio
async def test_reference_video_premium_routes_seedance2_reference():
    args = {"prompt": "p", "reference_video": "https://x/v.mp4"}
    got = await route_video(args, make_access(subscriber=True, premium=True))
    assert got == "seedance2_reference"


async def route_image(args, access):
    from eve.tools.media_utils.create import handler as create_handler

    calls = []
    with patch.object(create_handler.Tool, "load",
                      side_effect=lambda key, **kw: RecordingTool(key, calls)), \
         patch("eve.agent.generation.resolve_generation_access",
               return_value=access), \
         patch.object(create_handler, "get_loras", return_value=[]):
        try:
            await create_handler.handle_image_creation(dict(args), user="u1")
        except Exception:
            if not calls:
                raise
    return calls[0][0] if calls else None


@pytest.mark.asyncio
@pytest.mark.parametrize("args,access_kw,expected", [
    # standard default
    ({"prompt": "p"}, {}, "nano_banana_2_fal"),
    # pro subscriber without premium -> nano_banana route (loads the
    # nano_banana_pro tool), NOT gpt_image_2
    ({"prompt": "p", "quality": "pro"}, {"subscriber": True}, "nano_banana_pro"),
    # pro + premium -> gpt_image_2
    ({"prompt": "p", "quality": "pro"},
     {"subscriber": True, "premium": True}, "gpt_image_2"),
    # openai preference without premium -> gpt_image_15_edit successor path
    ({"prompt": "p", "model_preference": "openai"}, {}, "gpt_image_15_edit"),
    # openai preference with premium -> gpt_image_2 at any quality
    ({"prompt": "p", "model_preference": "openai"},
     {"premium": True}, "gpt_image_2"),
    # stored image preference applies
    ({"prompt": "p"}, {"image_pref": "seedream"}, "seedream45"),
])
async def test_image_routing(args, access_kw, expected):
    assert await route_image(args, make_access(**access_kw)) == expected
