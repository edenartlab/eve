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
    # flux EDIT preference no longer routes to the retired flux_kontext; it falls
    # through to the tier default (NB2). Regression guard for the 2026-07 retirement.
    ({"prompt": "p", "model_preference": "flux",
      "reference_images": ["https://example.com/a.png"]}, {}, "nano_banana_2_fal"),
    # flux GENERATION preference is a DIFFERENT path and stays on the FLUX.1 LoRA
    # tool — retiring flux_kontext must not touch LoRA workflows.
    ({"prompt": "p", "model_preference": "flux"}, {}, "flux_dev_lora"),
])
async def test_image_routing(args, access_kw, expected):
    assert await route_image(args, make_access(**access_kw)) == expected


@pytest.mark.asyncio
async def test_content_policy_video_fallback():
    from eve.tools.media_utils.create import handler as ch

    # 1. detection
    assert ch._is_content_policy_error(
        Exception("Rejected by the provider's content policy: resembles a real person"))
    assert ch._is_content_policy_error(Exception("content_policy_violation likeness"))
    assert not ch._is_content_policy_error(Exception("FAL server error (500): boom"))

    # 2. fallback chain
    f = ch._permissive_video_fallbacks
    assert f({"reference_images": ["a"], "model_preference": "seedance"}) == ["kling", "wan"]
    assert f({"reference_images": ["a"]}) == ["wan"]              # default i2v == kling -> skip
    assert f({"reference_images": ["a"], "model_preference": "kling"}) == ["wan"]
    assert f({"reference_images": ["a"], "model_preference": "wan"}) == ["kling", "wan"]
    assert f({}) == []                                            # no start image (txt2vid)
    assert f({"reference_video": "v", "reference_images": ["a"]}) == []  # reference-to-video

    # 3. full path: strict rejects, kling also rejects, permissive wan succeeds
    calls = []

    async def fake_hvc(args, user, agent, cancellation_event):
        pref = args.get("model_preference")
        calls.append((pref, args.get("_permissive_fallback", False)))
        if pref == "seedance":
            raise Exception("content policy: resembles a real person")
        if pref == "kling":
            raise Exception("content_policy_violation likeness")
        return {"output": "http://vid", "subtool_calls": [{"tool": "wan_27", "args": {}}]}

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args = {"output": "video", "reference_images": ["a"], "model_preference": "seedance"}
    ctx.user, ctx.agent = "u", None
    with patch.object(ch, "handle_video_creation", side_effect=fake_hvc):
        res = await ch._create_video_with_policy_fallback(ctx, None)
    assert res["output"] == "http://vid"
    assert any(sc["tool"] == "content_policy_fallback" for sc in res["subtool_calls"])
    assert calls == [("seedance", False), ("kling", True), ("wan", True)]

    # 4. a NON-policy error is never swallowed by the fallback
    async def fake_raise(args, user, agent, cancellation_event):
        raise Exception("FAL server error (503): down")

    ctx2 = Ctx()
    ctx2.args = {"output": "video", "reference_images": ["a"]}
    ctx2.user, ctx2.agent = "u", None
    with patch.object(ch, "handle_video_creation", side_effect=fake_raise):
        with pytest.raises(Exception, match="503"):
            await ch._create_video_with_policy_fallback(ctx2, None)


@pytest.mark.asyncio
async def test_flux_kontext_never_loaded():
    """flux_kontext is retired — create must never Tool.load it, on any path."""
    for args in (
        {"prompt": "p", "model_preference": "flux",
         "reference_images": ["https://example.com/a.png"]},
        {"prompt": "p", "model_preference": "flux"},
    ):
        tool = await route_image(args, make_access())
        assert tool != "flux_kontext"
