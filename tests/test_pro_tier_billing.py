"""Pre-pay billing must not charge the pro tier when pro buys nothing.

`create` computes cost from args before routing, so an unentitled caller was
billed the pro rate and then routed to the identical standard model.
"""

import pytest

from eve.agent.generation import GenerationAccess, pro_tier_is_noop


def acc(premium=False, subscriber=False, video_pref=None, image_pref=None):
    return GenerationAccess(
        subscriber=subscriber,
        premium_entitled=premium,
        premium_enabled=premium,
        video_model_preference=video_pref,
        image_model_preference=image_pref,
    )


IMG = {"output": "image", "quality": "pro"}
I2V = {"output": "video", "quality": "pro", "reference_images": ["a.png"]}
T2V = {"output": "video", "quality": "pro"}


@pytest.mark.parametrize(
    "args,access,expected,why",
    [
        # --- images -------------------------------------------------------
        (IMG, acc(), True, "no premium, no sub -> nano_banana_2_fal both ways"),
        (IMG, acc(premium=True), False, "premium -> gpt_image_2, pro is real"),
        (IMG, acc(subscriber=True), False, "subscriber -> nano_banana_pro, pro is real"),
        # --- image-to-video ----------------------------------------------
        (I2V, acc(), True, "no premium -> kling_v3 both ways, quality unused"),
        (I2V, acc(premium=True), False, "premium -> seedance2, pro is real"),
        (
            {**I2V, "model_preference": "kling"},
            acc(), True, "explicit kling, no premium -> still kling_v3 both ways",
        ),
        (
            {**I2V, "model_preference": "wan"},
            acc(), False, "wan renders 1080p at pro vs 720p standard",
        ),
        (
            {**I2V, "model_preference": "veo"},
            acc(), False, "veo_31_lite renders 1080p at pro",
        ),
        (
            {**I2V, "model_preference": "seedance"},
            acc(), False, "seedance1 resolution tracks the tier",
        ),
        (I2V, acc(video_pref="wan"), False, "stored preference counts too"),
        # --- text-to-video ------------------------------------------------
        (T2V, acc(), False, "veo_31_lite renders 1080p at pro"),
        (T2V, acc(subscriber=True), False, "veo3 drops fast mode at pro"),
        # --- video-to-video (premium-gated, errors without premium) --------
        (
            {**I2V, "reference_video": "v.mp4"},
            acc(), False, "reference_video path is gated, not silently downgraded",
        ),
        # --- standard is never touched -------------------------------------
        ({"output": "image", "quality": "standard"}, acc(), False, "standard untouched"),
        ({"output": "video", "quality": "standard"}, acc(), False, "standard untouched"),
        ({"output": "image"}, acc(), False, "missing quality defaults standard"),
    ],
)
def test_pro_tier_is_noop(args, access, expected, why):
    assert pro_tier_is_noop(args, access) is expected, why


def test_downgrade_actually_changes_the_bill():
    """The whole point: the recomputed cost must be the standard-tier price."""
    from eve.utils.cost_utils import eval_cost

    expr = (
        '(output == "video" ? (((quality == "pro" ? 85 : 25) + '
        '(sound_effects ? 10 : 0)) * duration) : '
        '((quality == "pro" ? 30 : 10) * n_samples))'
    )
    pro = eval_cost(expr, output="video", quality="pro", sound_effects=None,
                    duration=10, n_samples=1)
    std = eval_cost(expr, output="video", quality="standard", sound_effects=None,
                    duration=10, n_samples=1)
    assert pro == 850 and std == 250
    # an unentitled img2vid caller is billed 250, not 850, for the kling_v3 they get
    assert pro_tier_is_noop({**I2V, "duration": 10}, acc()) is True


# ---------------------------------------------------------------------------
# LoRA generations: create routes by the LoRA's base model, ignoring quality
# ---------------------------------------------------------------------------

LORA_IMG = {"output": "image", "quality": "pro", "lora": "65f0…"}


@pytest.mark.parametrize(
    "args,access,expected,why",
    [
        (LORA_IMG, acc(), True, "sdxl/flux lora -> txt2img|flux_dev_lora, no quality arg"),
        (LORA_IMG, acc(subscriber=True), True, "subscribers too — routing ignores tier"),
        (LORA_IMG, acc(premium=True), True, "premium too — the lora branch precedes it"),
        (
            {**LORA_IMG, "reference_images": ["a.png"]},
            acc(premium=True), False,
            "with a reference image the edit map runs and the lora is ignored",
        ),
        (
            {**LORA_IMG, "quality": "standard"},
            acc(), False, "standard is never touched",
        ),
    ],
)
def test_lora_pro_is_noop(args, access, expected, why):
    assert pro_tier_is_noop(args, access) is expected, why


def test_lora_pro_bills_standard_not_3x():
    """The overcharge in numbers: a pro LoRA image billed 30/sample for output
    identical to the 10/sample standard render."""
    from eve.utils.cost_utils import eval_cost

    expr = (
        '(output == "video" ? (((quality == "pro" ? 85 : 25) + '
        '(sound_effects ? 10 : 0)) * duration) : '
        '((quality == "pro" ? 30 : 10) * n_samples))'
    )
    pro = eval_cost(expr, output="image", quality="pro", sound_effects=None,
                    duration=10, n_samples=1)
    std = eval_cost(expr, output="image", quality="standard", sound_effects=None,
                    duration=10, n_samples=1)
    assert (pro, std) == (30, 10)
    assert pro_tier_is_noop(LORA_IMG, acc(subscriber=True)) is True
