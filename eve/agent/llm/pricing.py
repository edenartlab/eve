from __future__ import annotations

from typing import Optional, Tuple

import litellm
from loguru import logger


def calculate_cost_usd(
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Tuple[float, float, float]:
    """Return (prompt_cost, completion_cost, total_cost)."""
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        logger.warning(f"Failed to calculate cost for model {model}: {exc}")
        return 0.0, 0.0, 0.0

    total = (prompt_cost or 0.0) + (completion_cost or 0.0)
    return prompt_cost or 0.0, completion_cost or 0.0, total
