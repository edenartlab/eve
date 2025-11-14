from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    FAKE = "fake"


class ModelTier(Enum):
    PREMIUM = "premium"
    FREE = "free"


DEFAULT_MODEL_PREMIUM = "claude-sonnet-4-5"
DEFAULT_MODEL_FREE = "claude-haiku-4-5"

FALLBACK_MODEL_PREMIUM = "gpt-5"
FALLBACK_MODEL_FREE = "gpt-5-nano"

TEST_MODE_TEXT_STRING = "===test"
TEST_MODE_TOOL_STRING = "===tool"

MODEL_PROVIDER_OVERRIDES = {
    # Anthropic
    "claude-sonnet-4-5": ModelProvider.ANTHROPIC,
    "claude-4-5-sonnet": ModelProvider.ANTHROPIC,
    "claude-haiku-4-5": ModelProvider.ANTHROPIC,
    "claude-4-5-haiku": ModelProvider.ANTHROPIC,
    # OpenAI
    "gpt-4o-mini": ModelProvider.OPENAI,
    "gpt-5": ModelProvider.OPENAI,
    "gpt-5-nano": ModelProvider.OPENAI,
    "gpt-5-mini": ModelProvider.OPENAI,
    # Gemini
    "gemini-2.5-pro": ModelProvider.GEMINI,
    "gemini-2.5-flash": ModelProvider.GEMINI,
}


@dataclass(frozen=True)
class ModelDefaults:
    model: str
    provider: ModelProvider
    fallbacks: Tuple[Tuple[str, ModelProvider], ...] = ()


_DEFAULT_MODEL_MAP = {
    ModelTier.PREMIUM: ModelDefaults(
        model=DEFAULT_MODEL_PREMIUM,
        provider=ModelProvider.ANTHROPIC,
        fallbacks=((FALLBACK_MODEL_PREMIUM, ModelProvider.OPENAI),),
    ),
    ModelTier.FREE: ModelDefaults(
        model=DEFAULT_MODEL_FREE,
        provider=ModelProvider.ANTHROPIC,
        fallbacks=((FALLBACK_MODEL_FREE, ModelProvider.OPENAI),),
    ),
}


def get_provider_for_model(model_name: str) -> ModelProvider:
    """Infer the model provider from a model identifier."""
    name = model_name.lower()
    if name.startswith("claude"):
        return ModelProvider.ANTHROPIC
    if name.startswith("gpt"):
        return ModelProvider.OPENAI
    if name.startswith("gemini"):
        return ModelProvider.GEMINI
    return ModelProvider.OPENAI


def get_default_model_defaults(tier: ModelTier = ModelTier.PREMIUM) -> ModelDefaults:
    """Return the default model/provider and fallback chain for a tier."""
    defaults = _DEFAULT_MODEL_MAP.get(tier)
    if defaults:
        return defaults
    # Fallback to premium if an unknown tier is provided
    return _DEFAULT_MODEL_MAP[ModelTier.PREMIUM]
