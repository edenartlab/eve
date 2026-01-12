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

#DEFAULT_MODEL_PREMIUM = "gemini-3-flash-preview"

FALLBACK_MODEL_PREMIUM = "gpt-5.2"
FALLBACK_MODEL_FREE = "gpt-5-nano"

TEST_MODE_TEXT_STRING = "===test"
TEST_MODE_TOOL_STRING = "===tool"

MODEL_PROVIDER_OVERRIDES = {
    # Anthropic
    "claude-4-5-sonnet": ModelProvider.ANTHROPIC,
    "claude-4-5-haiku": ModelProvider.ANTHROPIC,
    # OpenAI
    "gpt-4o-mini": ModelProvider.OPENAI,
    "gpt-5.2": ModelProvider.OPENAI,
    "gpt-5-nano": ModelProvider.OPENAI,
    "gpt-5-mini": ModelProvider.OPENAI,
    # Gemini
    #"gemini-3-pro-preview": ModelProvider.GEMINI,
    "gemini-3-flash-preview": ModelProvider.GEMINI,
}


@dataclass(frozen=True)
class ModelDefaults:
    model: str
    fallback_models: Tuple[str, ...] = ()

    @property
    def provider(self) -> ModelProvider:
        return get_provider_for_model(self.model)

    @property
    def fallbacks(self) -> Tuple[Tuple[str, ModelProvider], ...]:
        return tuple((m, get_provider_for_model(m)) for m in self.fallback_models)


_DEFAULT_MODEL_MAP = {
    ModelTier.PREMIUM: ModelDefaults(
        model=DEFAULT_MODEL_PREMIUM,
        fallback_models=(FALLBACK_MODEL_PREMIUM,),
    ),
    ModelTier.FREE: ModelDefaults(
        model=DEFAULT_MODEL_FREE,
        fallback_models=(FALLBACK_MODEL_FREE,),
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
