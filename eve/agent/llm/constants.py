from enum import Enum


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
