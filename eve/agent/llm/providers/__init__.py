from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any
from eve.agent.session.models import LLMContext, LLMResponse


class LLMProvider(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def prompt(self, context: LLMContext) -> LLMResponse:
        pass

    @abstractmethod
    async def prompt_stream(self, context: LLMContext) -> AsyncGenerator[Any, None]:
        pass
