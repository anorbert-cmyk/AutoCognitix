"""
LLM Provider Abstraction Layer for AutoCognitix.

This module provides a unified interface for multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (Local models)

Features:
- Automatic provider detection based on available API keys
- Fallback chain support
- Async and sync interfaces
- Structured output parsing
- Token counting and rate limiting support

Author: AutoCognitix Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

# Python 3.9 compatible string enum
from enum import Enum


logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class LLMProviderType(str, Enum):
    """Supported LLM provider types (Python 3.9 compatible)."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    RULE_BASED = "rule_based"  # Fallback when no API available

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class LLMMessage:
    """Message for LLM conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    provider: LLMProviderType
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0) or self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0) or self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMStreamChunk:
    """A chunk of streaming response from LLM."""

    content: str
    model: str
    provider: LLMProviderType
    is_final: bool = False
    finish_reason: str | None = None


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""

    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False


# =============================================================================
# Abstract Base Provider
# =============================================================================


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._initialized = False

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (has valid credentials)."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ):
        """
        Generate a streaming response from the LLM.

        Yields LLMStreamChunk objects as the response is generated.
        Default implementation falls back to non-streaming.

        Args:
            messages: List of messages for the conversation.
            config: Optional LLM configuration override.

        Yields:
            LLMStreamChunk objects with partial content.
        """
        # Default fallback: use non-streaming and yield single chunk
        response = await self.generate(messages, config)
        yield LLMStreamChunk(
            content=response.content,
            model=response.model,
            provider=response.provider,
            is_final=True,
            finish_reason=response.finish_reason,
        )

    async def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """
        Convenience method to generate with system and user prompts.

        Args:
            system_prompt: System prompt for context setting.
            user_prompt: User prompt with the actual request.
            config: Optional LLM configuration override.

        Returns:
            LLMResponse with generated content.
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        return await self.generate(messages, config)

    async def generate_stream_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig | None = None,
    ):
        """
        Convenience method to generate streaming response with system and user prompts.

        Args:
            system_prompt: System prompt for context setting.
            user_prompt: User prompt with the actual request.
            config: Optional LLM configuration override.

        Yields:
            LLMStreamChunk objects with partial content.
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        async for chunk in self.generate_stream(messages, config):
            yield chunk


# =============================================================================
# Anthropic Provider
# =============================================================================


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self._client = None

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.ANTHROPIC

    @property
    def model_name(self) -> str:
        return settings.ANTHROPIC_MODEL

    def is_available(self) -> bool:
        return bool(settings.ANTHROPIC_API_KEY)

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info(f"Anthropic client initialized with model: {self.model_name}")
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate response using Anthropic Claude."""
        cfg = config or self.config
        client = self._get_client()

        # Extract system message
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        try:
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                system=system_content,
                messages=api_messages,
                stop_sequences=cfg.stop_sequences if cfg.stop_sequences else None,
            )

            content = ""
            if response.content:
                content = response.content[0].text

            return LLMResponse(
                content=content,
                model=self.model_name,
                provider=self.provider_type,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ):
        """Generate streaming response using Anthropic Claude."""
        cfg = config or self.config
        client = self._get_client()

        # Extract system message
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        try:
            async with client.messages.stream(
                model=self.model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                system=system_content,
                messages=api_messages,
                stop_sequences=cfg.stop_sequences if cfg.stop_sequences else None,
            ) as stream:
                async for text in stream.text_stream:
                    yield LLMStreamChunk(
                        content=text,
                        model=self.model_name,
                        provider=self.provider_type,
                        is_final=False,
                    )

                # Final chunk
                final_message = await stream.get_final_message()
                yield LLMStreamChunk(
                    content="",
                    model=self.model_name,
                    provider=self.provider_type,
                    is_final=True,
                    finish_reason=final_message.stop_reason,
                )

        except Exception as e:
            logger.error(f"Anthropic streaming API error: {e}")
            raise


# =============================================================================
# OpenAI Provider
# =============================================================================


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self._client = None

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI

    @property
    def model_name(self) -> str:
        return settings.OPENAI_MODEL

    def is_available(self) -> bool:
        return bool(settings.OPENAI_API_KEY)

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"OpenAI client initialized with model: {self.model_name}")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate response using OpenAI GPT."""
        cfg = config or self.config
        client = self._get_client()

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                stop=cfg.stop_sequences if cfg.stop_sequences else None,
            )

            choice = response.choices[0]

            return LLMResponse(
                content=choice.message.content or "",
                model=self.model_name,
                provider=self.provider_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ):
        """Generate streaming response using OpenAI GPT."""
        cfg = config or self.config
        client = self._get_client()

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                stop=cfg.stop_sequences if cfg.stop_sequences else None,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield LLMStreamChunk(
                        content=chunk.choices[0].delta.content,
                        model=self.model_name,
                        provider=self.provider_type,
                        is_final=False,
                    )

                if chunk.choices and chunk.choices[0].finish_reason:
                    yield LLMStreamChunk(
                        content="",
                        model=self.model_name,
                        provider=self.provider_type,
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            logger.error(f"OpenAI streaming API error: {e}")
            raise


# =============================================================================
# Ollama Provider
# =============================================================================


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self._client = None

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OLLAMA

    @property
    def model_name(self) -> str:
        return settings.OLLAMA_MODEL

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import httpx

            response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate response using Ollama."""
        cfg = config or self.config

        try:
            import httpx

            # Convert messages to Ollama format
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": api_messages,
                        "stream": False,
                        "options": {
                            "temperature": cfg.temperature,
                            "num_predict": cfg.max_tokens,
                            "top_p": cfg.top_p,
                        },
                    },
                    timeout=120.0,
                )

                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=self.model_name,
                    provider=self.provider_type,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
                    finish_reason=data.get("done_reason", "stop"),
                    raw_response=data,
                )

        except ImportError:
            raise ImportError("httpx package not installed. Run: pip install httpx")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


# =============================================================================
# Rule-Based Fallback Provider
# =============================================================================


class RuleBasedProvider(BaseLLMProvider):
    """
    Rule-based fallback provider when no LLM API is available.

    This provider returns structured responses based on pattern matching
    and database lookups, without using an LLM.
    """

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.RULE_BASED

    @property
    def model_name(self) -> str:
        return "rule-based-v1"

    def is_available(self) -> bool:
        return True  # Always available as fallback

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """
        Generate a response using rule-based logic.

        This is a placeholder that returns a message indicating
        rule-based processing is being used.
        """
        # Extract user message
        for msg in messages:
            if msg.role == "user":
                break

        # Return a basic response
        response_content = (
            "A rendszer jelenleg szabaly-alapu modban mukodik (nincs LLM API elerheto). "
            "A diagnosztika az adatbazisban talalhato informaciok alapjan keszult.\n\n"
            "A pontos diagnosztika erdekeben keressen fel egy szakszerviz muhelyt."
        )

        return LLMResponse(
            content=response_content,
            model=self.model_name,
            provider=self.provider_type,
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="stop",
        )


# =============================================================================
# LLM Provider Factory
# =============================================================================


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    _providers: dict[LLMProviderType, type[BaseLLMProvider]] = {
        LLMProviderType.ANTHROPIC: AnthropicProvider,
        LLMProviderType.OPENAI: OpenAIProvider,
        LLMProviderType.OLLAMA: OllamaProvider,
        LLMProviderType.RULE_BASED: RuleBasedProvider,
    }

    _instances: dict[LLMProviderType, BaseLLMProvider] = {}

    @classmethod
    def get_provider(
        cls,
        provider_type: LLMProviderType | None = None,
        config: LLMConfig | None = None,
    ) -> BaseLLMProvider:
        """
        Get or create a provider instance.

        Args:
            provider_type: Specific provider to use. If None, auto-detect.
            config: Optional LLM configuration.

        Returns:
            BaseLLMProvider instance.
        """
        if provider_type is None:
            provider_type = cls._auto_detect_provider()

        if provider_type not in cls._instances:
            provider_class = cls._providers.get(provider_type)
            if provider_class is None:
                raise ValueError(f"Unknown provider type: {provider_type}")
            cls._instances[provider_type] = provider_class(config)

        return cls._instances[provider_type]

    @classmethod
    def _auto_detect_provider(cls) -> LLMProviderType:
        """
        Auto-detect the best available provider.

        Checks providers in order of preference:
        1. Configured provider (from settings)
        2. Anthropic (if API key available)
        3. OpenAI (if API key available)
        4. Ollama (if running locally)
        5. Rule-based fallback
        """
        # Check configured provider first
        configured = settings.LLM_PROVIDER.lower()
        if configured in ("anthropic", "openai", "ollama"):
            provider_type = LLMProviderType(configured)
            provider = cls._providers[provider_type]()
            if provider.is_available():
                logger.info(f"Using configured LLM provider: {configured}")
                return provider_type

        # Try providers in order of preference
        for provider_type in [
            LLMProviderType.ANTHROPIC,
            LLMProviderType.OPENAI,
            LLMProviderType.OLLAMA,
        ]:
            provider_class = cls._providers[provider_type]
            provider = provider_class()
            if provider.is_available():
                logger.info(f"Auto-detected LLM provider: {provider_type.value}")
                return provider_type

        # Fall back to rule-based
        logger.warning("No LLM API available, using rule-based fallback")
        return LLMProviderType.RULE_BASED

    @classmethod
    def get_available_providers(cls) -> list[LLMProviderType]:
        """Get list of all available providers."""
        available = []
        for provider_type, provider_class in cls._providers.items():
            provider = provider_class()
            if provider.is_available():
                available.append(provider_type)
        return available

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached provider instances."""
        cls._instances.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_provider: BaseLLMProvider | None = None


def get_llm_provider(
    provider_type: LLMProviderType | None = None,
    config: LLMConfig | None = None,
) -> BaseLLMProvider:
    """
    Get an LLM provider instance.

    Args:
        provider_type: Specific provider to use. If None, auto-detect.
        config: Optional LLM configuration.

    Returns:
        BaseLLMProvider instance ready for use.
    """
    return LLMProviderFactory.get_provider(provider_type, config)


async def generate_response(
    system_prompt: str,
    user_prompt: str,
    provider_type: LLMProviderType | None = None,
    config: LLMConfig | None = None,
) -> LLMResponse:
    """
    Convenience function to generate a response.

    Args:
        system_prompt: System prompt for context.
        user_prompt: User prompt with request.
        provider_type: Optional specific provider.
        config: Optional LLM configuration.

    Returns:
        LLMResponse with generated content.
    """
    provider = get_llm_provider(provider_type, config)
    return await provider.generate_with_system(system_prompt, user_prompt, config)


def is_llm_available() -> bool:
    """Check if any LLM provider (except rule-based) is available."""
    available = LLMProviderFactory.get_available_providers()
    return any(p != LLMProviderType.RULE_BASED for p in available)


def get_current_provider_info() -> dict[str, Any]:
    """Get information about the current LLM provider."""
    provider = get_llm_provider()
    return {
        "provider": provider.provider_type.value,
        "model": provider.model_name,
        "available": provider.is_available(),
    }
