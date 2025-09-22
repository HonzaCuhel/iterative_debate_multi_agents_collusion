import os
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional retry support via tenacity
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:  # fallback no-op if tenacity is unavailable
    def retry(*args: Any, **kwargs: Any):
        def _decorator(fn):
            return fn
        return _decorator
    def stop_after_attempt(*args: Any, **kwargs: Any):
        return None
    def wait_exponential(*args: Any, **kwargs: Any):
        return None
    def retry_if_exception_type(*args: Any, **kwargs: Any):
        return None


# Minimal message type expected by callers
@dataclass
class AIMessage:
    content: str


class BaseChatModel:
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> AIMessage:
        raise NotImplementedError


class OpenAIChatModel(BaseChatModel):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
        )
        self.model = model
        self.temperature = temperature
        self.extra = extra or {}

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> AIMessage:
        # OpenAI Python SDK is synchronous; run in thread to avoid blocking
        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                messages=messages,
                **self.extra,
            )
            choice = resp.choices[0]
            content = getattr(choice.message, "content", None)
            if content is None and hasattr(choice, "text"):
                content = choice.text
            return content or ""

        @retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _call_with_retry() -> str:
            return _call()

        content = await asyncio.to_thread(_call_with_retry)
        return AIMessage(content=content)


class OllamaChatModel(BaseChatModel):
    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        temperature: float = 0,
    ) -> None:
        # The official ollama python client exposes sync and async APIs
        from ollama import Client  # type: ignore

        self.client = Client(host=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        self.model = model
        self.temperature = temperature

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> AIMessage:
        def _call() -> str:
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": kwargs.get("temperature", self.temperature)},
            )
            try:
                content = resp["message"]["content"]
            except Exception:
                content = getattr(getattr(resp, "message", object()), "content", "")
            return content or ""

        @retry(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _call_with_retry() -> str:
            return _call()

        content = await asyncio.to_thread(_call_with_retry)
        return AIMessage(content=content)


class AnthropicChatModel(BaseChatModel):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> None:
        # Lazy import to avoid hard dependency unless used
        from anthropic import Anthropic  # type: ignore

        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> AIMessage:
        # Transform OpenAI-style messages into Anthropic messages API
        def _to_anthropic_payload(msgs: List[Dict[str, str]]):
            system_parts: List[str] = []
            convo: List[Dict[str, str]] = []
            for m in msgs:
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if role == "system":
                    system_parts.append(content)
                elif role in ("user", "assistant"):
                    convo.append({"role": role, "content": content})
                else:
                    # Fallback: treat as user content
                    convo.append({"role": "user", "content": content})
            system_prompt = "\n".join(system_parts).strip() if system_parts else None
            return system_prompt, convo

        def _call() -> str:
            system_prompt, convo = _to_anthropic_payload(messages)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Call Anthropic Messages API
            if system_prompt:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=convo,
                )
            else:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=convo,
                )

            # Extract text from content blocks
            try:
                parts = getattr(resp, "content", []) or []
                texts: List[str] = []
                for p in parts:
                    try:
                        text = getattr(p, "text", None)
                        if text is None and isinstance(p, dict):
                            text = p.get("text")
                        if text:
                            texts.append(text)
                    except Exception:
                        continue
                return "\n".join(texts).strip()
            except Exception:
                try:
                    blocks = resp["content"]  # type: ignore[index]
                    if isinstance(blocks, list):
                        return "\n".join([b.get("text", "") for b in blocks if isinstance(b, dict)]).strip()
                except Exception:
                    pass
                return str(resp)

        @retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _call_with_retry() -> str:
            return _call()

        content = await asyncio.to_thread(_call_with_retry)
        return AIMessage(content=content)


def init_chat_model(
    model: str,
    provider: str = "openai",
    *,
    temperature: float = 0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    """
    Initialize a chat model abstraction.

    - provider="openai": uses OpenAI SDK. If base_url points to an OpenAI-compatible server
      (e.g., Ollama at http://localhost:11434/v1) it will work as well.
    - provider="ollama": uses native Ollama python client.
    """
    prov = (provider or "openai").lower()
    if prov == "openai":
        return OpenAIChatModel(model=model, api_key=api_key, base_url=base_url, temperature=temperature, extra=extra)
    if prov == "ollama":
        return OllamaChatModel(model=model, host=host, temperature=temperature)
    if prov == "anthropic":
        return AnthropicChatModel(model=model, api_key=api_key, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")


