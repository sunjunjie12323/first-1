from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM API integration engine.

    Supports multiple local LLM backends:
    - Ollama (default, most common for local deployment)
    - OpenAI-compatible API (vLLM, llama.cpp server, etc.)
    - Custom API endpoints

    The LLM serves as the "reasoning engine" in the brain-inspired
    architecture, analogous to how the cerebral cortex performs
    complex reasoning and language generation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        api_type: str = "ollama",
        embedding_model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_type = api_type.lower()
        self.embedding_model = embedding_model or model
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        self._embedding_cache: Dict[str, List[float]] = {}

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text using the local LLM.
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        if self.api_type == "ollama":
            return await self._generate_ollama(
                prompt, system, temperature, max_tokens, stop
            )
        else:
            return await self._generate_openai_compatible(
                prompt, system, temperature, max_tokens, stop
            )

    async def generate_with_memory_context(
        self,
        user_message: str,
        reconstructed_memory: Optional[str] = None,
        working_memory_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response enriched with brain-inspired memory context.

        This is the key integration point where the LLM receives
        not just the user's message, but also:
        - Reconstructed memories (from hippocampus + neocortex)
        - Working memory context (from prefrontal cortex)
        - System personality/role definition

        The memory context is NOT just "stuffed" into the prompt —
        it's structured to mirror how the brain presents memory
        to consciousness: as reconstructed narratives, not raw data.
        """
        if system_prompt is None:
            system_prompt = (
                "你是一个具有类脑记忆系统的智能体。你像人一样记忆和回忆——"
                "不是机械地检索信息，而是从记忆碎片中重建回忆。"
                "你的记忆可能不完美，有时会模糊或整合，这是正常的。"
                "请自然地表达你的回忆和想法。"
            )

        memory_section = ""
        if reconstructed_memory:
            memory_section += f"\n\n【你的相关回忆】\n{reconstructed_memory}"
        if working_memory_context:
            memory_section += f"\n\n【当前工作记忆】\n{working_memory_context}"

        enhanced_prompt = user_message
        if memory_section:
            enhanced_prompt = (
                f"{user_message}\n"
                f"---\n（系统内部记忆上下文，不要直接复述，自然融入回答）{memory_section}"
            )

        return await self.generate(
            prompt=enhanced_prompt,
            system=system_prompt,
            temperature=temperature,
        )

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using the local LLM's
        embedding endpoint.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.api_type == "ollama":
            embedding = await self._get_embedding_ollama(text)
        else:
            embedding = await self._get_embedding_openai(text)

        if embedding:
            self._embedding_cache[text] = embedding
            if len(self._embedding_cache) > 10000:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]

        return embedding

    async def _generate_ollama(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Ollama HTTP error (attempt {attempt+1}): {e}")
            except httpx.RequestError as e:
                logger.warning(f"Ollama request error (attempt {attempt+1}): {e}")

        logger.error("All Ollama generation attempts failed")
        return ""

    async def _generate_openai_compatible(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                logger.warning(f"OpenAI API error (attempt {attempt+1}): {e}")
            except (httpx.RequestError, KeyError) as e:
                logger.warning(f"OpenAI request error (attempt {attempt+1}): {e}")

        logger.error("All OpenAI-compatible generation attempts failed")
        return ""

    async def _get_embedding_ollama(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"Ollama embedding error: {e}")
            return []

    async def _get_embedding_openai(self, text: str) -> List[float]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": text,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
            logger.warning(f"OpenAI embedding error: {e}")
            return []

    async def check_health(self) -> Dict[str, Any]:
        """Check if the LLM backend is accessible."""
        if self.api_type == "ollama":
            url = f"{self.base_url}/api/tags"
            try:
                response = await self._client.get(url)
                response.raise_for_status()
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return {
                    "status": "healthy",
                    "backend": "ollama",
                    "available_models": models,
                    "current_model": self.model,
                    "model_available": self.model in models,
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "backend": "ollama",
                    "error": str(e),
                }
        else:
            url = f"{self.base_url}/v1/models"
            try:
                response = await self._client.get(url)
                response.raise_for_status()
                return {
                    "status": "healthy",
                    "backend": "openai_compatible",
                    "current_model": self.model,
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "backend": "openai_compatible",
                    "error": str(e),
                }

    async def close(self) -> None:
        await self._client.aclose()

    def clear_embedding_cache(self) -> None:
        self._embedding_cache.clear()
