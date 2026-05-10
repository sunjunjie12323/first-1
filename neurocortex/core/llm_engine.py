from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        api_type: str = "ollama",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self.api_type = api_type
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> str:
        if self.api_type == "ollama":
            return await self._generate_ollama(prompt, system, max_tokens, temperature)
        return await self._generate_openai(prompt, system, max_tokens, temperature)

    async def get_embedding(self, text: str) -> np.ndarray:
        if self.api_type == "ollama":
            return await self._embedding_ollama(text)
        return await self._embedding_openai(text)

    async def generate_with_memory_context(
        self,
        prompt: str,
        memory_context: str = "",
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        enriched_system = system
        if memory_context:
            enriched_system = f"{system}\n\nRelevant memory context:\n{memory_context}" if system else f"Relevant memory context:\n{memory_context}"
        return await self.generate(prompt, enriched_system, max_tokens, temperature)

    async def health_check(self) -> bool:
        try:
            if self.api_type == "ollama":
                response = await self._client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
            response = await self._client.get(f"{self.base_url}/models", headers=self._get_headers())
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _generate_ollama(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        response = await self._client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

    async def _generate_openai(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _embedding_ollama(self, text: str) -> np.ndarray:
        payload = {"model": self.embedding_model, "prompt": text}
        response = await self._client.post(f"{self.base_url}/api/embeddings", json=payload)
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    async def _embedding_openai(self, text: str) -> np.ndarray:
        payload = {"model": self.embedding_model, "input": text}
        response = await self._client.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
