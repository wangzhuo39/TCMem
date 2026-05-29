from __future__ import annotations

from openai import OpenAI


class OpenAICompatibleLLMClient:
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: int = 120) -> None:
        if not api_key:
            raise ValueError("api_key is required for OpenAICompatibleLLMClient")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.json_max_attempts = 5
        self.json_retry_delay = 0.5

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return response.choices[0].message.content or ""
