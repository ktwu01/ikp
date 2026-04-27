"""OpenRouter API client with rate limiting, retries, and structured logging."""

import json
import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "api_cache"


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completions with rate limiting and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        requests_per_minute: int = 30,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: int = 60,
        cache_enabled: bool = True,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.base_url = base_url.rstrip("/")
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.cache_enabled = cache_enabled

        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, model: str, messages: list, temperature: float, max_tokens: int) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[dict]:
        if not self.cache_enabled:
            return None
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _set_cached(self, key: str, response: dict):
        if not self.cache_enabled:
            return
        cache_file = CACHE_DIR / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f)

    def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0,
        max_tokens: int = 150,
    ) -> dict:
        """Send a chat completion request. Returns the full API response dict."""
        cache_key = self._cache_key(model, messages, temperature, max_tokens)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {model}")
            return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ikp-research",
            "X-Title": "IKP Research",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    wait = self.retry_delay * attempt * 2
                    logger.warning(f"Rate limited on {model}, waiting {wait}s (attempt {attempt})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()

                # Track usage
                usage = result.get("usage", {})
                self._request_count += 1
                self._total_input_tokens += usage.get("prompt_tokens", 0)
                self._total_output_tokens += usage.get("completion_tokens", 0)

                self._set_cached(cache_key, result)
                return result

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {model} (attempt {attempt}/{self.max_retries})")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error on {model}: {e} (attempt {attempt}/{self.max_retries})")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"All {self.max_retries} attempts failed for {model}")

    def get_response_text(self, model: str, messages: list, **kwargs) -> str:
        """Convenience: return just the text content of the first choice."""
        result = self.chat_completion(model, messages, **kwargs)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(f"Unexpected response structure from {model}: {json.dumps(result)[:500]}")
            return ""

    def get_stats(self) -> dict:
        return {
            "total_requests": self._request_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
