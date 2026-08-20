"""Thin HTTP client for local Ollama LLM inference."""

import json
from typing import Any
import urllib.error
import urllib.request


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama service is unreachable or not running."""


class OllamaClient:
    """Model-agnostic HTTP client for Ollama's local REST API."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        default_model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.host = host.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a prompt to Ollama's /api/generate endpoint (blocking/non-streaming).

        Args:
            prompt: Raw prompt string.
            model: Target model name. Defaults to self.default_model.
            system: Optional system instruction prompt.
            options: Optional runtime parameters (e.g. {"temperature": 0.0}).
            timeout: Optional request timeout in seconds.

        Returns:
            The raw text response from the model.

        Raises:
            ValueError: If prompt is empty.
            OllamaConnectionError: If Ollama cannot be reached at self.host.
            OllamaError: If the API returns an error or malformed payload.
        """
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        target_model = model or self.default_model
        endpoint = f"{self.host}/api/generate"

        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        effective_timeout = timeout if timeout is not None else self.timeout

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as response:
                response_bytes = response.read()
                response_json = json.loads(response_bytes.decode("utf-8"))
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError) or "Connection refused" in str(exc):
                raise OllamaConnectionError(
                    f"Could not connect to Ollama at {self.host}. "
                    "Make sure the Ollama service is running (e.g., `ollama serve`)."
                ) from exc
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama at {self.host}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Received invalid JSON from Ollama: {exc}") from exc
        except Exception as exc:
            raise OllamaError(f"Unexpected error communicating with Ollama: {exc}") from exc

        if "error" in response_json:
            raise OllamaError(f"Ollama API error: {response_json['error']}")

        if "response" not in response_json:
            raise OllamaError(f"Malformed Ollama response, missing 'response' field: {response_json}")

        return str(response_json["response"])


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    system: str | None = None,
    options: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Convenience wrapper for single-shot generation."""
    client = OllamaClient(host=host, default_model=model, timeout=timeout)
    return client.generate(prompt, system=system, options=options)
