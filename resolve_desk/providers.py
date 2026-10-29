"""Actual local inference clients; no cloud fallback or rule-generated model replies."""

import threading
import time
from urllib.parse import urlparse

import httpx

from .protocol import Action


def local_url(value):
    url = urlparse(value)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Inference endpoints must be local loopback HTTP URLs")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Inference URL cannot contain credentials, query, or fragment")
    return value.rstrip("/")


class Ollama:
    def __init__(self, url="http://127.0.0.1:11434", model="resolvedesk-qwen3:4b"):
        self.url, self.model = local_url(url), model
        self.lock = threading.Lock()

    def identity(self):
        with httpx.Client(trust_env=False, timeout=15) as client:
            version = client.get(self.url + "/api/version")
            version.raise_for_status()
            tags = client.get(self.url + "/api/tags")
            tags.raise_for_status()
        found = next((m for m in tags.json()["models"] if m["name"] == self.model), None)
        if not found:
            raise ValueError("Configured Ollama model is not installed; no automatic pull")
        return {
            "provider": "ollama",
            "model": self.model,
            "digest": found["digest"],
            "version": version.json()["version"],
            "transport": "application JSON dispatch",
        }

    def generate(self, messages):
        # Application JSON dispatch, not native tool_calls. Ollama needs a JSON grammar;
        # this path is a separate integration smoke, never the paired MLX training baseline.
        with self.lock, httpx.Client(trust_env=False, timeout=180) as client:
            start = time.perf_counter()
            response = client.post(
                self.url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "format": Action.model_json_schema(),
                    "keep_alive": "5m",
                    "options": {"temperature": 0, "seed": 20261026, "num_ctx": 4096, "num_predict": 160},
                },
            )
            response.raise_for_status()
            result = response.json()
        text = result["message"].get("content", "")
        return text, {
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "input_tokens": result.get("prompt_eval_count"),
            "output_tokens": result.get("eval_count"),
            "finish_reason": result.get("done_reason"),
            "provider": "ollama",
            "model": self.model,
            "thinking_returned": bool(result["message"].get("thinking")),
            "json_schema_constrained": True,
        }


class MLXClient:
    def __init__(self, url="http://127.0.0.1:8772"):
        self.url = local_url(url)

    def identity(self):
        with httpx.Client(trust_env=False, timeout=15) as c:
            r = c.get(self.url + "/health")
            r.raise_for_status()
            return r.json()

    def generate(self, messages):
        with httpx.Client(trust_env=False, timeout=180) as c:
            r = c.post(self.url + "/generate", json={"messages": messages})
            r.raise_for_status()
            result = r.json()
            return result["text"], result["timing"]
