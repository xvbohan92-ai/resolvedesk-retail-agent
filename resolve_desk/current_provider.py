"""Ollama transport for the longer typed-intent action; legacy sessions retain v4."""

import json
import time

import httpx

from .protocol import Action
from .providers import Ollama


class CurrentOllama(Ollama):
    def generate(self, messages):
        visible = json.loads(messages[1]["content"])
        if "decision_gate" not in visible:
            return super().generate(messages)
        with self.lock, httpx.Client(trust_env=False, timeout=180) as client:
            started = time.perf_counter()
            response = client.post(
                self.url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "format": Action.model_json_schema(),
                    "keep_alive": "5m",
                    "options": {"temperature": 0, "seed": 202610275, "num_ctx": 4096, "num_predict": 512},
                },
            )
            response.raise_for_status()
            result = response.json()
        return result["message"].get("content", ""), {
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "input_tokens": result.get("prompt_eval_count"),
            "output_tokens": result.get("eval_count"),
            "finish_reason": result.get("done_reason"),
            "provider": "ollama",
            "model": self.model,
            "workflow": visible.get("workflow_version", "v5"),
            "thinking_returned": bool(result["message"].get("thinking")),
            "json_schema_constrained": True,
        }
