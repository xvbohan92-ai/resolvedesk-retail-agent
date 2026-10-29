"""Reproduce the local decoding profile; this does not train or download model weights."""

import hashlib
import json
import os

import httpx

from resolve_desk.providers import local_url
from training.common import ROOT


def main():
    url = local_url(os.getenv("RESOLVEDESK_OLLAMA_URL", "http://127.0.0.1:11434"))
    with httpx.Client(trust_env=False, timeout=120) as c:
        r = c.post(url + "/api/show", json={"model": "qwen3:4b"})
        r.raise_for_status()
        original = r.json()["template"]
        needle = "<think>\n{{ end }}"
        if original.count(needle) != 1:
            raise ValueError(
                "Installed template differs; inspect it rather than rewriting an unknown template"
            )
        template = original.replace(needle, "<think>\n\n</think>\n\n{{ end }}")
        current = c.post(url + "/api/show", json={"model": "resolvedesk-qwen3:4b"})
        if current.status_code == 200:
            if current.json()["template"] != template:
                raise ValueError("Existing ResolveDesk profile differs; no silent replacement")
        else:
            current.raise_for_status() if current.status_code != 404 else None
            r = c.post(
                url + "/api/create",
                json={
                    "model": "resolvedesk-qwen3:4b",
                    "from": "qwen3:4b",
                    "template": template,
                    "stream": False,
                },
            )
            r.raise_for_status()
    record = {
        "base": "qwen3:4b",
        "profile": "resolvedesk-qwen3:4b",
        "weights_trained": False,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/ollama-profile.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
