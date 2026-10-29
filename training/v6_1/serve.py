import json
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from training.common import ROOT, sha
from training.v6_1.common import DATA, REPORTS, verify_frozen, verify_local_base
from training.v6_1.resources import wait_for_room


def main():
    import mlx.core as mx
    from mlx_lm import load

    from training.runtime import generate_action

    verify_frozen()
    report = json.loads((REPORTS / "training.json").read_text())
    config = report["evaluation_config"]
    verify_local_base(config)
    path = ROOT / config["best_path"]
    assert sha(path / "adapters.safetensors") == report["selected_adapter_sha256"]
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    wait_for_room()
    model, tokenizer = load(ROOT / config["model"], adapter_path=path, trust_remote_code=False)
    app, lock = FastAPI(docs_url=None, redoc_url=None), threading.Lock()
    identity = {
        "provider": "mlx",
        "variant": "qlora_v6_1",
        "workflow": "v6_1",
        "model_interface": "v6",
        "cloud_api": False,
        "base": config["model_repo"],
        "base_revision": config["model_revision"],
        "adapter_sha256": report["selected_adapter_sha256"],
        "training_report_sha256": sha(REPORTS / "training.json"),
        "manifest_sha256": sha(DATA / "manifest.json"),
    }

    class Request(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        messages: list[dict[str, str]]

    @app.get("/health")
    def health():
        return identity

    @app.post("/generate")
    async def generate(body: Request):
        if len(body.messages) != 2 or json.loads(body.messages[1]["content"]).get("workflow_version") != "v6":
            raise HTTPException(400, "This adapter service supports scoped-consent v6 sessions.")
        with lock:
            raw, timing = generate_action(model, tokenizer, body.messages, config)
            mx.clear_cache()
        return {
            "text": raw,
            "timing": {
                **timing,
                "provider": "mlx",
                "variant": "qlora_v6_1",
                "adapter_sha256": identity["adapter_sha256"],
            },
        }

    uvicorn.run(app, host="127.0.0.1", port=8772)


if __name__ == "__main__":
    main()
