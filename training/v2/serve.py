import json
import threading

import uvicorn
from fastapi import FastAPI
from mlx_lm import load
from pydantic import BaseModel, ConfigDict

from training.common import ROOT, sha
from training.runtime import generate_action
from training.v2.common import verify_frozen


def main():
    verify_frozen()
    report = json.loads((ROOT / "reports/v2/training.json").read_text())
    config = report["config"]
    adapter_sha = sha(ROOT / config["best_path"] / "adapters.safetensors")
    assert adapter_sha == report["selected_adapter_sha256"]
    model, tokenizer = load(
        ROOT / config["model"], adapter_path=ROOT / config["best_path"], trust_remote_code=False
    )
    app, lock = FastAPI(docs_url=None, redoc_url=None), threading.Lock()
    identity = {
        "provider": "mlx",
        "variant": "workflow_qlora_v2",
        "base": config["model_repo"],
        "base_revision": config["model_revision"],
        "adapter_sha256": adapter_sha,
        "training_report_sha256": sha(ROOT / "reports/v2/training.json"),
        "cloud_api": False,
        "workflow": "v2",
        "workflow_sha256": sha(ROOT / "resolve_desk/workflow.py"),
    }

    class Request(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        messages: list[dict[str, str]]

    @app.get("/health")
    def health():
        return identity

    @app.post("/generate")
    async def generate(body: Request):
        # MLX streams belong to the thread that loaded the model. Keep inference
        # on the event-loop/main thread instead of FastAPI's sync worker pool.
        with lock:
            text, timing = generate_action(model, tokenizer, body.messages, config)
        return {
            "text": text,
            "timing": {
                **timing,
                "provider": "mlx",
                "variant": "workflow_qlora_v2",
                "adapter_sha256": adapter_sha,
            },
        }

    uvicorn.run(app, host="127.0.0.1", port=8772)


if __name__ == "__main__":
    main()
