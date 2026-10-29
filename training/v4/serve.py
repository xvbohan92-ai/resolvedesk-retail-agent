import json
import threading

import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from mlx_lm import load
from pydantic import BaseModel, ConfigDict

from resolve_desk.decision_facts import augment_messages
from training.common import ROOT, sha
from training.runtime import generate_action
from training.v4.action_preference import MANIFEST as ACTION_MANIFEST
from training.v4.action_preference import verify_protocol as verify_action_protocol
from training.v4.calibration import MANIFEST as CALIBRATION_MANIFEST
from training.v4.calibration import verify_protocol as verify_calibration
from training.v4.common import verify_frozen
from training.v4.preference import MANIFEST, verify_protocol
from training.v4.recovery import MANIFEST as RECOVERY_MANIFEST
from training.v4.recovery import verify_protocol as verify_recovery


def main():
    verify_frozen()
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    report = json.loads((ROOT / "reports/v4/training.json").read_text())
    verify_protocol()
    verify_action_protocol()
    verify_recovery()
    verify_calibration()
    assert report["calibration_manifest_sha256"] == sha(CALIBRATION_MANIFEST)
    assert report["recovery_manifest_sha256"] == sha(RECOVERY_MANIFEST)
    assert report["action_manifest_sha256"] == sha(ACTION_MANIFEST)
    assert report["preference_manifest_sha256"] == sha(MANIFEST)
    config = report["config"]
    adapter_sha = sha(ROOT / config["best_path"] / "adapters.safetensors")
    assert adapter_sha == report["selected_adapter_sha256"]
    model, tokenizer = load(
        ROOT / config["model"], adapter_path=ROOT / config["best_path"], trust_remote_code=False
    )
    app, lock = FastAPI(docs_url=None, redoc_url=None), threading.Lock()
    identity = {
        "provider": "mlx",
        "variant": "qlora_v4",
        "base": config["model_repo"],
        "base_revision": config["model_revision"],
        "adapter_sha256": adapter_sha,
        "training_report_sha256": sha(ROOT / "reports/v4/training.json"),
        "cloud_api": False,
        "workflow": "v4",
        "workflow_sha256": sha(ROOT / "resolve_desk/progress.py"),
        "facts_sha256": sha(ROOT / "resolve_desk/decision_facts.py"),
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
            text, timing = generate_action(model, tokenizer, augment_messages(body.messages), config)
            mx.clear_cache()
        return {
            "text": text,
            "timing": {
                **timing,
                "provider": "mlx",
                "variant": "qlora_v4",
                "adapter_sha256": adapter_sha,
            },
        }

    uvicorn.run(app, host="127.0.0.1", port=8772)


if __name__ == "__main__":
    main()
