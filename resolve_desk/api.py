import os
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .agent_v7 import advance
from .current_provider import CurrentOllama
from .fixtures import (
    load_demo_cases,
    load_v2_demo_cases,
    load_v3_demo_cases,
    load_v4_demo_cases,
    load_v5_demo_cases,
    load_v6_1_demo_cases,
    load_v6_demo_cases,
    load_v7_demo_cases,
)
from .progress import ProgressStore as Store
from .providers import MLXClient
from .resolution import ResolutionStore as PreviousResolutionStore
from .resolution_v6 import ResolutionStore as DevelopmentStore
from .resolution_v6 import current_request as current_v6_request
from .resolution_v6_1 import ResolutionStore as PreviousFinalStore
from .resolution_v6_1 import current_request as current_v61_request
from .resolution_v7 import ResolutionStore
from .resolution_v7 import current_request as current_final_request
from .store import Conflict

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="ResolveDesk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
store = Store(Path(os.getenv("RESOLVEDESK_DATA", str(Path.cwd() / "data"))) / "retail.sqlite")
typed_store = ResolutionStore(store.path)
previous_typed_store = PreviousResolutionStore(store.path)
development_store = DevelopmentStore(store.path)
previous_final_store = PreviousFinalStore(store.path)
providers = {
    "ollama": CurrentOllama(
        os.getenv("RESOLVEDESK_OLLAMA_URL", "http://127.0.0.1:11434"),
        os.getenv("RESOLVEDESK_OLLAMA_MODEL", "resolvedesk-qwen3:4b"),
    ),
    "finetuned": MLXClient(os.getenv("RESOLVEDESK_MLX_URL", "http://127.0.0.1:8772")),
}
case_map = {
    c["id"]: c
    for c in [
        *load_demo_cases(),
        *load_v2_demo_cases(),
        *load_v3_demo_cases(),
        *load_v4_demo_cases(),
        *load_v5_demo_cases(),
        *load_v6_demo_cases(),
        *load_v6_1_demo_cases(),
        *load_v7_demo_cases(),
    ]
}
locks, locks_guard = {}, threading.Lock()


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Start(Body):
    case_id: str
    provider: str = "ollama"
    experimental_flow: bool = False
    message: str | None = Field(default=None, min_length=1, max_length=2000)


class Continue(Body):
    revision: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=2000)
    provider: str = "ollama"


class Confirmation(Body):
    revision: int = Field(ge=0)
    proposal_id: str
    accepted: bool


def locked(sid):
    with locks_guard:
        return locks.setdefault(sid, threading.Lock())


def choose(name):
    if name not in providers:
        raise HTTPException(400, "Unknown inference provider")
    return providers[name]


def snapshot(sid):
    state = store.get(sid)
    # The browser receives actual trace and sandbox orders, not fixture answers or test gold labels.
    return {
        "session": state,
        "orders": store.orders(sid),
        "timeline": store.timeline(sid),
        "interpretation": (
            current_final_request(state)
            if state.get("workflow_version") == "v7"
            else current_v61_request(state)
            if state.get("workflow_version") == "v6_1"
            else current_v6_request(state)
            if state.get("workflow_version") == "v6"
            else None
        ),
    }


def inference_provider(sid, name):
    """Reject a mismatched model before a customer turn can revoke a proposal."""
    provider = choose(name)
    version = store.get(sid).get("workflow_version")
    try:
        if name == "finetuned":
            if version not in {"v5", "v6", "v6_1", "v7"}:
                raise ValueError(
                    "This QLoRA profile cannot continue a legacy session; use its original model."
                )
            if provider.identity().get("variant") != "qlora_" + version:
                raise ValueError(f"Start the matching QLoRA {version} service for this session")
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(503, f"Local model unavailable: {str(e)[:180]}") from e
    return provider


def infer(sid, name, checked_provider=None):
    try:
        version = store.get(sid).get("workflow_version")
        controller = {
            "v5": previous_typed_store,
            "v6": development_store,
            "v6_1": previous_final_store,
            "v7": typed_store,
        }.get(version, store)
        provider = checked_provider or inference_provider(sid, name)
        advance(controller, sid, provider)
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(503, f"Local model unavailable: {str(e)[:180]}") from e
    return snapshot(sid)


@app.exception_handler(Conflict)
async def conflict_handler(request, exception):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(exception)}, status_code=409)


@app.exception_handler(KeyError)
async def missing_handler(request, exception):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": "Session not found"}, status_code=404)


@app.exception_handler(ValueError)
async def value_handler(request, exception):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(exception)[:250]}, status_code=400)


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/api/cases")
def cases():
    return [
        {"id": c["id"], "family": c["family"], "message": c["initial_message"]} for c in case_map.values()
    ]


@app.get("/api/health")
def health():
    results = {}
    for name, provider in providers.items():
        try:
            identity = provider.identity()
            available = name != "finetuned" or identity.get("variant") == "qlora_v7"
            results[name] = {"available": available, **identity}
        except (httpx.HTTPError, ValueError) as e:
            results[name] = {"available": False, "error": str(e)[:150]}
    return {"models": results, "simulated": True}


@app.post("/api/sessions")
def start(body: Start):
    choose(body.provider)
    if body.case_id not in case_map:
        raise HTTPException(400, "Select a validation demo case")
    case = dict(case_map[body.case_id])
    if body.message is not None:
        if not body.message.strip():
            raise HTTPException(400, "Enter a customer request")
        case["initial_message"] = body.message
    # A rejected research candidate must remain an explicit experiment.
    controller = typed_store if body.experimental_flow or body.provider == "finetuned" else store
    state = controller.create(case)
    with locked(state["id"]):
        return infer(state["id"], body.provider)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return snapshot(sid)


@app.post("/api/sessions/{sid}/messages")
def customer_message(sid: str, body: Continue):
    with locked(sid):
        store.check_revision(store.get(sid), body.revision)
        provider = inference_provider(sid, body.provider)
        store.customer_message(sid, body.message, body.revision)
        return infer(sid, body.provider, checked_provider=provider)


@app.post("/api/sessions/{sid}/confirm")
def confirm(sid: str, body: Confirmation):
    with locked(sid):
        receipt = store.confirm(sid, body.proposal_id, body.revision, body.accepted)
        return {**snapshot(sid), "confirmation_result": receipt}


@app.get("/api/evidence")
def evidence(experiment: str = "v7"):
    from .evidence import validated_report, validated_v2_report, validated_v3_report, validated_v4_report
    from .evidence_cuda import validated_report as validated_cuda_report
    from .evidence_cuda2 import validated_report as validated_cuda2_report
    from .evidence_v5 import validated_report as validated_v5_report
    from .evidence_v6 import validated_report as validated_v6_report
    from .evidence_v6_1 import validated_report as validated_v61_report
    from .evidence_v7 import validated_report as validated_final_report

    if experiment not in {"v1", "v2", "v3", "v4", "v5", "v6", "v6_1", "v7", "cuda_eval", "cuda_eval2"}:
        raise HTTPException(400, "Unknown experiment")
    return {
        "v1": validated_report,
        "v2": validated_v2_report,
        "v3": validated_v3_report,
        "v4": validated_v4_report,
        "v5": validated_v5_report,
        "v6": validated_v6_report,
        "v6_1": validated_v61_report,
        "v7": validated_final_report,
        "cuda_eval": validated_cuda_report,
        "cuda_eval2": validated_cuda2_report,
    }[experiment](ROOT)


@app.get("/api/figure/{name}")
def figure(name: str, experiment: str = "v1"):
    if name not in {
        "evaluation.png",
        "training.png",
        "families.png",
        "confusion.png",
        "intent.png",
        "consent.png",
        "recovery.png",
    } or experiment not in {
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v6_1",
        "v7",
        "cuda_eval",
        "cuda_eval2",
    }:
        raise HTTPException(404, "Unknown figure")
    from .evidence import validated_report, validated_v2_report, validated_v3_report, validated_v4_report
    from .evidence_cuda import validated_report as validated_cuda_report
    from .evidence_cuda2 import validated_report as validated_cuda2_report
    from .evidence_v5 import validated_report as validated_v5_report
    from .evidence_v6 import validated_report as validated_v6_report
    from .evidence_v6_1 import validated_report as validated_v61_report
    from .evidence_v7 import validated_report as validated_final_report

    evidence = {
        "v1": validated_report,
        "v2": validated_v2_report,
        "v3": validated_v3_report,
        "v4": validated_v4_report,
        "v5": validated_v5_report,
        "v6": validated_v6_report,
        "v6_1": validated_v61_report,
        "v7": validated_final_report,
        "cuda_eval": validated_cuda_report,
        "cuda_eval2": validated_cuda2_report,
    }[experiment](ROOT)
    if not evidence["available"]:
        raise HTTPException(404, "Matching evidence is not ready")
    figure_folder = {"cuda_eval": "cuda-eval", "cuda_eval2": "cuda-eval2"}.get(
        experiment, experiment if experiment != "v1" else ""
    )
    folder = ROOT / "docs/figures" / figure_folder
    if not (folder / name).is_file():
        raise HTTPException(404, "Unknown figure")
    return FileResponse(folder / name)
