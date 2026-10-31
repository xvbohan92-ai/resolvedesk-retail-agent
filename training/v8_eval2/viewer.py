"""Local, read-only delivery viewer; no provider, database or partial-score endpoint."""

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from training.common import ROOT
from training.v8_eval2.metrics import ARMS, LABELS, RUN_ID
from training.v8_eval2.report_results import (
    CSV_NAMES,
    FIGURES,
    OUTPUT,
    checked_path,
    load_verified,
    verify_delivery,
)
from training.v8_eval2.terminal_status import OUTPUT as STATUS_OUTPUT
from training.v8_eval2.terminal_status import (
    STATUS_ASSETS,
    progress_rows,
    read_terminal,
    verify_status_delivery,
)
from training.v8_eval2.terminal_status import SUMMARY as TERMINAL_SUMMARY

ASSETS = {
    **{f"{name}.png": "image/png" for name in FIGURES},
    **{f"{name}.svg": "image/svg+xml" for name in FIGURES},
    **{name: "text/csv; charset=utf-8" for name in CSV_NAMES},
}
PENDING = (
    "等待完整 144 条新评测、独立 CPU 重放及六组图表/CSV 核验。"
    "此页面不展示部分结果或推算分数，也不会发起模型调用。"
)


def verified_bundle(root):
    """Reuse the actual full-result/asset gates without writing any verification receipt."""
    bundle = load_verified(root)
    expected = verify_delivery(root, write_receipt=False)
    receipt = checked_path(root, "reports/v8-eval2/figure-verification.json")
    if not receipt.is_file() or json.loads(receipt.read_text()) != expected:
        raise ValueError("Persisted six-group verification is missing or differs")
    return bundle, expected


def payload(bundle, verification):
    """Aggregate and complete failures only; never expose prompts, auth, weights or raw snapshots."""
    values = bundle["values"]
    return {
        "available": True,
        "run_id": RUN_ID,
        "read_only": True,
        "arms": list(ARMS),
        "labels": LABELS,
        "summary": values["summary"],
        "paired": values["paired"],
        "protocol_summary": values["protocol_summary"],
        "failed_records": values["failed_records"],
        "family_success": values["family_success"],
        "training": values["training"],
        "saved_weights": values["saved_weights"],
        "scope": values["scope"],
        "figures": list(FIGURES),
        "csv": list(CSV_NAMES),
        "counts": verification["counts"],
        "visual_qa_performed": verification["visual_qa_performed"],
        "verification_is_quality_pass": False,
        "report_only": True,
        "deployment_or_default_provider_change_allowed": False,
        "historical_boundary": {
            "old_completed_records_excluded": 34,
            "old_completed_journal_calls_excluded": 223,
            "old_pending_task_responses_excluded": 11,
            "existing_validation_rerun": False,
            "additional_training_performed": False,
        },
    }


def create_app(root=ROOT):
    root = Path(root).resolve()
    app = FastAPI(
        title="ResolveDesk V8 evaluation · read only", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.get("/")
    def index():
        template = Path(__file__).with_suffix(".html")
        if template.is_symlink() or not template.is_file():
            raise HTTPException(404, "Local viewer template is missing")
        return FileResponse(template, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"run_id": RUN_ID, "mode": "read_only_saved_evidence", "provider_routes": False}

    @app.get("/api/report")
    def report():
        if (root / TERMINAL_SUMMARY).exists() or (root / TERMINAL_SUMMARY).is_symlink():
            try:
                terminal = read_terminal(root)
                public_fields = (
                    "status",
                    "process_return_code",
                    "process_stopped_at_utc",
                    "completed_test_records",
                    "planned_test_records",
                    "missing_completed_records",
                    "unstarted_task_count",
                    "ambiguous_pending_task_count",
                    "model_calls_issued",
                    "completed_call_entries",
                    "started_response_unknown_call_entries",
                    "training_updates_this_run",
                    "saved_v8_training_updates",
                    "gpu_runtime_released",
                    "performance_conclusion",
                )
                try:
                    verify_status_delivery(root)
                    status_assets_ready = True
                except (OSError, ValueError, KeyError, AssertionError, TypeError):
                    status_assets_ready = False
                return {
                    "available": False,
                    "run_id": RUN_ID,
                    "read_only": True,
                    "terminal": {**{k: terminal[k] for k in public_fields}, "rows": progress_rows(terminal)},
                    "status_assets_ready": status_assets_ready,
                    "reason": "评测已于固定截止时间终止，记录未完整。完成进度不是模型成功率。",
                }
            except (OSError, ValueError, KeyError, AssertionError, TypeError):
                return {
                    "available": False,
                    "run_id": RUN_ID,
                    "read_only": True,
                    "terminal_integrity_error": True,
                    "reason": "终态摘要未通过完整性检查；不展示进度、效果指标或图表。",
                }
        try:
            bundle, verification = verified_bundle(root)
            return payload(bundle, verification)
        except (OSError, ValueError, KeyError, AssertionError, TypeError):
            return {"available": False, "run_id": RUN_ID, "read_only": True, "reason": PENDING}

    @app.get("/api/asset/{name}")
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(404, "Unknown or private artifact")
        if (root / TERMINAL_SUMMARY).exists() or (root / TERMINAL_SUMMARY).is_symlink():
            raise HTTPException(404, "Terminal incomplete run has no final performance assets")
        try:
            verified_bundle(root)
            path = checked_path(root, f"{OUTPUT}/{name}")
        except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
            raise HTTPException(404, "Complete verified evidence is not available") from error
        return FileResponse(
            path,
            media_type=ASSETS[name],
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            },
        )

    @app.get("/api/status-asset/{name}")
    def status_asset(name: str):
        if name not in STATUS_ASSETS:
            raise HTTPException(404, "Unknown or private terminal artifact")
        try:
            verify_status_delivery(root)
            path = checked_path(root, f"{STATUS_OUTPUT}/{name}")
        except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
            raise HTTPException(404, "Reviewed terminal progress export is unavailable") from error
        return FileResponse(path, media_type=STATUS_ASSETS[name], headers={"Cache-Control": "no-store"})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local non-privileged TCP port")
    import uvicorn

    uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
