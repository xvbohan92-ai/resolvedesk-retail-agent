"""Read-only localhost viewer for the separately authorized, truncated40-case analysis."""

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from training.common import ROOT
from training.v8_eval2.metrics import ARMS, LABELS
from training.v8_eval2.report_results import checked_path

OUTPUT = "docs/figures/v8-eval2-partial40"
REPORTS = "reports/v8-eval2-partial40"
FIGURES = ("evaluation", "intent", "recovery", "families", "consent", "training")
CSVS = ("per-case.csv", "case-mapping.csv", "metrics.csv", "training.csv")
ASSETS = {
    **{f"{name}.png": "image/png" for name in FIGURES},
    **{f"{name}.svg": "image/svg+xml" for name in FIGURES},
    **{name: "text/csv; charset=utf-8" for name in CSVS},
}


def verified_bundle(root):
    """No model, Store replay or receipt writing is exposed by the HTTP server."""
    from training.v8_eval2_partial.report_results import load_verified, verify_delivery

    bundle = load_verified(root)
    expected = verify_delivery(root, write_receipt=False)
    path = checked_path(root, f"{REPORTS}/figure-verification.json")
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("The persisted partial-analysis figure verification is missing or differs")
    values = bundle["values"]
    if values["paired_cases"] != 40 or values["paired_records"] != 120:
        raise ValueError("The approved matched cohort differs")
    return bundle


def public_payload(bundle):
    values = bundle["values"]
    return {
        "available": True,
        "read_only": True,
        "partial_analysis": True,
        "original144_complete": False,
        "verification_is_quality_pass": False,
        "analysis_id": values["analysis_id"],
        "paired_cases": values["paired_cases"],
        "paired_records": values["paired_records"],
        "arms": list(ARMS),
        "labels": LABELS,
        **{
            key: values[key]
            for key in (
                "summary",
                "paired",
                "protocol_summary",
                "failed_records",
                "family_success",
                "training",
                "saved_weights",
                "coverage",
                "scope",
            )
        },
        "figures": list(FIGURES),
        "csv": list(CSVS),
    }


def create_app(root=ROOT):
    root = Path(root).resolve()
    app = FastAPI(title="ResolveDesk partial40 · read only", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index():
        template = Path(__file__).with_suffix(".html")
        if template.is_symlink() or not template.is_file():
            raise HTTPException(404, "Local viewer template is missing")
        return FileResponse(template, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"mode": "read_only_partial40", "model_routes": False, "original144_complete": False}

    @app.get("/api/report")
    def report():
        try:
            return public_payload(verified_bundle(root))
        except (OSError, ValueError, KeyError, AssertionError, TypeError):
            return {
                "available": False,
                "read_only": True,
                "partial_analysis": True,
                "original144_complete": False,
                "reason": "等待40条共同场景的独立本地证据重放与图表核验。原144条评测仍未完成。",
            }

    @app.get("/api/asset/{name}")
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(404, "Unknown or private artifact")
        try:
            verified_bundle(root)
            path = checked_path(root, f"{OUTPUT}/{name}")
        except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
            raise HTTPException(404, "Verified partial-analysis exports are unavailable") from error
        return FileResponse(
            path,
            media_type=ASSETS[name],
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local non-privileged TCP port")
    import uvicorn

    uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
