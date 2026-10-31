"""Inspect the published partial40 projection without models or private run evidence."""

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from scripts.reconcile_partial40_public_results import ARMS, FIG, SUMMARY, reconcile

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("evaluation", "intent", "recovery", "families", "consent", "training")
CSVS = ("per-case.csv", "case-mapping.csv", "metrics.csv", "training.csv")
ASSETS = {
    **{f"{n}.png": "image/png" for n in FIGURES},
    **{f"{n}.svg": "image/svg+xml" for n in FIGURES},
    **{n: "text/csv; charset=utf-8" for n in CSVS},
}


def checked_projection(root):
    receipt = reconcile(root)
    values = json.loads((root / FIG / "values.json").read_text())
    summary = json.loads((root / SUMMARY).read_text())
    return receipt, values, summary


def create_app(root=ROOT):
    root = Path(root).resolve()
    app = FastAPI(title="ResolveDesk public partial40", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index():
        path = Path(__file__).parent / "static/public_partial40.html"
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"mode": "read_only_public_partial40", "model_routes": False, "original144_complete": False}

    @app.get("/api/report")
    def report():
        try:
            receipt, values, summary = checked_projection(root)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            return {
                "available": False,
                "read_only": True,
                "partial_analysis": True,
                "reason": "公开 CSV、分母或图表哈希未通过一致性核对；未展示分数。",
            }
        return {
            "available": True,
            "read_only": True,
            "partial_analysis": True,
            "original144_complete": False,
            "verification_is_quality_pass": False,
            "verification_scope": "public_csv_and_asset_consistency_only",
            "private_raw_or_tokens_verified_by_this_page": False,
            "analysis_id": summary["analysis_id"],
            "paired_cases": 40,
            "paired_records": 120,
            "arms": list(ARMS),
            "labels": summary["arm_labels"],
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
            "public_reconciliation": receipt,
        }

    @app.get("/api/asset/{name}")
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(404, "Unknown or private artifact")
        try:
            checked_projection(root)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            raise HTTPException(404, "Consistent public exports are unavailable") from error
        return FileResponse(
            root / FIG / name,
            media_type=ASSETS[name],
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local non-privileged TCP port")
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
