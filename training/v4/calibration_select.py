"""Select recovery or calibrated weights with predeclared validation guards."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.progress import ProgressStore
from training.common import ROOT, sha
from training.v4.calibration import MANIFEST, verify_protocol
from training.v4.calibration_choice import prefer_calibration
from training.v4.common import REPORTS
from training.v4.evaluation import run_case, summarize
from training.v4.facts_provider import FactsProvider
from training.v4.fixtures import load_cases


def main():
    verify_protocol()
    assert not (REPORTS / "training.json").exists()
    assert not (REPORTS / "evaluation-partial.json").exists()
    baseline = json.loads((REPORTS / "training-recovery-only.json").read_text())
    trial = json.loads((REPORTS / "calibration-updates.json").read_text())
    assert trial["completed_updates"] == 48 and not trial["preflight_only"]
    config = {**baseline["config"], "best_path": trial["protocol"]["best_path"]}
    path = ROOT / trial["protocol"]["output"]
    assert sha(path / "adapters.safetensors") == trial["adapter_sha256"]
    started = time.perf_counter()
    model, tokenizer = load(ROOT / config["model"], adapter_path=path, trust_remote_code=False)
    provider = FactsProvider(model, tokenizer, config, "calibration_validation")
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp) / "validation.sqlite")
        for case in load_cases("valid"):
            records.append(run_case(store, case, provider, max_steps=config["max_agent_steps"]))
            (REPORTS / "calibration-validation-partial.json").write_text(
                json.dumps({"records": records}, indent=2) + "\n"
            )
            print(
                json.dumps({"candidate": 744, "case": case["id"], "success": records[-1]["task_success"]}),
                flush=True,
            )
    result = {
        "completed_updates": 744,
        "path": trial["protocol"]["output"],
        "adapter_sha256": trial["adapter_sha256"],
        "weighted_valid_loss": None,
        "summary": summarize(records),
        "records": records,
        "selection_key": [],
    }
    (REPORTS / "validation-744.json").write_text(json.dumps(result, indent=2) + "\n")
    adopted = prefer_calibration(result["summary"], baseline["best_checkpoint"]["validation_summary"])
    report = {
        **baseline,
        "calibration": trial,
        "calibration_adopted": adopted,
        "calibration_manifest_sha256": sha(MANIFEST),
        "recovery_only_training_sha256": sha(REPORTS / "training-recovery-only.json"),
        "calibration_validation_seconds": round(time.perf_counter() - started, 2),
        "validation_candidates": baseline["validation_candidates"]
        + [{k: v for k, v in result.items() if k != "records"}],
        "checkpoint_selection": "Validation-only recovery696 vs calibration744; frozen regression "
        "guards and fallback/overall/question/call ordering. No test inference used.",
        "test_used_for_selection": False,
    }
    if adopted:
        destination = ROOT / config["best_path"]
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("adapters.safetensors", "adapter_config.json"):
            shutil.copyfile(path / name, destination / name)
        initial = mx.load(str(ROOT / config["adapter_path"] / "initial.safetensors"))
        final = mx.load(str(destination / "adapters.safetensors"))
        delta = (
            sum(
                float(mx.sum((final[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2))
                for k in initial
            )
            ** 0.5
        )
        report.update(
            config=config,
            completed_updates=744,
            calibration_completed_updates=48,
            selected_adapter_sha256=trial["adapter_sha256"],
            adapter_l2_change=delta,
            elapsed_seconds=baseline["elapsed_seconds"] + trial["elapsed_seconds"],
            peak_memory_bytes=max(baseline["peak_memory_bytes"], trial["peak_memory_bytes"]),
            best_checkpoint={
                "iteration": 744,
                "weighted_valid_loss": None,
                "validation_summary": result["summary"],
                "selection_key": [],
            },
        )
    else:
        report["calibration_completed_updates"] = 0
    (REPORTS / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "calibration_adopted": adopted,
                "selected": report["best_checkpoint"]["iteration"],
                "validation": result["summary"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
