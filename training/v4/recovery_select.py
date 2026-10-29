"""Reload fixed action-preference weights; diagnose validation before new test."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.progress import ProgressStore
from training.common import ROOT, sha
from training.v4.common import REPORTS, verify_frozen
from training.v4.evaluation import run_case, summarize
from training.v4.facts_provider import FactsProvider as Provider
from training.v4.fixtures import load_cases
from training.v4.preference import MANIFEST as PREVIOUS_MANIFEST
from training.v4.recovery import MANIFEST, verify_protocol


def main():
    manifest = verify_frozen()
    verify_protocol()
    output = REPORTS / "training.json"
    assert not output.exists() and not (REPORTS / "evaluation-partial.json").exists()
    sft = json.loads((REPORTS / "training-action-only.json").read_text())
    preference = json.loads((REPORTS / "recovery-updates.json").read_text())
    assert preference["completed_updates"] == 80 and not preference["preflight_only"]
    config = {**sft["config"], "best_path": preference["protocol"]["best_path"]}
    adapter = ROOT / preference["protocol"]["output"]
    assert sha(adapter / "adapters.safetensors") == preference["adapter_sha256"]
    started = time.perf_counter()
    model, tokenizer = load(ROOT / config["model"], adapter_path=adapter, trust_remote_code=False)
    provider = Provider(model, tokenizer, config, "fixed_recovery_final")
    records = []
    partial = REPORTS / "recovery-validation-partial.json"
    assert not partial.exists(), "Archive an interrupted diagnostic before restarting"
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp) / "validation.sqlite")
        for case in load_cases("valid"):
            records.append(run_case(store, case, provider, max_steps=config["max_agent_steps"]))
            partial.write_text(
                json.dumps({"adapter_sha256": preference["adapter_sha256"], "records": records}, indent=2)
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "candidate": "SFT480+action136+recovery80",
                        "case": case["id"],
                        "success": records[-1]["task_success"],
                        "last_action": records[-1]["trace"][-1]["action"],
                    }
                ),
                flush=True,
            )
    final = {
        "completed_updates": 696,
        "sft_updates": 480,
        "recovery_updates": 80,
        "path": preference["protocol"]["output"],
        "adapter_sha256": preference["adapter_sha256"],
        "weighted_valid_loss": None,
        "summary": summarize(records),
        "records": records,
        "selection_key": [],
        "selection": "Fixed final checkpoint; diagnostic business validation only.",
    }
    (REPORTS / "validation-696.json").write_text(json.dumps(final, indent=2) + "\n")
    destination = ROOT / config["best_path"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["adapters.safetensors", "adapter_config.json"]:
        shutil.copyfile(adapter / name, destination / name)
    initial = mx.load(str(ROOT / config["adapter_path"] / "initial.safetensors"))
    best = mx.load(str(destination / "adapters.safetensors"))
    delta = (
        sum(float(mx.sum((best[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2)) for k in initial)
        ** 0.5
    )
    diagnostics = [json.loads((REPORTS / f"validation-{n}.json").read_text()) for n in (240, 604, 616)]
    report = {
        **sft,
        "config": config,
        "status": "trained",
        "dataset": manifest,
        "completed_updates": 696,
        "sft_completed_updates": 480,
        "preference_completed_updates": 136,
        "recovery": preference,
        "recovery_completed_updates": 80,
        "preference_manifest_sha256": sha(PREVIOUS_MANIFEST),
        "recovery_manifest_sha256": sha(MANIFEST),
        "action_only_training_sha256": sha(REPORTS / "training-action-only.json"),
        "unregularized_attempt": {
            "updates_sha256": sha(REPORTS / "preference-updates.json"),
            "validation_sha256": sha(REPORTS / "validation-604.json"),
            "training_sha256": sha(REPORTS / "training-unregularized.json"),
            "weights_reused": False,
        },
        "elapsed_seconds": sft["elapsed_seconds"] + preference["elapsed_seconds"],
        "sft_elapsed_seconds": sft["sft_elapsed_seconds"],
        "peak_memory_bytes": max(sft["peak_memory_bytes"], preference["peak_memory_bytes"]),
        "best_checkpoint": {
            "iteration": 696,
            "weighted_valid_loss": None,
            "selection_key": [],
            "validation_summary": final["summary"],
        },
        "selected_adapter_sha256": preference["adapter_sha256"],
        "adapter_l2_change": delta,
        "validation_candidates": [
            {k: v for k, v in r.items() if k != "records"} for r in [*diagnostics, final]
        ],
        "validation_seconds": round(time.perf_counter() - started, 2),
        "checkpoint_selection": (
            "SFT validation and then unregularized preference validation prompted revisions before test. "
            "Fixed SFT480 + action136 + recovery80 with observed affordability facts; "
            "no test-based selection. "
            "Final validation measures business tasks; diagnostic weighted validation loss not recomputed."
        ),
        "test_used_for_selection": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixed_checkpoint": 696, "summary": final["summary"]}), flush=True)


if __name__ == "__main__":
    main()
