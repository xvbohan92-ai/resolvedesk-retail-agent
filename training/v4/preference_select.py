"""Diagnose the fixed final preference checkpoint before any new test inference."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import evaluate
from transformers import AutoTokenizer

from resolve_desk.progress import ProgressStore
from training.common import ROOT, sha
from training.evaluate import Provider
from training.v4.common import REPORTS, read_rows, verify_frozen
from training.v4.evaluation import run_case, summarize
from training.v4.fixtures import load_cases
from training.v4.objective import iterate_batches, weighted_loss
from training.v4.preference import MANIFEST, verify_protocol
from training.v4.tokenization import encode_row


def main():
    manifest = verify_frozen()
    verify_protocol()
    output = REPORTS / "training.json"
    assert not output.exists() and not (REPORTS / "evaluation-partial.json").exists()
    sft = json.loads((REPORTS / "updates.json").read_text())
    preference = json.loads((REPORTS / "preference-updates.json").read_text())
    assert preference["completed_updates"] == 124 and not preference["preflight_only"]
    config = sft["config"]
    adapter = ROOT / preference["protocol"]["output"]
    assert sha(adapter / "adapters.safetensors") == preference["adapter_sha256"]
    started = time.perf_counter()
    hf = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True, trust_remote_code=False)
    data = [encode_row(r, hf, config["tool_name_loss_weight"]) for r in read_rows("valid")]
    model, tokenizer = load(ROOT / config["model"], adapter_path=adapter, trust_remote_code=False)
    loss = evaluate(
        model,
        data,
        1,
        -1,
        max_seq_length=config["max_seq_length"],
        loss=weighted_loss,
        iterate_batches=iterate_batches,
    )
    provider = Provider(model, tokenizer, config, "fixed_preference_final")
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp) / "validation.sqlite")
        for case in load_cases("valid"):
            records.append(run_case(store, case, provider, max_steps=config["max_agent_steps"]))
            print(
                json.dumps(
                    {
                        "candidate": "SFT480+preference124",
                        "case": case["id"],
                        "success": records[-1]["task_success"],
                    }
                ),
                flush=True,
            )
    final = {
        "completed_updates": 604,
        "sft_updates": 480,
        "preference_updates": 124,
        "path": preference["protocol"]["output"],
        "adapter_sha256": preference["adapter_sha256"],
        "weighted_valid_loss": loss,
        "summary": summarize(records),
        "records": records,
        "selection_key": [],
        "selection": "Fixed final checkpoint; diagnostic validation only.",
    }
    (REPORTS / "validation-604.json").write_text(json.dumps(final, indent=2) + "\n")
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
    original240 = json.loads((REPORTS / "validation-240.json").read_text())
    report = {
        **sft,
        "status": "trained",
        "dataset": manifest,
        "completed_updates": 604,
        "sft_completed_updates": 480,
        "preference_completed_updates": 124,
        "preference": preference,
        "preference_manifest_sha256": sha(MANIFEST),
        "elapsed_seconds": sft["elapsed_seconds"] + preference["elapsed_seconds"],
        "sft_elapsed_seconds": sft["elapsed_seconds"],
        "peak_memory_bytes": max(sft["peak_memory_bytes"], preference["peak_memory_bytes"]),
        "best_checkpoint": {
            "iteration": 604,
            "weighted_valid_loss": loss,
            "selection_key": [],
            "validation_summary": final["summary"],
        },
        "selected_adapter_sha256": preference["adapter_sha256"],
        "adapter_l2_change": delta,
        "validation_candidates": [
            {k: v for k, v in r.items() if k != "records"} for r in [original240, final]
        ],
        "validation_seconds": round(time.perf_counter() - started, 2),
        "checkpoint_selection": (
            "Original two-SFT-candidate selection abandoned after validation regressions; "
            "SFT480 is fixed preference initialization. Fixed 124 pair updates chosen before test; "
            "no test inference used for either revision or checkpoint selection."
        ),
        "test_used_for_selection": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixed_checkpoint": 604, "summary": final["summary"]}), flush=True)


if __name__ == "__main__":
    main()
