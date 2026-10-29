import argparse
import gc
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
from training.v4.evaluation import run_case, selection_key, summarize
from training.v4.fixtures import load_cases
from training.v4.objective import iterate_batches, weighted_loss
from training.v4.tokenization import encode_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = verify_frozen()
    config = json.loads((ROOT / "training/v4/config.json").read_text())
    output = REPORTS / ("preflight-validation.json" if args.preflight else "training.json")
    if output.exists():
        raise ValueError("Selection already completed")
    updates = json.loads((REPORTS / ("preflight.json" if args.preflight else "updates.json")).read_text())
    assert updates["dataset"] == manifest and updates["preflight_only"] == args.preflight
    candidates = (
        [
            {
                "completed_updates": 1,
                "path": "data/finetuning/retail-qlora-v4/preflight",
                "adapter_sha256": updates["final_adapter_sha256"],
            }
        ]
        if args.preflight
        else updates["candidates"]
    )
    hf = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True, trust_remote_code=False)
    data = [
        encode_row(r, hf, config["tool_name_loss_weight"])
        for r in (read_rows("valid")[:1] if args.preflight else read_rows("valid"))
    ]
    started = time.perf_counter()
    validations = []
    for candidate in candidates:
        adapter = ROOT / candidate["path"]
        assert sha(adapter / "adapters.safetensors") == candidate["adapter_sha256"]
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
        provider = Provider(model, tokenizer, config, "validation_candidate")
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "validation.sqlite")
            for case in load_cases("valid")[:1] if args.preflight else load_cases("valid"):
                rows.append(run_case(store, case, provider, max_steps=config["max_agent_steps"]))
                print(
                    json.dumps(
                        {
                            "candidate": candidate["completed_updates"],
                            "case": case["id"],
                            "success": rows[-1]["task_success"],
                        }
                    ),
                    flush=True,
                )
        summary = summarize(rows)
        result = {
            **candidate,
            "weighted_valid_loss": loss,
            "summary": summary,
            "records": rows,
            "selection_key": [] if args.preflight else list(selection_key(summary, loss)),
            "preflight_only": args.preflight,
        }
        validations.append(result)
        if not args.preflight:
            (REPORTS / f"validation-{candidate['completed_updates']}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
        del provider, model, tokenizer
        gc.collect()
        mx.clear_cache()
    if args.preflight:
        output.write_text(json.dumps(validations[0], indent=2) + "\n")
        return
    selected = max(validations, key=lambda r: tuple(r["selection_key"]))
    destination = ROOT / config["best_path"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["adapters.safetensors", "adapter_config.json"]:
        shutil.copyfile(ROOT / selected["path"] / name, destination / name)
    initial = mx.load(str(ROOT / config["adapter_path"] / "initial.safetensors"))
    best = mx.load(str(destination / "adapters.safetensors"))
    delta = (
        sum(float(mx.sum((best[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2)) for k in initial)
        ** 0.5
    )
    report = {
        **updates,
        "status": "trained",
        "best_checkpoint": {
            "iteration": selected["completed_updates"],
            "weighted_valid_loss": selected["weighted_valid_loss"],
            "selection_key": selected["selection_key"],
            "validation_summary": selected["summary"],
        },
        "selected_adapter_sha256": sha(destination / "adapters.safetensors"),
        "adapter_l2_change": delta,
        "validation_candidates": [{k: v for k, v in r.items() if k != "records"} for r in validations],
        "validation_seconds": round(time.perf_counter() - started, 2),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["best_checkpoint"], "adapter_l2_change": delta}), flush=True)


if __name__ == "__main__":
    main()
