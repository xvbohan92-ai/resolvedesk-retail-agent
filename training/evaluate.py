import gc
import json
import statistics
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.protocol import messages
from resolve_desk.store import Store
from training.common import ROOT, sha, verify_frozen
from training.evaluation import run_case
from training.fixtures import load_cases
from training.runtime import generate_action


class Provider:
    def __init__(self, model, tokenizer, config, variant):
        self.model, self.tokenizer, self.config, self.variant = model, tokenizer, config, variant

    def generate(self, messages):
        text, timing = generate_action(self.model, self.tokenizer, messages, self.config)
        return text, {**timing, "provider": "mlx", "variant": self.variant}


def summarize(rows):
    calls = sum(r["model_calls"] for r in rows)
    return {
        "tasks": len(rows),
        "task_success": sum(r["task_success"] for r in rows),
        "proposal_correct": sum(r["proposal_correct"] for r in rows),
        "model_calls": calls,
        "schema_valid": sum(r["schema_valid"] for r in rows),
        "reference_next_action_matches": sum(r["reference_next_action_matches"] for r in rows),
        "rejected_write_attempts": sum(r["rejected_write_attempts"] for r in rows),
        "confirmation_preserved": sum(r["confirmation_preserved"] for r in rows),
        "latency_p50_ms": statistics.median(r["latency_ms"] for r in rows),
        "median_model_calls": statistics.median(r["model_calls"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def main():
    manifest = verify_frozen()
    training = json.loads((ROOT / "reports/training.json").read_text())
    config = training["config"]
    assert training["dataset"] == manifest
    assert sha(ROOT / config["best_path"] / "adapters.safetensors") == training["selected_adapter_sha256"]
    path = ROOT / "reports/evaluation.json"
    if path.exists():
        raise ValueError("Completed evaluation exists; use another experiment")
    partial = ROOT / "reports/evaluation-partial.json"
    identity = {
        "training_sha256": sha(ROOT / "reports/training.json"),
        "source_sha256": sha(__file__),
        "runtime_sha256": sha(ROOT / "training/runtime.py"),
        "runner_sha256": sha(ROOT / "training/evaluation.py"),
        "store_sha256": sha(ROOT / "resolve_desk/store.py"),
        "protocol_sha256": sha(ROOT / "resolve_desk/protocol.py"),
    }
    report = {
        **identity,
        "dataset": manifest,
        "config": config,
        "records": [],
        "scope": "Custom synthetic retail sandbox; not the official tau-bench protocol or leaderboard. "
        "Same fixed MLX base/tokenizer/prompt/tools/greedy decoding for base and QLoRA. "
        "Single trial per task; no confidence or production generalization claim. "
        "Programmatic user answers and confirmation decisions; no real payment/email.",
    }
    if partial.exists():
        previous = json.loads(partial.read_text())
        assert all(previous[k] == v for k, v in identity.items()), "Partial run identity changed"
        report = previous
    cases = load_cases("test")
    for variant in ("base", "finetuned"):
        remaining = [
            c
            for c in cases
            if not any(r["variant"] == variant and r["case_id"] == c["id"] for r in report["records"])
        ]
        if not remaining:
            continue
        model, tokenizer = load(
            ROOT / config["model"],
            trust_remote_code=False,
            adapter_path=ROOT / config["best_path"] if variant == "finetuned" else None,
        )
        provider = Provider(model, tokenizer, config, variant)
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "evaluation.sqlite")
            warm = store.create(load_cases("valid")[0])
            provider.generate(messages(warm))
            for case in remaining:
                row = {"variant": variant, **run_case(store, case, provider, config["max_agent_steps"])}
                report["records"].append(row)
                partial.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "case": case["id"],
                            "success": row["task_success"],
                            "calls": row["model_calls"],
                            "status": row["status_final"],
                        }
                    ),
                    flush=True,
                )
        del provider, model, tokenizer
        gc.collect()
        mx.clear_cache()
    report["summary"] = {
        v: summarize([r for r in report["records"] if r["variant"] == v]) for v in ("base", "finetuned")
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
