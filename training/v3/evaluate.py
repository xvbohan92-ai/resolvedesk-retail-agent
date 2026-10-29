import gc
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.workflow import WorkflowStore, messages
from training.common import ROOT, sha
from training.evaluate import Provider
from training.v3.common import REPORTS, verify_frozen
from training.v3.evaluation import promotion, run_case, summarize
from training.v3.fixtures import load_cases


def main():
    manifest = verify_frozen()
    training = json.loads((REPORTS / "training.json").read_text())
    old = json.loads((ROOT / "reports/v2/training.json").read_text())
    config = training["config"]
    assert training["dataset"] == manifest
    for report in (old, training):
        assert (
            sha(ROOT / report["config"]["best_path"] / "adapters.safetensors")
            == report["selected_adapter_sha256"]
        )
    destination, partial = REPORTS / "evaluation.json", REPORTS / "evaluation-partial.json"
    if destination.exists():
        raise ValueError("Completed V3 evaluation exists")
    identity = {
        "training_sha256": sha(REPORTS / "training.json"),
        "v2_training_sha256": sha(ROOT / "reports/v2/training.json"),
        "source_sha256": sha(__file__),
        "runtime_sha256": sha(ROOT / "training/runtime.py"),
        "runner_sha256": sha(ROOT / "training/v3/evaluation.py"),
        "workflow_sha256": sha(ROOT / "resolve_desk/workflow.py"),
    }
    report = {
        **identity,
        "dataset": manifest,
        "config": config,
        "records": [],
        "scope": "New synthetic v3 test: 48 tasks in 16 families, with counterfactual pairs. "
        "All arms share the unchanged v2 controller/prompt/tools and unconstrained greedy decoding. "
        "Base, v2 adapter and v3 adapter run serially; latency is descriptive. One trial per task. "
        "Scripted field-keyed replies/confirmations; not official tau-bench or retailer data. "
        "Shared author/policy/schema/families; disjoint split IDs/phrasing; no human annotations.",
    }
    if partial.exists():
        report = json.loads(partial.read_text())
        assert all(report[k] == v for k, v in identity.items()), "Resume identity changed"
    adapters = {
        "base": None,
        "qlora_v2": ROOT / old["config"]["best_path"],
        "qlora_v3": ROOT / config["best_path"],
    }
    for arm, adapter in adapters.items():
        remaining = [
            c
            for c in load_cases("test")
            if not any(r["case_id"] == c["id"] and r["variant"] == arm for r in report["records"])
        ]
        if not remaining:
            continue
        model, tokenizer = load(ROOT / config["model"], adapter_path=adapter, trust_remote_code=False)
        provider = Provider(model, tokenizer, config, arm)
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp) / "evaluation.sqlite")
            warm = store.create(load_cases("valid")[0])
            provider.generate(messages(warm))
            for case in remaining:
                row = {"variant": arm, **run_case(store, case, provider, config["max_agent_steps"])}
                report["records"].append(row)
                partial.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "case": case["id"],
                            "success": row["task_success"],
                            "exchange_as_return": row["exchange_as_return"],
                        }
                    ),
                    flush=True,
                )
        del provider, model, tokenizer
        gc.collect()
        mx.clear_cache()
    report["summary"] = {v: summarize([r for r in report["records"] if r["variant"] == v]) for v in adapters}
    report["release_gate"] = promotion(report["records"])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"summary": report["summary"], "release_gate": report["release_gate"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
