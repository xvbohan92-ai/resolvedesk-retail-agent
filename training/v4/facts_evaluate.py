import gc
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.progress import ProgressStore, messages
from resolve_desk.workflow import WorkflowStore
from resolve_desk.workflow import messages as old_messages
from training.common import ROOT, sha
from training.v4.common import REPORTS, verify_frozen
from training.v4.evaluation import promotion, run_case, summarize
from training.v4.facts_provider import FactsProvider as Provider
from training.v4.fixtures import load_cases
from training.v4.recovery import MANIFEST as RECOVERY_MANIFEST
from training.v4.recovery import verify_protocol as verify_recovery


def main():
    manifest = verify_frozen()
    verify_recovery()
    training = json.loads((REPORTS / "training.json").read_text())
    old = json.loads((ROOT / "reports/v3/training.json").read_text())
    config = training["config"]
    assert training["dataset"] == manifest
    for report in (old, training):
        assert (
            sha(ROOT / report["config"]["best_path"] / "adapters.safetensors")
            == report["selected_adapter_sha256"]
        )
    destination, partial = REPORTS / "evaluation.json", REPORTS / "evaluation-partial.json"
    if destination.exists():
        raise ValueError("Completed V4 evaluation exists")
    identity = {
        "training_sha256": sha(REPORTS / "training.json"),
        "recovery_manifest_sha256": sha(RECOVERY_MANIFEST),
        "facts_source_sha256": sha(ROOT / "resolve_desk/decision_facts.py"),
        "previous_training_sha256": sha(ROOT / "reports/v3/training.json"),
        "source_sha256": sha(__file__),
        "runtime_sha256": sha(ROOT / "training/runtime.py"),
        "runner_sha256": sha(ROOT / "training/v4/evaluation.py"),
        "workflow_sha256": sha(ROOT / "resolve_desk/workflow.py"),
    }
    report = {
        **identity,
        "dataset": manifest,
        "config": config,
        "records": [],
        "scope": "Fresh 32-task synthetic v4 test, 16 families and within-split counterfactual pairs. "
        "Four arms isolate the latest memory/guard/policy/fact workflow with previous weights, "
        "and training under that identical latest flow, "
        "and a base-model quality comparator. Same pinned base/tokenizer, greedy unconstrained decoding, "
        "eight total calls, one trial/task. Scripted replies, correlated self-authored tasks; "
        "no official benchmark or production claims.",
    }
    if partial.exists():
        report = json.loads(partial.read_text())
        assert all(report[k] == v for k, v in identity.items()), "Resume identity changed"
    adapters = {
        "previous_old_flow": ROOT / old["config"]["best_path"],
        "previous_new_flow": ROOT / old["config"]["best_path"],
        "base_new_flow": None,
        "qlora_v4": ROOT / config["best_path"],
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
        provider = Provider(model, tokenizer, config, arm, enabled=arm != "previous_old_flow")
        with tempfile.TemporaryDirectory() as tmp:
            new_flow = arm != "previous_old_flow"
            store = (ProgressStore if new_flow else WorkflowStore)(Path(tmp) / "evaluation.sqlite")
            warm = store.create(load_cases("valid")[0])
            provider.generate((messages if new_flow else old_messages)(warm))
            for case in remaining:
                row = {
                    "variant": arm,
                    **run_case(store, case, provider, new_flow=new_flow, max_steps=config["max_agent_steps"]),
                }
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
