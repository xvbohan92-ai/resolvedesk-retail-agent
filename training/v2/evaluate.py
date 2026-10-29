import gc
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from resolve_desk.protocol import messages as legacy_messages
from resolve_desk.store import Store
from resolve_desk.workflow import WorkflowStore, messages
from training.common import ROOT, sha
from training.evaluate import Provider
from training.v2.common import REPORTS, verify_frozen
from training.v2.evaluation import promotion, run_case, summarize
from training.v2.fixtures import load_cases


def main():
    manifest = verify_frozen()
    training = json.loads((REPORTS / "training.json").read_text())
    config = training["config"]
    assert training["dataset"] == manifest
    assert sha(ROOT / config["best_path"] / "adapters.safetensors") == training["selected_adapter_sha256"]
    path, partial = REPORTS / "evaluation.json", REPORTS / "evaluation-partial.json"
    if path.exists():
        raise ValueError("Completed V2 evaluation exists; preserve this run")
    identity = {
        "training_sha256": sha(REPORTS / "training.json"),
        "source_sha256": sha(__file__),
        "runtime_sha256": sha(ROOT / "training/runtime.py"),
        "runner_sha256": sha(ROOT / "training/v2/evaluation.py"),
        "workflow_sha256": sha(ROOT / "resolve_desk/workflow.py"),
        "legacy_store_sha256": sha(ROOT / "resolve_desk/store.py"),
        "legacy_protocol_sha256": sha(ROOT / "resolve_desk/protocol.py"),
    }
    report = {
        **identity,
        "dataset": manifest,
        "config": config,
        "records": [],
        "scope": "New synthetic v2 test suite, not comparable to v1 task percentages. "
        "Three arms separate engineering (legacy_base to workflow_base) from "
        "training (workflow_base to workflow_qlora_v2). Same base weights/tokenizer, "
        "greedy decoding, unconstrained JSON and eight-call budget. One trial per case. "
        "Base arms interleaved by case; trained arm follows. Serial local latency "
        "is descriptive, not a controlled production-speed comparison. "
        "No official tau-bench/real-retailer score, cloud inference or real transactions.",
    }
    if partial.exists():
        report = json.loads(partial.read_text())
        assert all(report[k] == v for k, v in identity.items()), "Resume identity changed"
    cases = load_cases("test")
    for arms, adapter in [
        (["legacy_base", "workflow_base"], None),
        (["workflow_qlora_v2"], ROOT / config["best_path"]),
    ]:
        todo = {
            (arm, c["id"])
            for arm in arms
            for c in cases
            if not any(r["variant"] == arm and r["case_id"] == c["id"] for r in report["records"])
        }
        if not todo:
            continue
        model, tokenizer = load(ROOT / config["model"], trust_remote_code=False, adapter_path=adapter)
        with tempfile.TemporaryDirectory() as tmp:
            for arm in arms:
                store = (Store if arm == "legacy_base" else WorkflowStore)(Path(tmp) / f"{arm}.sqlite")
                warm = store.create(load_cases("valid")[0])
                Provider(model, tokenizer, config, arm).generate(
                    legacy_messages(warm) if arm == "legacy_base" else messages(warm)
                )
            for case in cases:
                for arm in arms:
                    if (arm, case["id"]) not in todo:
                        continue
                    workflow = arm != "legacy_base"
                    store = (WorkflowStore if workflow else Store)(Path(tmp) / f"{arm}.sqlite")
                    provider = Provider(model, tokenizer, config, arm)
                    row = {
                        "variant": arm,
                        **run_case(store, case, provider, workflow, config["max_agent_steps"]),
                    }
                    report["records"].append(row)
                    partial.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "variant": arm,
                                "case": case["id"],
                                "success": row["task_success"],
                                "calls": row["model_calls"],
                                "duplicates": row["duplicate_read_executions"],
                            }
                        ),
                        flush=True,
                    )
        del model, tokenizer, provider
        gc.collect()
        mx.clear_cache()
    report["summary"] = {
        v: summarize([r for r in report["records"] if r["variant"] == v]) for v in config["evaluation_arms"]
    }
    report["release_gate"] = promotion(report["records"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"summary": report["summary"], "release_gate": report["release_gate"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
