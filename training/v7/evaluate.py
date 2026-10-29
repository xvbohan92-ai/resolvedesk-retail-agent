"""Fresh four-arm test. No checkpoint, prompt or gate is selected from these outputs."""

import gc
import json
import tempfile
from pathlib import Path

from resolve_desk.resolution_v6_1 import ResolutionStore as PreviousStore
from resolve_desk.resolution_v6_1 import messages as previous_messages
from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT, sha
from training.v7.common import CONFIG, DATA, REPORTS, verify_frozen, verify_local_base
from training.v7.evaluation import ARMS, promotion, run_case, summarize
from training.v7.fixtures import load_cases
from training.v7.provider import Provider
from training.v7.resources import wait_for_room


def main():
    import mlx.core as mx
    from mlx_lm import load

    manifest = verify_frozen()
    config = json.loads(CONFIG.read_text())
    verify_local_base(config)
    training = json.loads((REPORTS / "training.json").read_text())
    previous = json.loads((ROOT / "reports/v6_1/training.json").read_text())
    assert training["dataset"] == manifest
    identity = {
        "manifest_sha256": sha(DATA / "manifest.json"),
        "training_sha256": sha(REPORTS / "training.json"),
        "previous_training_sha256": sha(ROOT / "reports/v6_1/training.json"),
        "evaluator_sha256": sha(__file__),
    }
    destination, partial = REPORTS / "evaluation.json", REPORTS / "evaluation-partial.json"
    assert not destination.exists(), "Never overwrite completed held-out evidence"
    report = {
        **identity,
        "config": config,
        "records": [],
        "scope": (
            "Fresh 48-task synthetic test in 48 independent worlds and 24 authored families. "
            "Four arms separate observed-validation/reply recovery from bundled SFT. "
            "Same pinned base, "
            "tokenizer, greedy unconstrained generation, 12 TOTAL calls including record_request. "
            "V6.1 historical tests are development evidence and are not pooled here. "
            "Initial permission slots remain unverified; only a scoped customer YES permits a fallback. "
            "No production or official benchmark claim."
        ),
    }
    if partial.exists():
        report = json.loads(partial.read_text())
        assert all(report[k] == v for k, v in identity.items()), "Resume identity changed"
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    adapters = {
        "previous_old_flow": previous["evaluation_config"]["best_path"],
        "previous_new_flow": previous["evaluation_config"]["best_path"],
        "base_new_flow": None,
        "qlora_v7": config["best_path"],
    }
    for arm in ARMS:
        remaining = [
            c
            for c in load_cases("test")
            if not any(r["variant"] == arm and r["case_id"] == c["id"] for r in report["records"])
        ]
        if not remaining:
            continue
        adapter = adapters[arm]
        if adapter:
            expected = (training if arm == "qlora_v7" else previous)["selected_adapter_sha256"]
            assert sha(ROOT / adapter / "adapters.safetensors") == expected
        wait_for_room()
        model, tokenizer = load(
            ROOT / config["model"], adapter_path=ROOT / adapter if adapter else None, trust_remote_code=False
        )
        old_flow = arm == "previous_old_flow"
        prompt_fn = previous_messages if old_flow else messages
        provider = Provider(model, tokenizer, config, arm, legacy=False)
        with tempfile.TemporaryDirectory() as tmp:
            store = (PreviousStore if old_flow else ResolutionStore)(Path(tmp) / "test.sqlite")
            for case in remaining:
                record = {
                    "variant": arm,
                    **run_case(store, case, provider, prompt_fn, config["max_agent_steps"]),
                }
                report["records"].append(record)
                partial.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "case": case["id"],
                            "success": record["task_success"],
                            "repeated_asks": record["repeated_question_attempts"],
                        }
                    ),
                    flush=True,
                )
        del provider, model, tokenizer
        gc.collect()
        mx.clear_cache()
    report["summary"] = {v: summarize([r for r in report["records"] if r["variant"] == v]) for v in ARMS}
    report["release_gate"] = promotion(report["records"])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "release_gate": report["release_gate"]}), flush=True)


if __name__ == "__main__":
    main()
