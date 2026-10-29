"""Two exact-update candidates; business validation only, before any test inference."""

import gc
import json
import shutil
import tempfile
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT, sha
from training.v7.common import CONFIG, DATA, REPORTS, verify_frozen, verify_local_base
from training.v7.evaluation import run_case, selection_key, summarize
from training.v7.fixtures import load_cases
from training.v7.provider import Provider
from training.v7.resources import wait_for_room


def main():
    import mlx.core as mx
    from mlx_lm import load

    manifest = verify_frozen()
    assert not (REPORTS / "training.json").exists()
    assert not (REPORTS / "evaluation-partial.json").exists()
    config = json.loads(CONFIG.read_text())
    verify_local_base(config)
    updates = json.loads((REPORTS / "updates.json").read_text())
    assert not updates["preflight_only"] and updates["dataset"] == manifest
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    candidates = []
    for candidate in updates["candidates"]:
        path = REPORTS / f"validation-{candidate['iteration']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            assert result["adapter_sha256"] == candidate["adapter_sha256"]
            assert result["manifest_sha256"] == sha(DATA / "manifest.json")
            candidates.append(result)
            continue
        wait_for_room()
        model, tokenizer = load(
            ROOT / config["model"], adapter_path=ROOT / candidate["path"], trust_remote_code=False
        )
        provider = Provider(model, tokenizer, config, f"validation_{candidate['iteration']}")
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ResolutionStore(Path(tmp) / "valid.sqlite")
            for case in load_cases("valid"):
                row = run_case(store, case, provider, messages, config["max_agent_steps"])
                rows.append(row)
                (REPORTS / "validation-partial.json").write_text(
                    json.dumps({**candidate, "records": rows}, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            "candidate": candidate["iteration"],
                            "case": case["id"],
                            "success": row["task_success"],
                        }
                    ),
                    flush=True,
                )
        summary = summarize(rows)
        result = {
            **candidate,
            "manifest_sha256": sha(DATA / "manifest.json"),
            "summary": summary,
            "selection_key": selection_key(summary),
            "records": rows,
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        candidates.append(result)
        del provider, model, tokenizer
        gc.collect()
        mx.clear_cache()
    best = max(candidates, key=lambda c: c["selection_key"])
    destination = ROOT / config["best_path"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["adapters.safetensors", "adapter_config.json"]:
        shutil.copyfile(ROOT / best["path"] / name, destination / name)
    report = {
        **updates,
        "manifest_sha256": sha(DATA / "manifest.json"),
        "status": "trained_and_validation_selected",
        "selected_adapter_sha256": best["adapter_sha256"],
        "best_checkpoint": {k: v for k, v in best.items() if k != "records"},
        "validation_candidates": [{k: v for k, v in c.items() if k != "records"} for c in candidates],
        "updates_sha256": sha(REPORTS / "updates.json"),
        "selected_sampling": {
            "completed_updates": best["iteration"],
            "unique_rows_sampled": len({m["sample_index"] for m in updates["metrics"][: best["iteration"]]}),
            "physical_rows": updates["sampling"]["physical_rows"],
            "bucket_updates": {
                k: sum(m["sampling_bucket"] == k for m in updates["metrics"][: best["iteration"]])
                for k in updates["sampling"]["bucket_rows"]
            },
        },
        "checkpoint_selection": config["checkpoint_selection"],
        "test_used_for_selection": False,
    }
    (REPORTS / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": best["iteration"], "summary": best["summary"]}), flush=True)


if __name__ == "__main__":
    main()
