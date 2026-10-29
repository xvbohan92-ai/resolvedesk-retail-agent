"""Small pre-training validation probes, never used for held-out scores."""

import json
import tempfile
from pathlib import Path

from resolve_desk.resolution import ResolutionStore, messages
from training.common import ROOT
from training.v5.common import CONFIG, REPORTS
from training.v5.evaluation import run_case
from training.v5.fixtures import load_cases
from training.v5.resources import wait_for_room


def main():
    import mlx.core as mx
    from mlx_lm import load

    from training.evaluate import Provider

    wait_for_room()
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    config = json.loads(CONFIG.read_text())
    model, tokenizer = load(
        ROOT / config["model"], adapter_path=ROOT / config["initialization"], trust_remote_code=False
    )
    provider = Provider(model, tokenizer, config, "pretraining_development")
    rows = []
    families = {"exchange_fallback", "exchange", "exchange_unavailable", "missing_payment"}
    with tempfile.TemporaryDirectory() as tmp:
        store = ResolutionStore(Path(tmp) / "probe.sqlite")
        for case in load_cases("valid"):
            if case["family"] in families and case["id"].endswith("-00"):
                wait_for_room()
                r = run_case(store, case, provider, messages)
                rows.append(r)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "success": r["task_success"],
                            "first_action": r["trace"][0]["action"],
                        }
                    ),
                    flush=True,
                )
                mx.clear_cache()
    (REPORTS / "development-before-training.json").write_text(
        json.dumps(
            {
                "scope": "Four validation probes before formal freeze/training; not test inference.",
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
