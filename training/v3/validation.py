"""Validate exported candidates in a clean process; only the validation split is callable."""

import argparse
import json
import tempfile
from pathlib import Path

from mlx_lm import load

from resolve_desk.workflow import WorkflowStore
from training.common import ROOT, sha
from training.evaluate import Provider
from training.v3.common import REPORTS, verify_frozen
from training.v3.evaluation import run_case, summarize
from training.v3.fixtures import load_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    verify_frozen()
    config = json.loads((ROOT / "training/v3/config.json").read_text())
    output = REPORTS / (
        "preflight-validation.json" if args.preflight else f"validation-{args.iteration}.json"
    )
    if output.exists():
        raise ValueError("Candidate validation already recorded")
    adapter = ROOT / args.adapter
    model, tokenizer = load(ROOT / config["model"], adapter_path=adapter, trust_remote_code=False)
    provider = Provider(model, tokenizer, config, "validation_candidate")
    cases = load_cases("valid")[:1] if args.preflight else load_cases("valid")
    with tempfile.TemporaryDirectory() as tmp:
        store = WorkflowStore(Path(tmp) / "validation.sqlite")
        rows = []
        for case in cases:
            rows.append(run_case(store, case, provider, config["max_agent_steps"]))
            print(
                json.dumps(
                    {
                        "validation_checkpoint": args.iteration,
                        "case": case["id"],
                        "success": rows[-1]["task_success"],
                    }
                ),
                flush=True,
            )
    result = {
        "completed_updates": args.iteration,
        "adapter_sha256": sha(adapter / "adapters.safetensors"),
        "config_sha256": sha(ROOT / "training/v3/config.json"),
        "source_sha256": sha(__file__),
        "preflight_only": args.preflight,
        "summary": summarize(rows),
        "records": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
