"""Conservative local prototype promotion check, added after observing the pilot regression."""

import json

from training.common import ROOT, sha


def decision(summary):
    base, candidate = summary["base"], summary["finetuned"]
    checks = {
        "task_success_not_lower": candidate["task_success"] >= base["task_success"],
        "all_actions_schema_valid": candidate["schema_valid"] == candidate["model_calls"],
        "rejected_writes_not_higher": candidate["rejected_write_attempts"] <= base["rejected_write_attempts"],
    }
    return {"promoted": all(checks.values()), "checks": checks}


def main():
    report = json.loads((ROOT / "reports/evaluation-graded.json").read_text())
    result = {
        **decision(report["summary"]),
        "evaluation_sha256": sha(ROOT / "reports/evaluation-graded.json"),
        "rule_source_sha256": sha(__file__),
        "default_provider": "ollama",
        "scope": "Local prototype rollout guard, implemented after the initial comparison. "
        "Not a preregistered statistical test or evidence of production readiness. "
        "The Ollama path is a separate integration, not established superior by the MLX comparison.",
    }
    (ROOT / "reports/release-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
