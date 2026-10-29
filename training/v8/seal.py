"""Seal V8 code, authored data and result-blind protocol once, before any GPU work."""

import ast
import json
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.cuda_eval2.budgeted_run import verify_addendum
from training.cuda_eval2.evaluation import verify_plan as verify_previous_plan
from training.v8.common import CONFIG, DATA, INITIAL_ADAPTER_SHA256, REPORTS


def local_module_files(module):
    parts = module.split(".")
    files = []
    for i in range(1, len(parts) + 1):
        package = ROOT.joinpath(*parts[:i], "__init__.py")
        if package.is_file():
            files.append(package)
    candidate = ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        files.append(candidate)
    return files


def dependency_closure(seeds):
    """AST walk imports inside functions too; external packages are version-bound separately."""
    pending, found = [ROOT / n for n in seeds], set()
    while pending:
        file = pending.pop()
        if file in found or not file.is_file():
            continue
        found.add(file)
        tree = ast.parse(file.read_text())
        package = list(file.relative_to(ROOT).parts[:-1])
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *([node.module] if node.module else [])])
                modules = [module, *(module + "." + item.name for item in node.names)]
            for module in modules:
                if module.split(".")[0] in {"training", "resolve_desk"}:
                    pending.extend(local_module_files(module))
    return {str(file.relative_to(ROOT)) for file in found}


def freeze():
    old = verify_previous_plan()
    verify_addendum()
    path = DATA / "manifest.json"
    if path.exists() or any(
        (REPORTS / n).exists()
        for n in ("cloud-launch.json", "cuda", "validation", "evaluation", "artifact-identity.json")
    ):
        raise FileExistsError("Never replace a frozen or issued V8 round")
    plan_path = REPORTS / "pre-inference-plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("sealed") is not False or plan.get("model_inference_started") is not False:
        raise ValueError("Only a result-blind draft can be frozen")
    if sha(ROOT / "training/v8/test_dataset.json") != plan["dataset"]["sha256"]:
        raise ValueError("The independently authored test changed after review")
    config = json.loads(CONFIG.read_text())
    if config["initialization_sha256"] != INITIAL_ADAPTER_SHA256:
        raise ValueError("Wrong actual predecessor")
    audits = [
        DATA / "reference-audit.json",
        DATA / "token-audit.json",
        DATA / "data-summary.json",
        DATA / "regeneration-audit.json",
        REPORTS / "cross-split-audit.json",
        REPORTS / "gpu-protocol-review.json",
        REPORTS / "colab-package-review.json",
    ]
    if any(not p.is_file() for p in audits):
        raise FileNotFoundError("Complete reference, tokenizer, isolation and execution reviews first")
    ref = json.loads(audits[0].read_text())
    for name, wanted in ref["reference_sources_sha256"].items():
        if sha(ROOT / name) != wanted:
            raise ValueError("Teacher trajectories are from an older source version: " + name)
    if (
        not ref["all_business_references_pass"]
        or not ref["all_references_within_12_total_calls_including_faults"]
    ):
        raise ValueError("Unresolved CPU teacher failure")
    seeds = {str(p.relative_to(ROOT)) for p in (ROOT / "training/v8").glob("*.py")}
    seeds |= {"resolve_desk/resolution_v8.py", "resolve_desk/agent_v8.py"}
    seeds |= {str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*v8*.py")}
    names = dependency_closure(seeds) | set(old["source_inputs_sha256"])
    # Prior manifests bind archived development/report inputs used by inherited pure helpers.
    for prior in (ROOT / "training").glob("v*/datasets/manifest.json"):
        body = json.loads(prior.read_text())
        names.add(str(prior.relative_to(ROOT)))
        names.update(body["files"])
    names |= {str(p.relative_to(ROOT)) for p in DATA.iterdir() if p.suffix in {".json", ".jsonl"}}
    names |= {str(p.relative_to(ROOT)) for p in audits}
    names |= {"training/v8/config.json", "training/v8/test_dataset.json", "training/base-model-spec.json"}
    if any(n.startswith("data/") or n.endswith((".safetensors", ".sqlite")) for n in names):
        raise ValueError("Physical weights/databases do not belong in the source manifest")
    hashes = {name: sha(ROOT / name) for name in sorted(names)}
    plan.update(
        status="frozen",
        sealed=True,
        sealed_at_utc=datetime.now(UTC).isoformat(),
        source_inputs_sha256=hashes,
        seal_blockers=[],
        parent_case_review={
            "all48_initial_requests_and_gold_reviewed": True,
            "no_case_or_label_edits_after_independent_delivery": True,
            "scope": "Authored synthetic English, not external independent annotations",
        },
    )
    plan.pop("source_inputs_sha256_draft", None)
    plan["training_candidate"]["config_sha256"] = sha(CONFIG)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    files = {**hashes, str(plan_path.relative_to(ROOT)): sha(plan_path)}
    manifest = {
        "round": "v8",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
        "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
        "test_inference_started_at_freeze": False,
        "previous_round_commit": "ecedee4",
        "previous_plan_sha256": sha(ROOT / "reports/cuda-eval2/analysis-plan.json"),
        "scope": "One fixed-final continuation; new48 tasks, three arms; no old output pooling or replay",
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "files": len(files),
        "plan_sha256": sha(plan_path),
        "manifest_sha256": sha(path),
        "resource_authorization": "not yet granted for V8; no cloud execution permitted by sealing",
    }


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
