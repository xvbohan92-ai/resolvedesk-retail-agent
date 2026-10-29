"""Separate eval-only notebook. Its default inspect mode performs no cloud/model work."""

import json
from pathlib import Path

from training.colab.build_notebook import CELLS as ORIGINAL_CELLS
from training.colab.build_notebook import validate

NOTEBOOK = Path(__file__).with_name("resolvedesk_cuda_eval2.ipynb")
BOOTSTRAP = ORIGINAL_CELLS[4][1].replace(
    '{"inspect", "setup", "preflight", "train", "select", "evaluate"}',
    '{"inspect", "setup", "seal", "run", "status", "snapshot"}',
)
CELLS = [
    ("markdown", """# ResolveDesk: independent matched evaluation, round 2

## Goal
Use the existing final **1,344-step** adapter and exact V6.1 predecessor on **48 new authored
synthetic tasks × 2 arms = 96 records**. No training or checkpoint selection is included.
The old runtime ended with 84 verified records and an ambiguous call; those records stay
separate and are never resumed, relabeled or pooled with this new comparison.

## Setup
Only use a **free Colab GPU**. Account quota is currently exhausted; no reset time is known.
Default `inspect` installs nothing, loads no models and sends no requests. Copy the source ZIP
and plan SHA256 from the reviewed local receipts. Upload source, initializer and the exact
completed weights archive separately. Never mount Google Drive or use Run all on the old notebook.
"""),
    ("code", '''MODE = "inspect"  # inspect | setup | seal | run | status | snapshot
FREE_GPU_CONFIRMED = False  # Operator must verify free allocation in Colab UI before setup/run.
BUNDLE_ARCHIVE = "/content/resolvedesk-colab-source.zip"
EXPECTED_SOURCE_SHA256 = ""
EXPECTED_PLAN_SHA256 = ""
INITIALIZER_ARCHIVE = "/content/resolvedesk-initializer.zip"
FINAL_ARCHIVE = "/content/resolvedesk-torch-formal-1344-complete.zip"
WORKSPACE = "/content/resolvedesk-cuda-eval-002"
'''),
    ("markdown", """## Steps
### 1. Verify the source archive before importing its code
The SHA-verified source export contains the frozen new plan and historical interruption
receipts, with synthetic data only. Outputs remain under `reports/cuda-eval2`.
"""),
    ("code", BOOTSTRAP),
    ("markdown", """### 2. Setup and seal the existing weights
`setup` explicitly checks a free GPU, installs the recorded CUDA packages, downloads only
the pinned public base and imports the two weight archives. It performs no model generation.
The local seal already CPU-checked all 112 final F32 tensors. `seal` verifies imported
files against that same sealed identity, without creating another identity.
No mode in this notebook starts training. No CPU inference fallback is available.
"""),
    ("code", '''import os
import subprocess

if source_ready:
    def checked(command, **kwargs):
        return subprocess.run(command, cwd=workspace, check=True, **kwargs)

    plan_file = workspace / "reports/cuda-eval2/analysis-plan.json"
    assert EXPECTED_PLAN_SHA256 and hashlib.sha256(plan_file.read_bytes()).hexdigest() == EXPECTED_PLAN_SHA256
    plan = json.loads(plan_file.read_text())
    assert plan["run_id"] == "resolvedesk-cuda-eval-002"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if MODE == "setup":
        assert FREE_GPU_CONFIRMED, "Confirm free GPU allocation in Colab UI first"
        checked([sys.executable, "-B", "-c",
                 "from training.colab.runner import validate_platform, probe_gpu; "
                 "print(validate_platform()); print(probe_gpu(12288))"], timeout=35)
        checked([sys.executable, "-m", "pip", "install", "--index-url",
                 "https://download.pytorch.org/whl/cu128", "torch==2.11.0+cu128"])
        checked([sys.executable, "-m", "pip", "install", "peft==0.19.1",
                 "accelerate==1.14.0", "transformers==5.21.0", "safetensors==0.8.0",
                 "numpy==2.5.4", "-e", str(workspace)])
        # This unused optional package was incompatible in the original isolated runtime.
        checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
        sys.path.insert(0, str(workspace))
        from training.colab.bundle import import_initializer, verify_source
        from training.colab.runner import provision_base
        verify_source(workspace)
        import_initializer(INITIALIZER_ARCHIVE, workspace)
        final_path = Path(FINAL_ARCHIVE)
        final_sha = "909a5d94662925c820811b359a4a464058579185ab627405ad716d0f749f3715"
        assert hashlib.sha256(final_path.read_bytes()).hexdigest() == final_sha
        members = {"train-1344/" + p for p in ["plan.json", "report.json", "formal-completion.json",
                    "adapter/adapter_model.safetensors", "adapter/adapter_config.json"]}
        members.add("formal-console.log")
        with zipfile.ZipFile(final_path) as final_zip:
            assert set(final_zip.namelist()) == members and len(final_zip.infolist()) == len(members)
            assert sum(e.file_size for e in final_zip.infolist()) < 64 * 1024**2
            for entry in final_zip.infolist():
                assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
                name = "reports/cuda-torch/probe-003/" + entry.filename
                target = workspace / name
                for parent in [target, *target.parents]:
                    assert not parent.is_symlink()
                content = final_zip.read(entry)
                if target.exists():
                    assert target.read_bytes() == content, "Never overwrite different evidence"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        output.write(content)
        provision_base(workspace)
        checked([sys.executable, "-B", "-c",
                 "from training.cuda_torch.train import dependency_precheck; "
                 "print(dependency_precheck())"])
        checked([sys.executable, "-B", "-m", "training.cuda_eval2.run", "preflight"])
    elif MODE == "seal":
        # The reviewed source bundle contains the original local CPU-sealed identity.
        checked([sys.executable, "-B", "-c",
                 "from pathlib import Path; "
                 "from training.cuda_eval2.run import verify_identity; "
                 "root = Path.cwd(); "
                 "print(verify_identity(root / 'reports/cuda-eval2', "
                 "root / 'reports/cuda-torch/probe-003/train-1344/report.json', "
                 "root / 'reports/cuda-torch/probe-003/train-1344/adapter', "
                 "root / 'data/finetuning/retail-qlora-v6_1/best', "
                 "root / 'data/models/qwen3-4b-mlx'))"])
    else:
        print("Setup/sealing skipped for this explicit mode.")
else:
    print("Inspect mode: no package installation, downloads, GPU access or weight loading.")
'''),
    ("markdown", """### 3. Launch once, then inspect or export
`run` requires an existing sealed identity and free GPU confirmation. A persistent launch
receipt prevents repeated launches; do not erase it or replay an ambiguous call. `status`
reads existing reports only. `snapshot` exports retained evidence including unfinished calls.
Keep downloaded snapshots distinct; a completed 96-record report is required for final scores.
"""),
    ("code", '''if source_ready:
    from datetime import UTC, datetime
    output_dir = workspace / "reports/cuda-eval2"
    launch_path = output_dir / "evaluation-launch.json"
    if MODE == "run":
        assert FREE_GPU_CONFIRMED, "Confirm free GPU allocation in Colab UI first"
        assert (output_dir / "artifact-identity.json").is_file()
        checked([sys.executable, "-B", "-c",
                 "from training.cuda_torch.train import dependency_precheck; "
                 "from training.colab.runner import probe_gpu; "
                 "print(dependency_precheck()); print(probe_gpu(12288))"])
        assert not any((output_dir / p).exists() for p in [
            "evaluation-launch.json", "evaluation-partial.json", "evaluation.json", "call-journal.json"])
        command = [sys.executable, "-u", "-B", "-m", "training.cuda_eval2.run", "run", "--free-gpu-confirmed"]
        with launch_path.open("x") as receipt:
            json.dump({"run_id": plan["run_id"], "status": "launch_attempt_recorded",
                       "created_at_utc": datetime.now(UTC).isoformat(),
                       "analysis_plan_sha256": EXPECTED_PLAN_SHA256, "command": command,
                       "automatic_replay_allowed": False}, receipt, indent=2)
        eval_console = (output_dir / "evaluation-console.log").open("x")
        eval_process = subprocess.Popen(command, cwd=workspace, stdout=eval_console,
                                        stderr=subprocess.STDOUT)
        print({"pid": eval_process.pid, "run_id": plan["run_id"], "launched_once": True})
    elif MODE in {"status", "snapshot"}:
        for name in ["evaluation.json", "evaluation-partial.json"]:
            path = output_dir / name
            if path.is_file():
                current = json.loads(path.read_text())
                print({"run_id": current["run_id"], "status": current["status"],
                       "completed_records": len(current["records"]),
                       "pending_case": current.get("pending_case")})
                break
        else:
            print({"run_id": plan["run_id"], "status": "not_started"})
        if "eval_process" in globals():
            print({"process_return_code": eval_process.poll()})
    else:
        print("Evaluation launch skipped.")
else:
    print("Inspect mode: no evaluation launch or model call.")
'''),
    ("code", '''if source_ready and MODE == "snapshot":
    import base64
    import io
    from IPython.display import HTML, display
    keep = {"analysis-plan.json", "artifact-identity.json", "evaluation.json",
            "evaluation-partial.json", "call-journal.json", "protocol-audit.json",
            "evaluation-launch.json", "evaluation-console.log"}
    files = {"reports/cuda-eval2/" + n: (output_dir / n).read_bytes()
             for n in sorted(keep) if (output_dir / n).is_file()}
    receipt = {"run_id": plan["run_id"], "snapshot_at_utc": datetime.now(UTC).isoformat(),
               "files": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("snapshot-receipt.json", json.dumps(receipt, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    display(HTML('<a download="resolvedesk-cuda-eval2-snapshot.zip" href="data:application/zip;base64,'
                 + payload + '">Download independent round-2 snapshot</a>'))
else:
    print("Snapshot export skipped.")
'''),
    ("markdown", """## Checks
Independent local replay must reconcile all 96 records, token IDs, raw first outputs,
failures, required clarifications and scoped-consent actions against the frozen inputs.
The six figure groups are generated only after full verification, not from partial records.

## Next Steps
After the complete snapshot is downloaded and hash-checked, run the documented local
`training.cuda_eval2.verify_results`, `visualize`, verification with `--figures`, then
`write_summary`. No round-2 model results or chart images exist yet because free GPU
allocation is blocked. The old round-1 snapshot remains inspectable and unchanged.
"""),
]


def document():
    cells = []
    for i, (kind, source) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"rd-eval2-{i:02d}", "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "resolvedesk": {"run_id": "resolvedesk-cuda-eval-002", "default_mode": "inspect",
                        "model_inference_executed": False}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(document(), indent=1) + "\n")
    print(json.dumps(validate(NOTEBOOK)))
