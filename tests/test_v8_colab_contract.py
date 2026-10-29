"""CPU toy inventory/AST contracts only; no Colab cell, GPU or cloud action executes."""

import ast
import hashlib
import io
import json
import zipfile

import pytest

from training.v8 import colab_cells, snapshot
from training.v8.bundle import make_notebook, safe_member, zip_files


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()
    path.write_bytes(raw)
    return digest(raw)


def unpack(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.fixture
def complete_inventory(tmp_path):
    """Fake bytes/metadata exercise presence, never tensor or model correctness."""
    root = tmp_path / "workspace"
    base = "reports/v8/"
    analysis = put(root, base + "pre-inference-plan.json", {"status": "frozen"})
    put(root, base + "resource-authorization.json", {"round": "v8", "explicit_human_approval": True})
    metadata = {
        name: put(root, name, {"files": {"synthetic-source.py": "0" * 64}}) for name in snapshot.METADATA
    }
    run_hashes = {}
    for run, kind, updates in (("smoke-4", "smoke", 4), ("train-1344", "train", 1344)):
        folder = base + f"cuda/{run}/"
        initial = put(root, folder + "initial-peft.safetensors", b"synthetic initial CPU fixture")
        final = put(root, folder + "adapter/adapter_model.safetensors", b"synthetic final CPU fixture")
        config = put(root, folder + "adapter/adapter_config.json", {"r": 8})
        plan = put(root, folder + "plan.json", {"kind": kind})
        report = put(
            root,
            folder + "report.json",
            {
                "status": "completed",
                "kind": kind,
                "completed_true_optimizer_updates": updates,
                "initial_peft_sha256": initial,
                "final_adapter": {"sha256": final},
            },
        )
        run_hashes[run] = {
            "initial": initial,
            "final": final,
            "config": config,
            "plan": plan,
            "report": report,
        }
    final = run_hashes["train-1344"]
    proof = put(
        root,
        base + "training-proof.json",
        {
            "training_report_sha256": final["report"],
            "training_plan_sha256": final["plan"],
            "initial_snapshot_sha256": final["initial"],
            "final_adapter_sha256": final["final"],
        },
    )
    put(
        root,
        base + "artifact-identity.json",
        {
            "training_proof_sha256": proof,
            "analysis_plan_sha256": analysis,
            "manifest_sha256": metadata[snapshot.METADATA[0]],
            "after_adapter_sha256": final["final"],
            "after_adapter_config_sha256": final["config"],
        },
    )
    for split, arms in (("validation", snapshot.ARMS[-1:]), ("evaluation", snapshot.ARMS)):
        records, journal = [], []
        for arm in arms:
            for n in range(48):
                cid = f"CPU-only-{split}-{n:02d}"
                observation = {
                    "raw": "synthetic output; no model called",
                    "timing": {"provider": "cpu_fixture_only"},
                }
                records.append(
                    {
                        "variant": arm,
                        "case_id": cid,
                        "task_success": False,
                        "model_calls": 1,
                        "trace": [observation],
                    }
                )
                journal.append(
                    {"variant": arm, "case_id": cid, "call_index": 1, "status": "completed", **observation}
                )
        put(
            root,
            base + split + "/evaluation.json",
            {"status": "complete", "arms": list(arms), "records": records},
        )
        put(root, base + split + "/call-journal.json", journal)
        put(root, base + split + "/protocol-audit.json", {"records": [{} for _ in records]})
    return root


def test_notebook_is_inspect_only_and_all_cells_parse_without_execution():
    notebook = make_notebook(
        {"resolvedesk-v8-source.zip": "a" * 64, "resolvedesk-v8-initializers.zip": "b" * 64}, "c" * 64
    )
    codes = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(codes) == 4 and all(
        cell["outputs"] == [] and cell["execution_count"] is None for cell in codes
    )
    trees = [ast.parse("".join(cell["source"])) for cell in codes]
    assignments = [n for n in ast.walk(trees[0]) if isinstance(n, ast.Assign)]
    assert any(
        any(isinstance(t, ast.Name) and t.id == "OPERATOR_EXECUTE_V8" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is False
        for n in assignments
    )
    assert all("PLACEHOLDER" not in "".join(cell["source"]) for cell in codes)
    assert (
        "runtime.unassign" in colab_cells.CONFIG_CODE
        and "v8_release_timer.is_alive()" in colab_cells.LAUNCH_CODE
    )
    assert "files.upload" not in colab_cells.STATUS_CODE and "subprocess.Popen" not in colab_cells.STATUS_CODE
    assert "drive.mount" not in "\n".join("".join(cell["source"]) for cell in codes)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.py",
        "/etc/passwd",
        "x\\evil.py",
        "data/model.safetensors",
        "nested/.env",
        "nested/.env.local",
        "x.db",
        "x.sqlite3",
        "nested/key.pem",
    ],
)
def test_source_path_filter_rejects_escape_secret_and_database_containers(name):
    assert not safe_member(name)


def test_source_zip_is_manifest_bound_and_never_overwrites(tmp_path):
    path = tmp_path / "source.zip"
    manifest = zip_files(path, {"training/v8/example.py": b"# CPU fixture\n"}, "manifest.json")
    files = unpack(path.read_bytes())
    assert json.loads(files["manifest.json"])["files"] == manifest["files"]
    assert digest(files["training/v8/example.py"]) == manifest["files"]["training/v8/example.py"]
    with pytest.raises(FileExistsError):
        zip_files(path, {}, "manifest.json")


def test_partial_json_is_preserved_and_each_raw_file_is_read_once(tmp_path, monkeypatch):
    bad = b'{"status":"training","metrics":['
    name = "reports/v8/cuda/train-1344/report.json"
    put(tmp_path, name, bad)
    put(tmp_path, "reports/v8/failure.log", b"CPU failure prefix retained\n")
    reads = []
    original = snapshot._read_beneath

    def observe(root, relative, limit):
        reads.append(relative)
        return original(root, relative, limit)

    monkeypatch.setattr(snapshot, "_read_beneath", observe)
    raw, receipt = snapshot.collect_snapshot(tmp_path, None)
    files = unpack(raw)
    assert files[name] == bad and name in receipt["parse_errors"]
    assert reads.count(name) == 1 and len(reads) == len(set(reads))
    assert receipt["complete_artifacts_present"] is False
    assert snapshot.METADATA[0] in receipt["read_errors"]
    assert "reports/v8/failure.log" in files


def test_complete_inventory_keeps_all_four_weights_journals_and_false_outcomes(complete_inventory):
    raw, receipt = snapshot.collect_snapshot(complete_inventory, 0)
    files = unpack(raw)
    assert receipt["complete_artifacts_present"] is True
    assert receipt["independent_verification_performed"] is False
    assert all(receipt["weight_validation"][name]["status"] == "verified" for name in snapshot.KNOWN_WEIGHTS)
    assert all(name in files for name in (*snapshot.KNOWN_WEIGHTS, *snapshot.METADATA))
    assert len(json.loads(files["reports/v8/evaluation/evaluation.json"])["records"]) == 144
    assert all(
        not r["task_success"] for r in json.loads(files["reports/v8/evaluation/evaluation.json"])["records"]
    )
    assert len(json.loads(files["reports/v8/validation/call-journal.json"])) == 48
    assert len(json.loads(files["reports/v8/evaluation/call-journal.json"])) == 144
    assert all(digest(files[name]) == wanted for name, wanted in receipt["files"].items())


@pytest.mark.parametrize(
    "leaf,missing",
    [
        ("initial-peft.safetensors", False),
        ("adapter/adapter_model.safetensors", False),
        ("initial-peft.safetensors", True),
        ("adapter/adapter_model.safetensors", True),
    ],
)
def test_completed_training_without_matching_weights_cannot_look_complete(complete_inventory, leaf, missing):
    name = "reports/v8/cuda/train-1344/" + leaf
    if missing:
        (complete_inventory / name).unlink()
    else:
        (complete_inventory / name).write_bytes(b"corrupt synthetic weight prefix")
    raw, receipt = snapshot.collect_snapshot(complete_inventory, 0)
    assert receipt["complete_artifacts_present"] is False
    assert receipt["weight_validation"][name]["status"] == ("missing" if missing else "unverified")
    assert (name in unpack(raw)) is not missing


def test_ambiguous_journal_and_duplicate_task_are_retained_but_not_complete(complete_inventory):
    name = "reports/v8/evaluation/call-journal.json"
    rows = json.loads((complete_inventory / name).read_text())
    rows[-1]["status"] = "started"
    put(complete_inventory, name, rows)
    raw, receipt = snapshot.collect_snapshot(complete_inventory, 0)
    assert receipt["complete_artifacts_present"] is False
    assert json.loads(unpack(raw)[name])[-1]["status"] == "started"
    report_path = "reports/v8/evaluation/evaluation.json"
    report = json.loads((complete_inventory / report_path).read_text())
    report["records"][-1] = report["records"][-2]
    put(complete_inventory, report_path, report)
    _, receipt = snapshot.collect_snapshot(complete_inventory, 0)
    assert receipt["artifact_presence_checks"]["complete_records:evaluation"] is False


def test_base_database_secret_unknown_weights_and_symlinks_are_not_archived(complete_inventory, tmp_path):
    root = complete_inventory
    for name in (
        "data/models/base.safetensors",
        "reports/v8/x.db",
        "reports/v8/x.sqlite",
        "reports/v8/.env",
        "reports/v8/nested/credentials.json",
        "reports/v8/unknown.safetensors",
    ):
        put(root, name, b"SECRET_SENTINEL_MUST_NOT_BE_READ")
    outside = tmp_path / "outside-secret.json"
    outside.write_bytes(b"SECRET_SENTINEL_MUST_NOT_BE_READ")
    (root / "reports/v8/linked.json").symlink_to(outside)
    (root / "reports/v8/linked-dir").symlink_to(tmp_path, target_is_directory=True)
    raw, receipt = snapshot.collect_snapshot(root, 0)
    files = unpack(raw)
    assert not any(b"SECRET_SENTINEL_MUST_NOT_BE_READ" in content for content in files.values())
    assert "reports/v8/linked.json" in receipt["excluded"]
    assert "reports/v8/linked-dir" in receipt["excluded"]
    assert not any(n.startswith("data/") or n.endswith((".db", ".sqlite")) for n in files)


def test_running_or_failed_process_never_claims_final_completion(complete_inventory):
    for code in (None, 124):
        _, receipt = snapshot.collect_snapshot(complete_inventory, code)
        assert receipt["complete_artifacts_present"] is False
    with pytest.raises(TypeError):
        snapshot.collect_snapshot(complete_inventory, False)
