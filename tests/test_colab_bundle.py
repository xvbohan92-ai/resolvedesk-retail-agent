"""CPU-only contracts; no model, MLX import, GPU execution, cloud call or fake score report."""

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.colab import build_notebook, bundle, runner


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in bundle.FROZEN_MANIFESTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"files": {}}))
    for name in bundle.REPORT_INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"synthetic_planning_input":true}')
    for name in bundle.ATTRIBUTION_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Retained policy attribution fixture: {name}\n")
    initializer = root / "data/finetuning/retail-qlora-v6_1/best"
    initializer.mkdir(parents=True)
    # Bytes are only integrity fixtures, never loaded as model weights or passed to a model.
    (initializer / "adapters.safetensors").write_bytes(b"integrity-test-only")
    (initializer / "adapter_config.json").write_text('{"rank":8}')
    config = {
        "initialization": str(initializer.relative_to(root)),
        "initialization_sha256": bundle.sha(initializer / "adapters.safetensors"),
        "model": "data/models/qwen3-4b-mlx",
        "adapter_path": "data/finetuning/retail-qlora-v7/run",
        "best_path": "data/finetuning/retail-qlora-v7/best",
        "checkpoint_candidates_completed_updates": [672, 1344],
    }
    (root / "training/v7/config.json").write_text(json.dumps(config))
    (root / "training/v7/train.py").write_text("# exact source fixture\n")
    (root / "training/colab").mkdir()
    (root / "training/colab/runner.py").write_text("# wrapper fixture\n")
    (root / "training/colab/bundle.py").write_text("# verified bootstrap fixture\n")
    return root


def test_source_export_is_allowlisted_and_weights_are_separate(source_tree, tmp_path):
    for name in (
        ".env",
        "user-state.json",
        "reports/v7/evaluation.json",
        "sessions.sqlite",
        "training/v7/local-key.json",
        "resolve_desk/.env",
    ):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DO NOT EXPORT")
    destination = tmp_path / "bundle"
    receipt = bundle.prepare(destination, source_tree, stage_initializer=True)
    assert not receipt["uploaded"] and not receipt["cuda_execution_validated"]
    with zipfile.ZipFile(destination / "resolvedesk-colab-source.zip") as archive:
        names = set(archive.namelist())
        assert "training/v7/train.py" in names
        assert set(bundle.ATTRIBUTION_FILES) <= names
        for name in bundle.ATTRIBUTION_FILES:
            assert archive.read(name) == (source_tree / name).read_bytes()
        assert not any("evaluation.json" in n or ".env" in n or n.endswith(".safetensors") for n in names)
        assert all(bundle.allowed_source(n) for n in names - {bundle.MANIFEST})
    reproduced = tmp_path / "reproduction"
    bundle.safe_extract(destination / "resolvedesk-colab-source.zip", reproduced)
    bundle.import_initializer(destination / "resolvedesk-initializer.zip", reproduced)
    for item in bundle.initializer_spec(source_tree)["files"]:
        assert bundle.sha(reproduced / item["target"]) == item["sha256"]
    with pytest.raises(FileExistsError):
        bundle.safe_extract(destination / "resolvedesk-colab-source.zip", reproduced)


def test_default_export_stages_manifest_not_weights(source_tree, tmp_path):
    destination = tmp_path / "no-weights"
    bundle.prepare(destination, source_tree)
    assert (destination / "initializer-manifest.json").is_file()
    assert not (destination / "resolvedesk-initializer.zip").exists()
    with pytest.raises(ValueError, match="outside"):
        bundle.prepare(source_tree / "new-output", source_tree)


def test_changed_frozen_input_and_symlink_source_are_rejected(source_tree):
    target = "training/v7/train.py"
    manifest = source_tree / bundle.FROZEN_MANIFESTS[-1]
    manifest.write_text(json.dumps({"files": {target: "0" * 64}}))
    with pytest.raises(ValueError, match="frozen"):
        bundle.source_paths(source_tree)
    manifest.write_text('{"files":{}}')
    link = source_tree / "training/colab/leak.py"
    link.symlink_to(source_tree / target)
    with pytest.raises(ValueError, match="Symlink"):
        bundle.source_paths(source_tree)


def test_export_rejects_source_directory_ancestor_link_before_globbing(source_tree, tmp_path):
    directory = source_tree / "training/colab"
    outside = tmp_path / "outside-source"
    shutil.move(directory, outside)
    (outside / "plainmodule.py").write_text("# outside the approved source root")
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        bundle.prepare(tmp_path / "must-not-exist", source_tree)
    assert not (tmp_path / "must-not-exist").exists()


def test_export_rejects_initializer_ancestor_link_even_with_exact_weight_sha(source_tree, tmp_path):
    initializer = source_tree / "data/finetuning/retail-qlora-v6_1/best"
    outside = tmp_path / "outside-weights"
    shutil.move(initializer, outside)
    initializer.symlink_to(outside, target_is_directory=True)
    expected = json.loads((source_tree / "training/v7/config.json").read_text())["initialization_sha256"]
    assert bundle.sha(outside / "adapters.safetensors") == expected
    with pytest.raises(ValueError, match="Symlink"):
        bundle.initializer_spec(source_tree)


@pytest.mark.parametrize(
    "name", ["../escape.py", "/tmp/escape.py", "training/../../escape.py", "training\\escape.py"]
)
def test_archive_traversal_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(name, "do not execute")
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.safe_extract(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_archive_symlink_and_unlisted_secret_rejected(source_tree, tmp_path):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    archive = destination / "resolvedesk-colab-source.zip"
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr(".env", "do not export")
    with pytest.raises(ValueError, match="allowlist"):
        bundle.safe_extract(archive, tmp_path / "unpacked")
    link_archive = tmp_path / "link.zip"
    entry = zipfile.ZipInfo("link.py")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(link_archive, "w") as handle:
        handle.writestr(entry, "/tmp/secret")
    with pytest.raises(ValueError, match="link"):
        bundle.safe_extract(link_archive, tmp_path / "link-unpacked")


def test_export_and_initializer_integrity_fail_closed(source_tree, tmp_path):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree, stage_initializer=True)
    root = tmp_path / "unpacked"
    bundle.safe_extract(destination / "resolvedesk-colab-source.zip", root)
    (root / "training/v7/train.py").write_text("# accidental change")
    with pytest.raises(ValueError, match="changed"):
        bundle.verify_source(root)
    (root / "training/v7/train.py").write_text("# exact source fixture\n")
    bad = tmp_path / "bad-init.zip"
    with (
        zipfile.ZipFile(destination / "resolvedesk-initializer.zip") as source,
        zipfile.ZipFile(bad, "w") as target,
    ):
        for name in source.namelist():
            data = source.read(name)
            target.writestr(name, data[::-1] if name == "adapters.safetensors" else data)
    with pytest.raises(ValueError, match="SHA"):
        bundle.import_initializer(bad, root)


def test_output_parent_symlink_cannot_redirect_training_into_another_checkout(source_tree, tmp_path):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    root = tmp_path / "reproduction"
    bundle.safe_extract(destination / "resolvedesk-colab-source.zip", root)
    (root / "data").symlink_to(source_tree / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        bundle.verify_source(root)


@pytest.mark.parametrize(
    "relative",
    [
        "data/models",
        "data/finetuning/retail-qlora-v7",
        "data/finetuning/retail-qlora-v7/run/candidates/672",
        "colab-execution/preflight.json",
        "reports/v7/updates.json",
        "colab-installed-packages.txt",
    ],
)
def test_nested_artifact_links_fail_before_download_or_gpu_import(source_tree, tmp_path, relative):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    root = tmp_path / "reproduction"
    bundle.safe_extract(destination / "resolvedesk-colab-source.zip", root)
    link = root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "must-not-be-written"
    link.symlink_to(outside, target_is_directory=not relative.endswith((".json", ".txt")))
    assert not outside.exists()  # Includes a dangling stage-receipt leaf link.
    for function in (
        lambda: bundle.verify_source(root),
        lambda: runner.provision_base(root),
        lambda: runner.run_stage(root, "preflight", {}, {}, 8192),
    ):
        with pytest.raises(ValueError, match="Symlink"):
            function()
    assert not outside.exists()


def test_notebook_refuses_original_checkout_before_any_install(source_tree, tmp_path):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    archive = destination / "resolvedesk-colab-source.zip"
    original = tmp_path / "original"
    original.mkdir()
    (original / ".git").write_text("gitdir: original-worktree")
    notebook = json.loads(build_notebook.NOTEBOOK.read_text())
    code = next(c["source"] for c in notebook["cells"] if "archive_path, workspace =" in "".join(c["source"]))
    namespace = {
        "MODE": "setup",
        "BUNDLE_ARCHIVE": str(archive),
        "EXPECTED_SOURCE_SHA256": bundle.sha(archive),
        "WORKSPACE": str(original),
    }
    with pytest.raises(ValueError, match="original checkout"):
        exec("".join(code), namespace)


def test_notebook_checks_existing_code_with_stdlib_before_any_workspace_import(
    source_tree, tmp_path, monkeypatch
):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    archive = destination / "resolvedesk-colab-source.zip"
    root = tmp_path / "reproduction"
    bundle.safe_extract(archive, root)
    sentinel = tmp_path / "tampered-code-executed"
    (root / "training/colab/bundle.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
    )
    notebook = json.loads(build_notebook.NOTEBOOK.read_text())
    code_cells = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    namespace = {
        "MODE": "setup",
        "BUNDLE_ARCHIVE": str(archive),
        "EXPECTED_SOURCE_SHA256": bundle.sha(archive),
        "WORKSPACE": str(root),
    }

    def no_subprocess(*args, **kwargs):
        raise AssertionError("No workspace import, pip installation or GPU command may run")

    monkeypatch.setattr(runner.subprocess, "run", no_subprocess)
    with pytest.raises(ValueError, match="Changed workspace source"):
        for cell in code_cells[1:]:
            exec("".join(cell), namespace)
    assert not sentinel.exists()


def test_notebook_setup_probe_failure_stops_before_pip(source_tree, tmp_path, monkeypatch):
    destination = tmp_path / "export"
    bundle.prepare(destination, source_tree)
    archive = destination / "resolvedesk-colab-source.zip"
    root = tmp_path / "reproduction"
    notebook = json.loads(build_notebook.NOTEBOOK.read_text())
    cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    namespace = {
        "MODE": "setup",
        "BUNDLE_ARCHIVE": str(archive),
        "EXPECTED_SOURCE_SHA256": bundle.sha(archive),
        "WORKSPACE": str(root),
        "MINIMUM_FREE_MIB": 8192,
    }
    calls, gpu_checks = [], []

    def no_gpu(minimum):
        gpu_checks.append(minimum)
        raise RuntimeError("No usable NVIDIA GPU detected; CPU fallback is disabled")

    def fake_child(command, **kwargs):
        calls.append(command)
        assert "pip" not in command, "No package installation is permitted after the GPU failure"
        assert command[1:3] == ["-B", "-c"] and kwargs["timeout"] == 35
        assert "probe_gpu" in command[3] and "probe_cuda_toolkit" in command[3]
        with monkeypatch.context() as child:
            child.chdir(kwargs["cwd"])
            child.setattr(runner.sys, "argv", ["-c", command[4]])
            exec(command[3], {})

    monkeypatch.setattr(runner, "validate_platform", lambda: {"system": "Linux", "python": "3.12"})
    monkeypatch.setattr(runner, "probe_gpu", no_gpu)
    monkeypatch.setattr(runner, "probe_cuda_toolkit", lambda: pytest.fail("Stop on the absent GPU first"))
    monkeypatch.setattr(runner.subprocess, "run", fake_child)
    with pytest.raises(RuntimeError, match="No usable NVIDIA GPU"):
        for cell in cells[1:]:
            exec("".join(cell), namespace)
    assert len(calls) == 1 and gpu_checks == [8192]


def test_absent_nvidia_command_has_clear_no_cpu_fallback_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(runner.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="NVIDIA GPU.*CPU fallback is disabled"):
        runner.probe_gpu()


def test_gpu_probe_requires_eligible_device_and_free_memory():
    row = "Example NVIDIA GPU, 550.54.14, 16384, 12000, 7.5\n"
    assert runner.parse_gpu_rows(row, 1024)["memory_free_mib"] == 12000
    for bad in (row.replace("550.54.14", "535.10"), row.replace("7.5", "7.0")):
        with pytest.raises(RuntimeError):
            runner.parse_gpu_rows(bad, 1024)
    with pytest.raises(MemoryError):
        runner.parse_gpu_rows(row.replace("12000", "512"), 1024)
    with pytest.raises(ValueError):
        runner.parse_gpu_rows(row + row, 1024)


@pytest.mark.parametrize(
    "available,metal,selected", [(False, False, "gpu"), (True, True, "gpu"), (True, False, "cpu")]
)
def test_mlx_backend_check_rejects_cpu_fallback_or_unavailable_gpu(available, metal, selected):
    fake = SimpleNamespace(
        gpu="gpu",
        metal=SimpleNamespace(is_available=lambda: metal),
        is_available=lambda device: available,
        set_default_device=lambda device: None,
        default_device=lambda: SimpleNamespace(type=selected),
    )
    with pytest.raises(RuntimeError, match="GPU"):
        runner.validate_mlx_gpu(fake)


def test_mlx_backend_check_requires_actual_gpu_selection():
    selected = []
    fake = SimpleNamespace(
        gpu="gpu",
        metal=SimpleNamespace(is_available=lambda: False),
        is_available=lambda device: device == "gpu",
        set_default_device=selected.append,
        default_device=lambda: SimpleNamespace(type=selected[-1]),
    )
    runner.validate_mlx_gpu(fake)
    assert selected == ["gpu"]


def test_bounded_probes_have_fixed_commands_no_gpu_mutation(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] == 10 and kwargs["capture_output"]
        return type(
            "Output",
            (),
            {
                "stdout": (
                    "Example NVIDIA GPU, 550.54.14, 16384, 12000, 7.5"
                    if command[0] == "nvidia-smi"
                    else "Cuda compilation tools, release 12.9, V12.9.0"
                )
            },
        )()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.probe_gpu()["device_index"] == 0
    assert runner.probe_cuda_toolkit() == "12.9"
    assert "--id=0" in calls[0][0] and not any("reset" in part for part in calls[0][0])
    assert runner.DEFAULT_MINIMUM_FREE_MIB == 8192
    low = type("Output", (), {"stdout": "Example NVIDIA GPU, 550.54.14, 16384, 7000, 7.5"})()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: low)
    with pytest.raises(MemoryError):
        runner.probe_gpu()


def test_cloud_report_adaptation_preserves_predictions_and_losses():
    report = {
        "cloud_compute_used": False,
        "metrics": [{"loss": 0.7}],
        "completed_updates": 1344,
        "config": {"seed": 123, "max_agent_steps": 12},
    }
    original = json.loads(json.dumps(report))
    wrapped = runner.ReportJSON({"backend": "mlx_cuda12", "not_resume": True, "wrapper_sha256": "proof"})
    saved = json.loads(wrapped.dumps(report))
    assert saved["cloud_compute_used"] is True and saved["execution_metadata"]["not_resume"]
    assert saved["metrics"] == report["metrics"] and saved["config"] == report["config"]
    assert report == original
    action = {"tool": "record_request", "arguments": {"fallback_return": "unknown"}}
    assert json.loads(wrapped.dumps(action)) == action


def test_stage_requires_prior_completed_same_export_and_cpu_initializer_audit(tmp_path):
    (tmp_path / bundle.MANIFEST).write_text("source identity")
    digest = bundle.sha(tmp_path / bundle.MANIFEST)
    reports = tmp_path / "reports/v7"
    reports.mkdir(parents=True)
    receipts = tmp_path / "colab-execution"
    receipts.mkdir()
    (receipts / "preflight.json").write_text(
        json.dumps({"status": "completed", "source_manifest_sha256": digest})
    )
    report = {
        "cloud_compute_used": True,
        "preflight_only": True,
        "execution_metadata": {"not_resume": True, "source_manifest_sha256": digest},
    }
    (reports / "preflight.json").write_text(json.dumps(report))
    audit = {"stages": {"preflight": {"exact_tensor_equality": True}}}
    (reports / "initialization-audit.json").write_text(json.dumps(audit))
    runner.check_stage_inputs(tmp_path, "train")
    audit["stages"]["preflight"]["exact_tensor_equality"] = False
    (reports / "initialization-audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="initializer"):
        runner.check_stage_inputs(tmp_path, "train")
    (receipts / "preflight.json").write_text(
        json.dumps({"status": "failed", "source_manifest_sha256": digest})
    )
    with pytest.raises(ValueError, match="prior CUDA"):
        runner.check_stage_inputs(tmp_path, "train")


def test_default_notebook_compiles_and_runs_without_gpu_or_upload(monkeypatch, capsys):
    validation = build_notebook.validate()
    assert validation["all_code_cells_compiled"] and not validation["cuda_executed"]
    notebook = json.loads(build_notebook.NOTEBOOK.read_text())

    def forbidden(*args, **kwargs):
        raise AssertionError("Default inspect mode must never start subprocess/GPU/install work")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert namespace["MODE"] == "inspect" and not namespace["source_ready"]
    assert "No CUDA results produced" in capsys.readouterr().out
    assert hashlib.sha256(Path(build_notebook.NOTEBOOK).read_bytes()).hexdigest()
