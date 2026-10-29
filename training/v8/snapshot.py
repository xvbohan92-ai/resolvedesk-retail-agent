"""CPU-only bounded raw-evidence inventory; never a model or business verifier."""

# ruff: noqa: E501 -- Keep declarative inventory checks and scope text together.
import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ARMS = ("prior_weights_old_flow", "prior_weights_new_flow", "v8_weights_new_flow")
REPORT_ROOT = "reports/v8"
METADATA = (
    "training/v8/datasets/manifest.json",
    "v8-source-manifest.json",
    "v8-initializer-manifest.json",
)
RUNS = ("smoke-4", "train-1344")
KNOWN_WEIGHTS = {
    f"reports/v8/cuda/{run}/{leaf}": (run, field)
    for run in RUNS
    for leaf, field in (
        ("initial-peft.safetensors", "initial_peft_sha256"),
        ("adapter/adapter_model.safetensors", "final_adapter"),
    )
}
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_WEIGHT_BYTES = 64 * 1024**2
MAX_SNAPSHOT_BYTES = 512 * 1024**2
SECRET_NAMES = {
    ".git",
    "secrets",
    "credentials",
    "credentials.json",
    "secrets.json",
    "token.json",
    "oauth.json",
    "service-account.json",
    "service_account.json",
    "private_key.json",
    "id_rsa",
    "id_ed25519",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _secret(name):
    return any(
        part.casefold().startswith(".env") or part.casefold() in SECRET_NAMES
        for part in PurePosixPath(name).parts
    )


def _read_beneath(workspace, name, limit):
    """Open each component without following symlinks; read the selected file only once."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError("Unsafe snapshot member path")
    descriptors = []
    try:
        directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(directory)
        for part in relative.parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("Not a bounded regular snapshot file")
            raw = source.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("Snapshot member exceeded its byte limit while being read")
            return raw
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _candidates(workspace, excluded):
    root = workspace / REPORT_ROOT
    if root.is_symlink() or (workspace / "reports").is_symlink():
        excluded[REPORT_ROOT] = "symlink report directory"
        return
    if not root.is_dir():
        return
    for directory, subdirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for child in list(subdirs):
            path = base / child
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink() or _secret(relative):
                excluded[relative] = "symlink or secret directory"
                subdirs.remove(child)
        for child in sorted(files):
            path = base / child
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink() or _secret(relative):
                excluded[relative] = "symlink or secret file"
            elif path.suffix in {".json", ".log"} or relative in KNOWN_WEIGHTS:
                yield relative
            else:
                excluded[relative] = "outside report JSON/log and known-weight allowlist"


def _complete_checks(parsed, files, weights, process_return_code):
    checks = {"process_returned_zero": process_return_code == 0}
    required = (
        f"{REPORT_ROOT}/pre-inference-plan.json",
        f"{REPORT_ROOT}/resource-authorization.json",
        f"{REPORT_ROOT}/training-proof.json",
        f"{REPORT_ROOT}/artifact-identity.json",
        *(
            f"{REPORT_ROOT}/cuda/{run}/{leaf}"
            for run in RUNS
            for leaf in (
                "plan.json",
                "report.json",
                "adapter/adapter_config.json",
            )
        ),
        *METADATA,
    )
    for name in required:
        checks["valid_json:" + name] = isinstance(parsed.get(name), dict)
    for name in KNOWN_WEIGHTS:
        checks["verified_weight:" + name] = weights[name]["status"] == "verified"
    for run, kind, updates in (("smoke-4", "smoke", 4), ("train-1344", "train", 1344)):
        row = parsed.get(f"{REPORT_ROOT}/cuda/{run}/report.json", {})
        row = row if isinstance(row, dict) else {}
        checks["completed_run:" + run] = (
            row.get("status") == "completed"
            and row.get("kind") == kind
            and type(row.get("completed_true_optimizer_updates")) is int
            and row["completed_true_optimizer_updates"] == updates
        )
    for split, arms, expected_count in (("validation", (ARMS[-1],), 48), ("evaluation", ARMS, 144)):
        path = f"{REPORT_ROOT}/{split}/evaluation.json"
        row = parsed.get(path)
        records = row.get("records") if isinstance(row, dict) else None
        valid = isinstance(records, list) and isinstance(row, dict) and row.get("status") == "complete"
        valid = valid and len(records) == expected_count and tuple(row.get("arms", ())) == tuple(arms)
        valid = valid and not row.get("pending_case")
        valid = valid and all(
            isinstance(r, dict)
            and r.get("variant") in arms
            and isinstance(r.get("case_id"), str)
            and type(r.get("model_calls")) is int
            and 0 < r["model_calls"] <= 12
            and isinstance(r.get("trace"), list)
            and len(r["trace"]) == r["model_calls"]
            and all(
                isinstance(o, dict) and isinstance(o.get("raw"), str) and isinstance(o.get("timing"), dict)
                for o in r["trace"]
            )
            for r in records
        )
        if valid:
            ids = [r["case_id"] for r in records if r["variant"] == arms[0]]
            valid = len(ids) == len(set(ids)) == 48 and [(r["variant"], r["case_id"]) for r in records] == [
                (arm, cid) for arm in arms for cid in ids
            ]
        checks["complete_records:" + split] = bool(valid)
        journal_path = f"{REPORT_ROOT}/{split}/call-journal.json"
        journal = parsed.get(journal_path)
        journal_valid = bool(valid) and isinstance(journal, list)
        if journal_valid:
            expected = [
                (r["variant"], r["case_id"], index, observation)
                for r in records
                for index, observation in enumerate(r["trace"], 1)
            ]
            journal_valid = len(journal) == len(expected) and all(
                isinstance(actual, dict)
                and actual.get("status") == "completed"
                and (actual.get("variant"), actual.get("case_id"), actual.get("call_index"))
                == (arm, cid, index)
                and actual.get("raw") == observation.get("raw")
                and actual.get("timing") == observation.get("timing")
                for actual, (arm, cid, index, observation) in zip(journal, expected, strict=True)
            )
        checks["complete_journal:" + split] = bool(journal_valid)
        protocol = parsed.get(f"{REPORT_ROOT}/{split}/protocol-audit.json")
        checks["protocol_present:" + split] = (
            isinstance(protocol, dict)
            and isinstance(protocol.get("records"), list)
            and len(protocol["records"]) == expected_count
        )
    bindings = {
        "proof_to_report": (
            f"{REPORT_ROOT}/training-proof.json",
            "training_report_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/report.json",
        ),
        "proof_to_plan": (
            f"{REPORT_ROOT}/training-proof.json",
            "training_plan_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/plan.json",
        ),
        "proof_to_initial": (
            f"{REPORT_ROOT}/training-proof.json",
            "initial_snapshot_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/initial-peft.safetensors",
        ),
        "proof_to_final": (
            f"{REPORT_ROOT}/training-proof.json",
            "final_adapter_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/adapter/adapter_model.safetensors",
        ),
        "identity_to_proof": (
            f"{REPORT_ROOT}/artifact-identity.json",
            "training_proof_sha256",
            f"{REPORT_ROOT}/training-proof.json",
        ),
        "identity_to_analysis": (
            f"{REPORT_ROOT}/artifact-identity.json",
            "analysis_plan_sha256",
            f"{REPORT_ROOT}/pre-inference-plan.json",
        ),
        "identity_to_manifest": (f"{REPORT_ROOT}/artifact-identity.json", "manifest_sha256", METADATA[0]),
        "identity_to_final": (
            f"{REPORT_ROOT}/artifact-identity.json",
            "after_adapter_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/adapter/adapter_model.safetensors",
        ),
        "identity_to_final_config": (
            f"{REPORT_ROOT}/artifact-identity.json",
            "after_adapter_config_sha256",
            f"{REPORT_ROOT}/cuda/train-1344/adapter/adapter_config.json",
        ),
    }
    for label, (source, field, target) in bindings.items():
        row = parsed.get(source)
        checks[label] = isinstance(row, dict) and target in files and row.get(field) == _sha(files[target])
    return checks


def collect_snapshot(workspace, process_return_code):
    """Return ZIP bytes and a raw-file inventory receipt; this executes no cloud/model action."""
    workspace = Path(workspace).absolute()
    if workspace.is_symlink():
        raise ValueError("Snapshot workspace must not be a symlink")
    workspace = workspace.resolve()
    if process_return_code is not None and type(process_return_code) is not int:
        raise TypeError("Process return code must be an integer or None")
    files, parsed, parse_errors, read_errors, excluded = {}, {}, {}, {}, {}
    names = sorted(set(_candidates(workspace, excluded)) | set(METADATA))
    total = 0
    for name in names:
        try:
            raw = _read_beneath(
                workspace, name, MAX_WEIGHT_BYTES if name in KNOWN_WEIGHTS else MAX_MEMBER_BYTES
            )
            if total + len(raw) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot exceeded its bounded aggregate byte limit")
            files[name] = raw
            total += len(raw)
        except (OSError, ValueError) as error:
            read_errors[name] = f"{type(error).__name__}: {error}"
            continue
        if name.endswith(".json"):
            try:
                parsed[name] = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                parse_errors[name] = f"{type(error).__name__}: {error}"
    weights = {}
    for name, (run, field) in KNOWN_WEIGHTS.items():
        report = parsed.get(f"{REPORT_ROOT}/cuda/{run}/report.json", {})
        report = report if isinstance(report, dict) else {}
        expected = report.get(field)
        if field == "final_adapter":
            expected = expected.get("sha256") if isinstance(expected, dict) else None
        actual = _sha(files[name]) if name in files else None
        weights[name] = {
            "status": "missing"
            if actual is None
            else "verified"
            if expected and actual == expected
            else "unverified",
            "expected_sha256": expected,
            "actual_sha256": actual,
            "raw_bytes_preserved": actual is not None,
        }
    progress = {"process_return_code": process_return_code, "files": {}}
    for name, record in parsed.items():
        if (
            name.startswith(REPORT_ROOT + "/")
            and isinstance(record, dict)
            and ("status" in record or "records" in record)
        ):
            progress["files"][name] = {
                "status": record.get("status"),
                "updates": record.get("completed_true_optimizer_updates"),
                "records": len(record.get("records", []))
                if isinstance(record.get("records", []), list)
                else None,
                "pending_case": record.get("pending_case"),
                "attempt_in_progress": record.get("attempt_in_progress"),
                "interruption": record.get("interruption"),
            }
    checks = _complete_checks(parsed, files, weights, process_return_code)
    receipt = {
        "round": "v8",
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "progress": progress,
        "files": {name: _sha(raw) for name, raw in files.items()},
        "parse_errors": parse_errors,
        "read_errors": read_errors,
        "excluded": excluded,
        "weight_validation": weights,
        "artifact_presence_checks": checks,
        "complete_artifacts_present": all(checks.values()),
        "missing_or_incomplete_artifacts": [name for name, present in checks.items() if not present],
        "independent_verification_performed": False,
        "scope": "Single reads of selected report files, four known weight paths and three fixed metadata files. Parse errors and unverified known weights retain raw bytes. Structural presence/hash bindings only; no tensor, tokenizer, business, consent or model-quality verification.",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr("snapshot-receipt.json", json.dumps(receipt, indent=2) + "\n")
    return output.getvalue(), receipt
