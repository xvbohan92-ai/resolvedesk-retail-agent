"""Single-read CPU backups of the independent eval-only run; no model or quality verifier."""

import hashlib
import io
import json
import os
import stat
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

RUN_ID = "resolvedesk-v8-eval-002"
ARMS = ("prior_weights_old_flow", "prior_weights_new_flow", "v8_weights_new_flow")
REPORT_ROOT = "reports/v8-eval2"
METADATA = ("v8-eval2-source-manifest.json", "v8-eval2-weights-manifest.json")
AUXILIARY = (
    "training/v8_eval2/datasets/cases.json",
    "training/v8_eval2/datasets/public.json",
    "training/v8_eval2/datasets/data-summary.json",
    "training/v8_eval2/datasets/reference-audit.json",
    *(f"data/v8-eval2/{arm}/adapter_config.json" for arm in ("before", "after", "legacy")),
)
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_SNAPSHOT_BYTES = 512 * 1024**2
SECRET_NAMES = {
    ".git",
    ".ssh",
    ".aws",
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
    if (
        relative.is_absolute()
        or not relative.parts
        or any(c in name for c in ("\\", ":", "\x00"))
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or str(relative) != name
    ):
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
            elif path.suffix in {".json", ".log"}:
                yield relative
            else:
                excluded[relative] = "outside new-run report JSON/log allowlist"


def _presence(parsed, files, return_code):
    """Structural completeness only; never accepts or repairs a business score."""
    checks = {"process_returned_zero": return_code == 0}
    required = (
        *METADATA,
        *AUXILIARY,
        f"{REPORT_ROOT}/analysis-plan.json",
        f"{REPORT_ROOT}/artifact-identity.json",
        f"{REPORT_ROOT}/resource-authorization.json",
    )
    for name in required:
        checks["valid_json:" + name] = isinstance(parsed.get(name), (dict, list))
    report = parsed.get(f"{REPORT_ROOT}/evaluation.json")
    records = report.get("records") if isinstance(report, dict) else None
    valid = (
        isinstance(records, list)
        and len(records) == 144
        and report.get("status") == "complete"
        and report.get("run_id") == RUN_ID
        and report.get("arms") == list(ARMS)
        and report.get("max_total_calls_per_task") == 12
        and report.get("resume_count") == 0
        and not report.get("pending_case")
        and not report.get("interruption")
    )
    if valid:
        valid = all(
            isinstance(r, dict)
            and r.get("variant") in ARMS
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
        cases = parsed.get("training/v8_eval2/datasets/cases.json")
        cases = cases.get("cases") if isinstance(cases, dict) else cases
        ids = [c.get("id") for c in cases if isinstance(c, dict)] if isinstance(cases, list) else []
        valid = (
            all(isinstance(cid, str) for cid in ids)
            and len(ids) == len(set(ids)) == 48
            and [(r["variant"], r["case_id"]) for r in records] == [(arm, cid) for arm in ARMS for cid in ids]
        )
    checks["complete144_unique_ordered_records"] = bool(valid)
    journal = parsed.get(f"{REPORT_ROOT}/call-journal.json")
    journal_valid = bool(valid) and isinstance(journal, list)
    if journal_valid:
        expected = [(r["variant"], r["case_id"], i, o) for r in records for i, o in enumerate(r["trace"], 1)]
        journal_valid = len(journal) == len(expected) and all(
            isinstance(e, dict)
            and e.get("status") == "completed"
            and (e.get("variant"), e.get("case_id"), e.get("call_index")) == (arm, cid, i)
            and e.get("raw") == o["raw"]
            and e.get("timing") == o["timing"]
            for e, (arm, cid, i, o) in zip(journal, expected, strict=True)
        )
    checks["all_journal_calls_attached_and_completed"] = bool(journal_valid)
    protocol = parsed.get(f"{REPORT_ROOT}/protocol-audit.json")
    checks["protocol144_present"] = (
        isinstance(protocol, dict)
        and protocol.get("run_id") == RUN_ID
        and isinstance(protocol.get("records"), list)
        and len(protocol["records"]) == 144
        and all(isinstance(p, dict) for p in protocol["records"])
        and valid
        and [(p.get("variant"), p.get("case_id")) for p in protocol["records"]]
        == [(r["variant"], r["case_id"]) for r in records]
    )
    for label, source, field, target in (
        ("report_plan", "evaluation.json", "analysis_plan_sha256", "analysis-plan.json"),
        ("report_identity", "evaluation.json", "artifact_identity_sha256", "artifact-identity.json"),
        ("report_journal", "evaluation.json", "call_journal_sha256", "call-journal.json"),
        ("report_protocol", "evaluation.json", "protocol_audit_sha256", "protocol-audit.json"),
    ):
        row = parsed.get(f"{REPORT_ROOT}/{source}")
        name = f"{REPORT_ROOT}/{target}"
        checks[label] = isinstance(row, dict) and name in files and row.get(field) == _sha(files[name])
    return checks


def collect_snapshot(workspace, process_return_code):
    """Return ZIP bytes/receipt; preserve malformed JSON and ambiguous calls, never replay."""
    workspace = Path(workspace).absolute()
    if workspace.is_symlink():
        raise ValueError("Snapshot workspace must not be a symlink")
    workspace = workspace.resolve()
    if process_return_code is not None and type(process_return_code) is not int:
        raise TypeError("Process return code must be an integer or None")
    files, parsed, parse_errors, read_errors, excluded = {}, {}, {}, {}, {}
    names = sorted(set(_candidates(workspace, excluded)) | set(METADATA) | set(AUXILIARY))
    total = 0
    for name in names:
        try:
            raw = _read_beneath(workspace, name, MAX_MEMBER_BYTES)
            if total + len(raw) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot exceeded its aggregate byte limit")
            files[name], total = raw, total + len(raw)
        except (OSError, ValueError) as error:
            read_errors[name] = f"{type(error).__name__}: {error}"
            continue
        if name.endswith(".json"):
            try:

                def reject_nonfinite(value):
                    raise ValueError("Nonfinite JSON constant: " + value)

                parsed[name] = json.loads(raw, parse_constant=reject_nonfinite)
            except (ValueError, UnicodeError) as error:
                parse_errors[name] = f"{type(error).__name__}: {error}"
    progress = {"process_return_code": process_return_code, "files": {}}
    for name, row in parsed.items():
        if (
            name.startswith(REPORT_ROOT + "/")
            and isinstance(row, dict)
            and ("status" in row or "records" in row)
        ):
            records = row.get("records")
            progress["files"][name] = {
                "status": row.get("status"),
                "records": len(records) if isinstance(records, list) else None,
                "records_by_arm": dict(
                    Counter(
                        r.get("variant") if isinstance(r.get("variant"), str) else "<invalid>"
                        for r in records
                        if isinstance(r, dict)
                    )
                )
                if isinstance(records, list)
                else {},
                "pending_case": row.get("pending_case"),
                "interruption": row.get("interruption"),
            }
    journal = parsed.get(f"{REPORT_ROOT}/call-journal.json")
    progress["journal"] = (
        {
            "entries": len(journal),
            "statuses": dict(
                Counter(
                    e.get("status") if isinstance(e.get("status"), str) else "<invalid>"
                    for e in journal
                    if isinstance(e, dict)
                )
            ),
        }
        if isinstance(journal, list)
        else None
    )
    checks = _presence(parsed, files, process_return_code)
    receipt = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "progress": progress,
        "files": {name: _sha(raw) for name, raw in files.items()},
        "parse_errors": parse_errors,
        "read_errors": read_errors,
        "excluded": excluded,
        "artifact_presence_checks": checks,
        "complete_artifacts_present": all(checks.values()),
        "missing_or_incomplete_artifacts": [k for k, present in checks.items() if not present],
        "independent_verification_performed": False,
        "physical_weights_copied_or_reverified": False,
        "automatic_resume_authorized": False,
        "scope": (
            "Single reads of new-run JSON/log files and explicit metadata/configuration files. "
            "Malformed JSON retains raw bytes. Structural presence and captured hashes only; "
            "no tensor, tokenizer, business, consent or quality verification. Saved adapters "
            "remain in the separately verified weight package; base, DB, credentials and "
            "weights are excluded. A snapshot does not authorize replay of any issued call."
        ),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr("snapshot-receipt.json", json.dumps(receipt, indent=2) + "\n")
    return output.getvalue(), receipt
