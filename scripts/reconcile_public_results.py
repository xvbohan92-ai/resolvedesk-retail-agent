"""Reconcile the published scientific projection, without private authorization files.

This does not reproduce tokenizer decoding, reconstruct prompts, read model weights,
or recertify the original private execution/authorization hash chain. It checks
public records, token-array integrity, completed calls and exported chart values.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("predecessor_fp16", "cloud1344_fp16")
REPORT = "reports/cuda-eval2/"
FIGURES = "docs/figures/cuda-eval2/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class PublicEvidence:
    def __init__(self, root):
        self.root = Path(root)
        self.archives = json.loads((self.root / "evidence-archives.json").read_text())
        self.files = {}

    def read(self, name):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe public path")
        path = self.root / relative
        if name in self.archives:
            entry = self.archives[name]
            physical = self.root / entry["archive"]
            require(
                physical.is_file() and not physical.is_symlink(), "Missing regular public archive: " + name
            )
            packed = physical.read_bytes()
            require(digest(packed) == entry["archive_sha256"], "Public gzip SHA differs: " + name)
            raw = gzip.decompress(packed)
            require(
                len(raw) == entry["size"] and digest(raw) == entry["sha256"],
                "Public unpacked SHA differs: " + name,
            )
            if path.exists():
                require(
                    not path.is_symlink() and path.read_bytes() == raw,
                    "Restored public evidence is stale: " + name,
                )
            physical_name = entry["archive"]
        else:
            require(path.is_file() and not path.is_symlink(), "Missing regular public evidence: " + name)
            packed = raw = path.read_bytes()
            physical_name = name
        self.files[name] = {
            "sha256": digest(raw),
            "bytes": len(raw),
            "physical_path": physical_name,
            "physical_sha256": digest(packed),
        }
        return raw

    def json(self, name):
        return json.loads(self.read(name))


def token_record(timing):
    for field, count in [("input_token_ids", "input_tokens"), ("generated_token_ids", "output_tokens")]:
        ids = timing[field]
        require(
            isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), "Missing/corrupt token IDs"
        )
        require(timing[count] == len(ids), "Token count differs")
        encoded = json.dumps(ids, separators=(",", ":")).encode()
        require(timing[field + "_sha256"] == digest(encoded), "Token-array SHA differs")
    require(timing["input_tokens"] > 0 and 0 <= timing["output_tokens"] <= 512, "Token budget differs")


def reconcile(root=ROOT):
    # These are pure public metric/scoring functions; do not invoke their private
    # authorization verifiers, model providers or proof issuers.
    sys.path.insert(0, str(root))
    from training.cuda_eval2.metrics import derived_values, raw_diagnostics
    from training.v7.evaluation import summarize
    from training.v7.protocol_audit import COUNTS, audit_record
    from training.v7.scoring import score

    evidence = PublicEvidence(root)
    report = evidence.json(REPORT + "evaluation.json")
    journal = evidence.json(REPORT + "call-journal.json")
    cases_list = evidence.json("training/cuda_eval2/datasets/cases.json")
    protocol = evidence.json(REPORT + "protocol-audit.json")
    values = evidence.json(FIGURES + "values.json")
    require(report["status"] == "complete" and not report.get("pending_case"), "Public evaluation is partial")
    require(report["run_id"] == "resolvedesk-cuda-eval-002", "Different experiment")
    require(
        report["resume_count"] == 0 and report["checkpoint_selection_performed"] is False,
        "Unexpected resume/selection",
    )
    require(tuple(report["summary"]) == ARMS, "Arm identity/order differs")
    cases = {case["id"]: case for case in cases_list}
    require(len(cases_list) == len(cases) == 48, "Expected 48 distinct public worlds")
    expected = [(arm, case["id"]) for arm in ARMS for case in cases_list]
    records = report["records"]
    require(
        [(r["variant"], r["case_id"]) for r in records] == expected,
        "Missing, duplicate or reordered paired records",
    )
    expected_calls = []
    for record in records:
        case = cases[record["case_id"]]
        trace = record["trace"]
        require(0 < record["model_calls"] == len(trace) <= 12, "Task call budget differs")
        require(
            [e["detail"] for e in record["events"] if e["kind"] == "model_tool"] == trace,
            "Raw tool events lost/reordered",
        )
        for key, actual in raw_diagnostics(record, case).items():
            if key in record:
                require(record[key] == actual, "First-raw metric differs: " + key)
        proposals = [o["result"]["proposal"] for o in trace if "proposal" in o["result"]]
        judged = score(
            case,
            {
                "status": record["status_final"],
                "proposal": proposals[-1] if proposals else None,
                "observations": trace,
            },
            record["order_after_confirmation"],
            record["order_before_confirmation"],
            record["questions"],
        )
        require(all(record[key] == value for key, value in judged.items()), "Public business grading differs")
        require(record["order_before_confirmation"] == case["orders"], "Order changed before confirmation")
        actions = Counter(canonical(o["action"]) for o in trace if o["action"])
        require(
            record["identical_action_repeats"] == sum(n - 1 for n in actions.values()),
            "Action-repeat count differs",
        )
        asks = Counter(
            o["action"]["arguments"]["field"]
            for o in trace
            if o["action"] and o["action"]["tool"] == "ask_customer"
        )
        require(
            record["repeated_question_attempts"] == sum(n - 1 for n in asks.values()),
            "Question-repeat count differs",
        )
        require(
            record["output_tokens"] == sum(o["timing"]["output_tokens"] for o in trace),
            "Output token total differs",
        )
        for i, observation in enumerate(trace, 1):
            token_record(observation["timing"])
            expected_calls.append((record["variant"], record["case_id"], i, observation))
    require(len(expected_calls) == len(journal) == 509, "Expected exactly 509 retained model calls")
    for entry, (arm, cid, index, observation) in zip(journal, expected_calls, strict=True):
        require(entry["status"] == "completed", "Ambiguous/inflight model call remains")
        require(
            (entry["variant"], entry["case_id"], entry["call_index"]) == (arm, cid, index),
            "Journal call order differs",
        )
        require(
            entry["raw"] == observation["raw"] and entry["timing"] == observation["timing"],
            "Journal/raw trace differs",
        )
        require(
            isinstance(entry["request_messages"], list) and bool(entry["request_messages"]),
            "Actual request messages missing",
        )
        require(
            entry["request_sha256"] == digest(canonical(entry["request_messages"]).encode()),
            "Actual request-message SHA differs",
        )
    actual_protocol = [audit_record(r, cases[r["case_id"]]) for r in records]
    require(
        protocol["records"] == actual_protocol and len(actual_protocol) == 96,
        "Public scoped-consent ledger differs",
    )
    for arm in ARMS:
        wanted = {
            "tasks": 48,
            "gold_exchange_tasks": sum(r["audited"] for r in actual_protocol if r["variant"] == arm),
            **{key: sum(r[key] for r in actual_protocol if r["variant"] == arm) for key in COUNTS},
        }
        require(protocol["summary"][arm] == wanted, "Scoped-consent totals differ")
        subset = [r for r in records if r["variant"] == arm]
        require(report["summary"][arm] == summarize(subset), "Raw summary differs")
    derived = derived_values(report, cases, protocol)
    for key, wanted in derived.items():
        require(values[key] == wanted, "Chart values differ: " + key)
    csv_bytes = evidence.read(FIGURES + "per-case.csv")
    require(values["csv_sha256"] == digest(csv_bytes), "CSV SHA differs")
    csv_rows = list(csv.DictReader(csv_bytes.decode().splitlines()))
    require([(r["variant"], r["case_id"]) for r in csv_rows] == expected, "CSV row pairs differ")
    index = {(r["variant"], r["case_id"]): r for r in records}
    for row in csv_rows:
        record = index[row["variant"], row["case_id"]]
        for key, text in row.items():
            require(key in record, "Unknown CSV scientific field")
            if key in {"intent_initial", "initial_enum_errors"}:
                require(json.loads(text) == record[key], "CSV raw JSON field differs")
            else:
                require(text == str(record[key]), "CSV field differs: " + key)
    expected_figures = {
        f"{name}.{ext}"
        for name in ("evaluation", "intent", "recovery", "families", "consent", "training")
        for ext in ("png", "svg")
    }
    require(set(values["figure_sha256"]) == expected_figures, "Expected six PNG/SVG chart groups")
    for name, wanted in values["figure_sha256"].items():
        require(digest(evidence.read(FIGURES + name)) == wanted, "Figure SHA differs: " + name)
    for record in records:
        require(
            type(record["latency_ms"]) in (int, float) and math.isfinite(record["latency_ms"]),
            "Nonfinite scientific timing",
        )
    if (Path(root) / "public-redactions.json").is_file():
        evidence.read("public-redactions.json")
    return {
        "schema_version": 1,
        "status": "public_projection_consistent",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "run_id": report["run_id"],
        "counts": {
            "cases": 48,
            "arms": 2,
            "records": 96,
            "completed_model_calls": 509,
            "csv_rows": len(csv_rows),
            "chart_groups": 6,
        },
        "checks": {
            "paired_records": True,
            "business_grading_from_public_state": True,
            "first_raw_fields": True,
            "scoped_consent_ledger": True,
            "repeat_counts": True,
            "all_calls_completed": True,
            "request_message_and_token_array_hashes": True,
            "csv_and_chart_values": True,
            "figure_file_hashes": True,
        },
        "summary": derived["summary"],
        "paired": derived["paired"],
        "scoped_consent_summary": derived["protocol_summary"],
        "failed_records": derived["failed_records"],
        "files": evidence.files,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "full_cpu_token_replay_performed": False,
        "private_authorization_chain_reverified": False,
        "historical_source_hash_fields_recertified": False,
        "model_or_cloud_calls": 0,
        "scope": (
            "Public projection content consistency; failed scores and scientific raw contents are preserved."
        ),
        "limitations": [
            (
                "Token arrays/counts and their published hashes were checked; "
                "no tokenizer decode or prompt reconstruction was performed."
            ),
            (
                "Business scores were recomputed from public state; "
                "this is not a full Store/transaction re-execution."
            ),
            (
                "Historical source/hash fields describe original local lineage. "
                "Removed private authorization files are not recertified."
            ),
            "Consistent evidence does not mean improved model quality or production safety.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "public-reconciliation.json")
    args = parser.parse_args()
    result = reconcile()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
