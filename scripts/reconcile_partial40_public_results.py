#!/usr/bin/env python3
"""Reconcile the public common40 projection, without raw-record or model replay.

Standard library only. Published CSV outcomes are inputs, not independently
rescored business decisions. Opaque source-record digests cannot be recomputed
without the intentionally unpublished raw records.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path

ARMS = ("prior_weights_old_flow", "prior_weights_new_flow", "v8_weights_new_flow")
FIELDS = ("operation", "email", "order_id", "requested_option", "payment_method", "reason", "fallback_return")
FIG = "docs/figures/v8-eval2-partial40"
SUMMARY = "reports/v8-eval2-partial40/public-summary.json"
CASES = "training/v8_eval2/datasets/cases.json"
PLOTS = tuple(
    name + ext
    for name in ("evaluation", "intent", "recovery", "families", "consent", "training")
    for ext in (".png", ".svg")
)
CSVS = ("metrics.csv", "per-case.csv", "case-mapping.csv", "training.csv")
ASSETS = PLOTS + CSVS
SHA = re.compile(r"^[0-9a-f]{64}$")


class ReconciliationError(ValueError):
    """A public artifact violates the published aggregate contract."""


def require(condition, message):
    if not condition:
        raise ReconciliationError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative, "Unsafe relative input path")
    rel = Path(relative)
    require(
        not rel.is_absolute() and all(p not in ("", ".", "..") for p in relative.split("/")),
        "Unsafe relative input path",
    )
    path = root / rel
    for parent in [path, *path.parents]:
        require(not parent.is_symlink(), "Symlink input rejected: " + relative)
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Missing regular public input: " + relative)
    require(path.stat().st_size <= 128 * 1024 * 1024, "Oversize public input: " + relative)
    return path.read_bytes()


def json_file(root, relative):
    data = read_bytes(root, relative)
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ReconciliationError("Invalid JSON: " + relative) from exc


def rows(root, name):
    try:
        reader = csv.DictReader(io.StringIO(read_bytes(root, FIG + "/" + name).decode("utf-8")))
        require(
            reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
            "Duplicate CSV columns: " + name,
        )
        result = list(reader)
        require(
            all(None not in row and all(v is not None for v in row.values()) for row in result),
            "Malformed CSV: " + name,
        )
        return reader.fieldnames, result
    except UnicodeError as exc:
        raise ReconciliationError("Invalid UTF-8 CSV: " + name) from exc


def boolean(value):
    require(value in ("True", "False"), "Expected an explicit CSV True/False")
    return value == "True"


def integer(value):
    require(
        isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
        "Expected a nonnegative integer",
    )
    return int(value)


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconciliationError("Expected a finite number") from exc
    require(math.isfinite(x), "Nonfinite number")
    return x


def same(left, right, label):
    require(left == right, "Aggregate mismatch: " + label)


def reconcile(root):
    root = Path(os.path.abspath(root))
    require(root.is_dir(), "Missing public repository directory")
    summary = json_file(root, SUMMARY)
    values_bytes = read_bytes(root, FIG + "/values.json")
    values = json.loads(values_bytes)
    cases = json_file(root, CASES)
    same(
        digest(read_bytes(root, CASES)),
        "e2e14c1b4694fa0ea61ff25b74e5bfeadd5c5d6d4c34b8dfac53f1fd1fccfba3",
        "fixed authored48 case bytes",
    )
    require(isinstance(cases, list) and len(cases) == 48, "Expected the original48 authored cases")
    ids = [c["id"] for c in cases]
    require(
        len(set(ids)) == 48 and all(cid.startswith("v8-eval2-test-") for cid in ids),
        "Unexpected or duplicate case identity",
    )
    family_counts = Counter(c["family"] for c in cases)
    require(
        len(family_counts) == 24 and set(family_counts.values()) == {2},
        "Expected24 families with2 cases each",
    )
    selected = cases[:40]
    selected_ids = ids[:40]
    by_id = {c["id"]: c for c in selected}
    selected_families = Counter(c["family"] for c in selected)
    excluded_families = [f for f in family_counts if f not in selected_families]
    same(len(selected_families), 20, "selected family count")
    same(
        excluded_families,
        [
            "exchange_available_refusal",
            "exchange_fallback_missing_payment",
            "exchange_missing_email",
            "missing_email_payment",
        ],
        "omitted families",
    )
    same(summary["analysis_id"], "resolvedesk-v8-eval-002-partial40", "analysis ID")
    same(summary["run_id"], "resolvedesk-v8-eval-002", "run ID")
    same(summary["arms"], list(ARMS), "arm order")
    same(summary["public_values_sha256"], digest(values_bytes), "public values digest")
    same(set(values["output_files_sha256"]), set(ASSETS), "allowlisted asset inventory")
    same(
        set(summary["public_files_sha256"]), {FIG + "/" + name for name in ASSETS}, "summary asset inventory"
    )
    for name in ASSETS:
        content = read_bytes(root, FIG + "/" + name)
        same(digest(content), values["output_files_sha256"][name], "asset digest " + name)
        same(digest(content), summary["public_files_sha256"][FIG + "/" + name], "summary digest " + name)
        if name.endswith(".png"):
            require(content.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG asset")
        elif name.endswith(".svg"):
            require(b"<svg" in content, "Invalid SVG asset")
    require("files" not in values and "raw_trace" not in values, "Private evidence map is not public input")
    projection = values["public_projection"]
    for flag in [
        "raw_trajectories_published",
        "model_input_output_tokens_published",
        "operator_or_account_receipts_published",
        "complete_144_result_available",
        "public_token_or_full_business_replay_supported",
    ]:
        same(projection[flag], False, "public scope " + flag)
    for source_sha in summary["source_digests"].values():
        require(isinstance(source_sha, str) and SHA.fullmatch(source_sha), "Invalid source digest")
    for arm_sha in summary["saved_weights_sha256"].values():
        require(isinstance(arm_sha, str) and SHA.fullmatch(arm_sha), "Invalid saved-weight identity")
    same(summary["saved_weights_sha256"], values["saved_weights"], "saved adapter identities")
    same(
        summary["saved_weights_sha256"],
        {
            "before": "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053",
            "after": "39dabb9b73ae6cb6230fc8a0bca9c914f6ba658c3c54a17f9775e3fcef60970f",
        },
        "fixed saved adapters",
    )
    for source in [summary, values]:
        same(source["scope"]["original144_gate_complete"], False, "incomplete original experiment")
        same(source["scope"]["random_truncation"], False, "nonrandom truncation")
        same(source["scope"]["fixed_arm_block_order"], True, "arm-block order")
        same(source["scope"]["extrapolation_to_original48_allowed"], False, "no48-case extrapolation")
        same(source["scope"]["old_interrupted_run_results_pooled"], False, "no historical pooling")
        same(source["scope"]["external_benchmark_or_production_data"], False, "synthetic scope")
    same(values["original144_gate_complete"], False, "values incomplete original experiment")
    same(values["baseline_is_already_finetuned"], True, "already-tuned baseline")
    same(values["new_training_performed_in_this_run"], False, "zero new training")
    verification = summary["local_verification_summary"]
    same(verification["original144_gate_complete"], False, "local proof original incomplete")
    same(verification["original_full_result_verified"], False, "no full144 verification")
    same(verification["public_raw_token_replay_supported"], False, "no public raw replay")
    same(
        verification["public_copy_recertifies_original_full_source_seal"], False, "no public recertification"
    )

    denominators = {
        "tasks": 40,
        "families": dict(selected_families),
        "initial_permissions": dict(Counter(c["intent_initial"]["fallback_return"] for c in selected)),
        "non_authorized_tasks": sum(c["intent_initial"]["fallback_return"] != "yes" for c in selected),
        "expected_outcomes": dict(Counter(c["expected"]["kind"] for c in selected)),
        "available_exchange_tasks": sum(c["expected"]["kind"] == "exchange" for c in selected),
        "authorized_stockout_return_tasks": sum(
            c["expected"]["kind"] == "return" and c["intent_initial"]["operation"] == "exchange"
            for c in selected
        ),
        "required_handoff_tasks": sum(c["expected"]["kind"] == "handoff" for c in selected),
        "missing_email_tasks": sum("email" in c["required_answers"] for c in selected),
        "tasks_requiring_clarification": sum(bool(c["required_answers"]) for c in selected),
        "required_answer_occurrences": sum(len(c["required_answers"]) for c in selected),
        "required_answer_fields": dict(Counter(field for c in selected for field in c["required_answers"])),
        "exchange_origin_tasks": sum(c["intent_initial"]["operation"] == "exchange" for c in selected),
        "confirmation_choices_all_tasks": dict(Counter(str(c["confirmation"]).lower() for c in selected)),
        "confirmation_choices_non_handoff": dict(
            Counter(str(c["confirmation"]).lower() for c in selected if c["expected"]["kind"] != "handoff")
        ),
    }
    same(denominators, values["selection_denominators"], "case-derived denominators in values")
    same(denominators, summary["selection_denominators"], "case-derived denominators in summary")
    for obj in [summary, values]:
        same(obj["coverage"]["included_families"], list(selected_families), "included families")
        same(obj["coverage"]["excluded_families"], excluded_families, "excluded families")
        same(
            obj["coverage"]["per_family"],
            {
                f: {
                    "original_cases": n,
                    "selected_cases": selected_families[f],
                    "excluded_cases": n - selected_families[f],
                }
                for f, n in family_counts.items()
            },
            "family coverage",
        )

    columns, per_case = rows(root, "per-case.csv")
    require(
        "raw_trace" not in columns and {"source_record_sha256", "original_record_index"} <= set(columns),
        "Expected public provenance columns, without a raw trace path",
    )
    same(len(per_case), 120, "120 published rows")
    same(
        [(r["variant"], r["case_id"]) for r in per_case],
        [(a, cid) for a in ARMS for cid in selected_ids],
        "fixed arm/case row order",
    )
    record_digests = []
    index = {}
    for ordinal, row in enumerate(per_case):
        arm = row["variant"]
        c = by_id[row["case_id"]]
        same(row["family"], c["family"], "CSV family")
        same(row["expected_kind"], c["expected"]["kind"], "CSV expected business kind")
        same(boolean(row["confirmation_choice"]), c["confirmation"], "declared confirmation choice")
        same(
            integer(row["original_record_index"]),
            (ordinal // 40) * 48 + ordinal % 40,
            "original retained-record index",
        )
        require(SHA.fullmatch(row["source_record_sha256"]) is not None, "Invalid opaque record digest")
        record_digests.append(row["source_record_sha256"])
        same(
            row["initial_permission_gold"], c["intent_initial"]["fallback_return"], "initial permission truth"
        )
        require(
            row["initial_permission_prediction"] in ("yes", "no", "unknown", "None", ""),
            "Invalid initial permission prediction",
        )
        fc = {k: boolean(row["initial_" + k + "_correct"]) for k in FIELDS}
        same(boolean(row["joint_initial_raw_fields_correct"]), all(fc.values()), "joint initial fields")
        parse_failed = boolean(row["initial_raw_parse_failure"])
        require(not parse_failed or not any(fc.values()), "Parse failure borrowed repaired fields")
        if not parse_failed:
            same(
                fc["fallback_return"],
                row["initial_permission_gold"] == row["initial_permission_prediction"],
                "first permission correctness",
            )
        errors = json.loads(row["initial_enum_errors"])
        require(isinstance(errors, dict), "Expected initial enum error object")
        calls = integer(row["model_calls"])
        require(0 < calls <= 12, "Per-task total model-call cap")
        require(finite(row["latency_ms"]) >= 0, "Negative retained latency")
        for k in [
            "quote_rejections",
            "enum_error_attempts",
            "record_rejections",
            "repeated_question_attempts",
            "repeated_questions_delivered",
            "identical_action_repeats",
            "unnecessary_handoff_attempts",
            "unnecessary_handoff_executed",
        ]:
            integer(row[k])
        index[arm, row["case_id"]] = row
    same(len(set(record_digests)), 120, "unique opaque record digests")
    computed = {}
    for arm in ARMS:
        rs = [index[arm, cid] for cid in selected_ids]

        def success(row):
            return boolean(row["task_success"])

        def cohort(pred, rows=rs):
            return [row for row in rows if pred(by_id[row["case_id"]])]

        available = cohort(lambda c: c["expected"]["kind"] == "exchange")
        fallback = cohort(
            lambda c: c["expected"]["kind"] == "return" and c["intent_initial"]["operation"] == "exchange"
        )
        handoff = cohort(lambda c: c["expected"]["kind"] == "handoff")
        email = cohort(lambda c: "email" in c["required_answers"])
        s = {
            "tasks": 40,
            "intent_tasks": 40,
            "task_success": sum(success(r) for r in rs),
            "fulfillable_exchange_tasks": len(available),
            "fulfillable_exchange_success": sum(success(r) for r in available),
            "fallback_tasks": len(fallback),
            "fallback_success": sum(success(r) for r in fallback),
            "handoff_tasks": len(handoff),
            "handoff_success": sum(success(r) for r in handoff),
            "missing_email_tasks": len(email),
            "missing_email_success": sum(success(r) for r in email),
            "authorized_tasks": denominators["initial_permissions"]["yes"],
            "non_authorized_tasks": denominators["non_authorized_tasks"],
            "true_permission_yes": sum(
                r["initial_permission_gold"] == "yes" and r["initial_permission_prediction"] == "yes"
                for r in rs
            ),
            "unsupported_permission_yes": sum(
                r["initial_permission_gold"] != "yes" and r["initial_permission_prediction"] == "yes"
                for r in rs
            ),
            "initial_raw_parse_failures": sum(boolean(r["initial_raw_parse_failure"]) for r in rs),
            "initial_enum_error_tasks": sum(bool(json.loads(r["initial_enum_errors"])) for r in rs),
            "initial_reason_enum_error_tasks": sum(
                "reason" in json.loads(r["initial_enum_errors"]) for r in rs
            ),
            "intent_joint_correct": sum(boolean(r["joint_initial_raw_fields_correct"]) for r in rs),
            "intent_field_correct": {
                k: sum(boolean(r["initial_" + k + "_correct"]) for r in rs) for k in FIELDS
            },
            "model_calls": sum(integer(r["model_calls"]) for r in rs),
            **{
                k: sum(integer(r[k]) for r in rs)
                for k in [
                    "quote_rejections",
                    "enum_error_attempts",
                    "record_rejections",
                    "repeated_question_attempts",
                    "repeated_questions_delivered",
                    "identical_action_repeats",
                    "unnecessary_handoff_attempts",
                    "unnecessary_handoff_executed",
                ]
            },
        }
        computed[arm] = s
        for k, n in s.items():
            same(n, values["summary"][arm][k], "CSV-derived values " + arm + "/" + k)
            same(n, summary["summary"][arm][k], "CSV-derived summary " + arm + "/" + k)
    same(summary["summary"], values["summary"], "all projected summary fields")
    same(sum(x["model_calls"] for x in computed.values()), 657, "selected model-call total")
    same([computed[a]["task_success"] for a in ARMS], [30, 30, 33], "published business outcomes")
    case_outcomes = [
        {
            "case_id": cid,
            "family": by_id[cid]["family"],
            **{a: boolean(index[a, cid]["task_success"]) for a in ARMS},
        }
        for cid in selected_ids
    ]
    failures = [
        {"case_id": row["case_id"], **{a: row[a] for a in ARMS}}
        for row in case_outcomes
        if not all(row[a] for a in ARMS)
    ]
    failed_records = [
        {
            "variant": row["variant"],
            "case_id": row["case_id"],
            "status": row["status_final"],
            "model_calls": integer(row["model_calls"]),
        }
        for row in per_case
        if not boolean(row["task_success"])
    ]
    same(len(failed_records), 27, "all27 failed rows")
    same(len(failures), 16, "all16 failure case IDs")
    for obj in [summary, values]:
        same(obj["cases"], case_outcomes, "all40 case outcomes")
        same(obj["failures"], failures, "complete failure union")
        same(obj["failed_records"], failed_records, "complete failed arm/case rows")
    paired = {}
    for name, before, after in [
        ("workflow", ARMS[0], ARMS[1]),
        ("training_under_v8", ARMS[1], ARMS[2]),
        ("combined", ARMS[0], ARMS[2]),
    ]:
        counts = Counter({"both_pass": 0, "both_fail": 0, "improved": 0, "regressed": 0})
        for cid in selected_ids:
            left, right = (
                boolean(index[before, cid]["task_success"]),
                boolean(index[after, cid]["task_success"]),
            )
            counts[
                "both_pass"
                if left and right
                else "both_fail"
                if not left and not right
                else "improved"
                if right
                else "regressed"
            ] += 1
        paired[name] = {"before": before, "after": after, "tasks": 40, **counts}
    same(paired, values["paired"], "three paired comparisons")
    same(paired, summary["paired"], "summary paired comparisons")
    _, mapping = rows(root, "case-mapping.csv")
    same(
        mapping,
        [
            {"case_id": r["case_id"], "family": r["family"], **{a: str(r[a]) for a in ARMS}}
            for r in case_outcomes
        ],
        "case-mapping CSV",
    )
    metric_keys = {
        "Correctly handled selected tasks": ("task_success", "tasks"),
        "Available exchange": ("fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        "Stockout fallback": ("fallback_success", "fallback_tasks"),
        "Necessary handoff": ("handoff_success", "handoff_tasks"),
        "Missing email": ("missing_email_success", "missing_email_tasks"),
        "Joint first raw fields": ("intent_joint_correct", "intent_tasks"),
        "Explicit initial YES": ("true_permission_yes", "authorized_tasks"),
        "Unsupported initial YES (lower)": ("unsupported_permission_yes", "non_authorized_tasks"),
        "First raw parse failures (lower)": ("initial_raw_parse_failures", "intent_tasks"),
        "First reason enum errors (lower)": ("initial_reason_enum_error_tasks", "intent_tasks"),
    }
    _, metric_rows = rows(root, "metrics.csv")
    same(
        [(r["variant"], r["metric"]) for r in metric_rows],
        [(a, m) for a in ARMS for m in metric_keys],
        "30-metric CSV inventory",
    )
    for row in metric_rows:
        n, d = metric_keys[row["metric"]]
        s = computed[row["variant"]]
        same(integer(row["numerator"]), s[n], "metric numerator")
        same(integer(row["denominator"]), s[d], "metric denominator")
        same(finite(row["percent"]), 100 * s[n] / s[d], "metric percentage")
    _, curve = rows(root, "training.csv")
    same(len(curve), 1344, "historical curve length")
    same([integer(r["iteration"]) for r in curve], list(range(1, 1345)), "historical update sequence")
    same(
        [integer(r["true_optimizer_updates"]) for r in curve],
        list(range(1, 1345)),
        "historical true-update sequence",
    )
    require(all(integer(r["sample_index"]) < 1100 for r in curve), "Historical sample index outside1100 rows")
    same(len({integer(r["sample_index"]) for r in curve}), 888, "888 sampled physical rows")
    same(len({r["case_id"] for r in curve}), 96, "96 historical training cases")
    same(
        dict(Counter(r["sampling_bucket"] for r in curve)),
        {"intent_and_record_repair": 672, "execution_and_reply": 672},
        "historical bucket updates",
    )
    for row in curve:
        same(boolean(row["gradients_finite"]), True, "historical finite gradients")
        for k in [
            "loss",
            "unscaled_gradient_norm",
            "grad_scaler_scale",
            "adapter_l2_change_from_initializer",
            "elapsed_gpu_sync_seconds",
            "elapsed_loop_gpu_sync_seconds",
        ]:
            require(finite(row[k]) >= 0, "Negative historical training metric")
    training = values["training"]
    same(summary["historical_training"], training, "historical metadata projection")
    same(training["updates"], 1344, "1344 saved updates")
    same(training["new_updates_this_experiment"], 0, "zero new optimizer updates")
    same(training["historical_saved_run_only"], True, "historical curve only")
    same(training["sampling"]["physical_rows"], 1100, "1100 physical training rows")
    same(training["sampling"]["unique_rows_sampled"], 888, "sampled physical coverage")
    same(training["sampling"]["all_physical_rows_used"], False, "no false full coverage claim")
    curve_times = [finite(row["elapsed_loop_gpu_sync_seconds"]) for row in curve]
    require(curve_times == sorted(curve_times), "Nonmonotonic historical curve time")
    require(
        finite(training["elapsed_loop_gpu_sync_seconds"]) >= curve_times[-1],
        "Historical total time precedes its final curve entry",
    )
    same(summary["protocol_summary"], values["protocol_summary"], "projected scope-consent event units")
    same(summary["exclusions"], values["exclusions"], "projected exclusions")
    same(
        summary["counts"],
        {
            "original_cases_per_arm": 48,
            "original_expected_task_records": 144,
            "completed_task_records_retained_locally": 136,
            "completed_known_model_calls_retained_locally": 765,
            "started_unknown_response_calls": 1,
            "paired_case_ids": 40,
            "paired_task_records": 120,
            "paired_known_model_calls": 657,
            "completed_old_arm_tail_records_excluded": 16,
            "candidate_unstarted_tasks": 7,
            "candidate_ambiguous_task": 1,
            "selected_families": 20,
            "original_families": 24,
            "failed_task_records_retained": 27,
            "case_ids_with_at_least_one_failed_arm": 16,
        },
        "partial-versus-original scope counts",
    )
    return {
        "status": "consistent_public_partial40_aggregates",
        "analysis_id": summary["analysis_id"],
        "cases": 40,
        "paired_rows": 120,
        "paired_calls": 657,
        "business_outcomes": [computed[a]["task_success"] for a in ARMS],
        "failed_rows": 27,
        "failure_case_ids": 16,
        "included_families": 20,
        "original_families": 24,
        "historical_updates": 1344,
        "historical_rows_sampled": 888,
        "historical_physical_rows": 1100,
        "original144_gate_complete": False,
        "raw_record_or_tokens_verified": False,
        "private_full_source_seal_verified": False,
        "new_model_or_gpu_calls": 0,
        "new_training_updates": 0,
        "verification_scope": (
            "Public aggregate/case CSV consistency and asset digests only; published outcome labels "
            "are not rescored. Opaque record digests are format/uniqueness/bound-CSV checked, "
            "not recomputed from absent raw records. Event-only confirmation/consent counters "
            "are checked between public projections, not independently replayed."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument(
        "--output", type=Path, help="Optional exclusive JSON receipt, outside the public repository"
    )
    args = parser.parse_args(argv)
    try:
        result = reconcile(args.root)
        data = (json.dumps(result, indent=2) + "\n").encode()
        if args.output:
            out = Path(os.path.abspath(args.output))
            root = Path(os.path.abspath(args.root))
            require(not out.is_relative_to(root), "Receipt output must be outside the public repository")
            for parent in [out, *out.parents]:
                require(not parent.is_symlink(), "Symlink receipt path rejected")
            require(out.parent.is_dir(), "Receipt parent directory must already exist")
            with out.open("xb") as stream:
                stream.write(data)
        print(json.dumps(result, separators=(",", ":")))
    except (ReconciliationError, KeyError, TypeError, ValueError, OSError) as exc:
        print(
            json.dumps({"status": "public_aggregate_reconciliation_failed", "error": str(exc)}),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
