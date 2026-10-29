"""Write the new comparison document only after real two-arm evidence is independently verified."""

# ruff: noqa: E501 -- Narrative Markdown stays readable as authored paragraphs.
import json
import math
from datetime import datetime

from resolve_desk.evidence_cuda2 import digest, local, validated_report
from training.common import ROOT
from training.cuda_eval2.metrics import ARMS, FIGURES, REPORTS, RUN_ID


def authorization_note(authorized):
    if authorized["authorization_mode"] == "metered_existing_units":
        assert authorized["cap_compute_units"] == 10
        assert authorized["purchases_allowed"] is authorized["renewal_allowed"] is False
        return (
            "Execution used the separately authorized **metered existing-unit path**, with a budget ceiling of "
            "**10 existing Colab compute units** and no purchases or renewal. The stated 80-unit starting "
            "balance, 72-unit operator stop, 70-unit hard floor and two-unit reserve are authorization and "
            "operating limits, **not measured cost or actual units consumed**. The "
            "[pre-inference resource addendum](../reports/cuda-eval2/execution-addendum.json) preserves the "
            "original plan and binds the reviewed supplemental entry point. Balance monitoring and runtime "
            "disconnection belong to the external operator; stopping the Python process alone does not stop "
            "billing. No successful free allocation or automatic compute-unit metering is claimed."
        )
    assert authorized["authorization_mode"] == "free_gpu"
    assert authorized["free_runtime_operator_confirmed"] is True
    return "Execution recorded an explicit free-runtime operator confirmation; paid fallback was prohibited."


def compute_observation_details(root=ROOT):
    """Optional account UI observations; neither is exact per-job consumption."""
    folder = root / "reports/cuda-eval2"
    try:
        release = json.loads((folder / "compute-final-observation.json").read_text())
        assert release["run_id"] == RUN_ID and release["status"] == "runtime_released"
        observed = datetime.fromisoformat(release["observed_at_utc"])
        assert type(release["active_sessions_observed"]) is int and release["active_sessions_observed"] == 0
        assert release["own_runtime_connected"] is False
        assert type(release["model_process_return_code"]) is int and release["model_process_return_code"] == 0
        assert type(release["completed_records"]) is int and release["completed_records"] == 96
        assert release["no_new_purchases_performed"] is release["no_renewal_performed"] is True
        start, end = release["initial_available_compute_units"], release["final_available_compute_units"]
        assert start == 80 and type(end) in (int, float) and math.isfinite(end) and 0 <= end <= start
        assert digest(local(root, release["screenshot"])) == release["screenshot_sha256"]
        assert isinstance(release["scope"], str) and release["scope"].strip()
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return {}
    result = {"release": release}
    try:
        later = json.loads((folder / "compute-post-release-observation.json").read_text())
        assert later["run_id"] == RUN_ID and later["status"] == "runtime_remains_released"
        assert datetime.fromisoformat(later["observed_at_utc"]) >= observed
        assert later["release_time_observation"] == "reports/cuda-eval2/compute-final-observation.json"
        assert later["release_time_available_compute_units"] == end
        assert later["initial_available_compute_units"] == start
        balance = later["latest_available_compute_units"]
        assert type(balance) in (int, float) and math.isfinite(balance) and 0 <= balance <= start
        assert type(later["active_sessions_observed"]) is int and later["active_sessions_observed"] == 0
        assert later["own_runtime_connected"] is later["new_model_or_code_execution_since_release"] is False
        assert later["observed_rate_units_per_hour"] == 0
        assert digest(local(root, later["screenshot"])) == later["screenshot_sha256"]
        assert isinstance(later["scope"], str) and later["scope"].strip()
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return result
    result["later"] = later
    return result


def compute_observation_note(root=ROOT):
    observations = compute_observation_details(root)
    if not observations:
        return ""
    row = observations["release"]
    start, end = row["initial_available_compute_units"], row["final_available_compute_units"]
    note = f"The [release-time UI observation](../reports/cuda-eval2/compute-final-observation.json) showed an account balance of **{start:g} → {end:g} units** (displayed difference **{start - end:.2f}**), zero active sessions and a disconnected runtime."
    if "later" in observations:
        row = observations["later"]
        balance = row["latest_available_compute_units"]
        note += f" A [later UI refresh](../reports/cuda-eval2/compute-post-release-observation.json) showed **{balance:g} units** (difference from the initial balance **{start - balance:.2f}**), still with zero sessions and a displayed rate of zero; no further Colab execution was reported."
    return (
        note
        + " Both are rounded/delayed account UI observations, **not exact final per-job metering**. The original release observation is retained unchanged."
    )


def failure_details(report, cases):
    """Bind explanatory prose to the actual retained failures."""
    index = {(row["variant"], row["case_id"]): row for row in report["records"]}
    payment_id = "cuda-fresh2-test-missing_payment-01"
    option_id = "cuda-fresh2-test-exchange_fallback_missing_payment-01"
    regression_id = "cuda-fresh2-test-exchange_fallback-01"
    payment_rows, option_rows = ([index[arm, cid] for arm in ARMS] for cid in (payment_id, option_id))
    assert cases[payment_id]["intent_initial"]["payment_method"] == "unknown"
    assert all(not row["task_success"] and row["status_final"] == "awaiting_customer" for row in payment_rows)
    assert all(
        not row["questions"] and row["trace"][0]["action"]["arguments"]["payment_method"] == "original"
        for row in payment_rows
    )
    bad_option = option_rows[0]["trace"][0]["action"]["arguments"]["requested_option"]
    wanted_option = cases[option_id]["intent_initial"]["requested_option"]
    assert bad_option != wanted_option
    assert all(
        not row["task_success"] and row["status_final"] == "handoff" and not row["questions"]
        for row in option_rows
    )
    assert all(
        row["trace"][0]["action"]["arguments"]["requested_option"] == bad_option for row in option_rows
    )
    before, after = (index[arm, regression_id] for arm in ARMS)
    assert before["task_success"] and not after["task_success"]
    bad_quote = after["trace"][0]["action"]["arguments"]["evidence"]
    assert bad_quote not in cases[regression_id]["initial_message"]
    assert after["model_calls"] == len(after["trace"]) == 12 and after["identical_action_repeats"] == 11
    assert all(
        o["action"] == after["trace"][0]["action"]
        and o["result"].get("error") == "policy_or_argument_rejection"
        and "exactly quote" in o["result"].get("detail", "")
        for o in after["trace"]
    )
    return {
        "payment_case_id": payment_id,
        "option_case_id": option_id,
        "regression_case_id": regression_id,
        "bad_option": bad_option,
        "wanted_option": wanted_option,
        "bad_quote": bad_quote,
        "quote_rejections": len(after["trace"]),
        "action_repeats": after["identical_action_repeats"],
    }


def metric_table(summary):
    rows = ["| Metric | Predecessor FP16 | Final 1,344 FP16 |", "| --- | ---: | ---: |"]
    for label, numerator, denominator in [
        ("Correctly handled authored tasks", "task_success", "tasks"),
        ("Available-exchange tasks", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("Stockout-fallback tasks", "fallback_success", "fallback_tasks"),
        ("Necessary-handoff tasks", "handoff_success", "handoff_tasks"),
        ("Missing-email tasks", "missing_email_success", "missing_email_tasks"),
        ("Explicit initial grants recognized", "true_permission_yes", "authorized_tasks"),
        ("Unsupported initial YES ↓", "unsupported_permission_yes", "non_authorized_tasks"),
        ("All seven raw initial fields correct", "intent_joint_correct", "intent_tasks"),
        ("Initial raw parse failures ↓", "initial_raw_parse_failures", "intent_tasks"),
        ("Initial enum-error tasks ↓", "initial_enum_error_tasks", "intent_tasks"),
        ("Initial reason-enum error tasks ↓", "initial_reason_enum_error_tasks", "intent_tasks"),
        ("First record accepted by backend", "initial_records_valid", "tasks"),
        ("Schema/enum-valid tool actions", "schema_valid", "model_calls"),
        (
            "Required-clarification tasks asked",
            "required_clarification_tasks_obtained",
            "required_clarification_tasks",
        ),
        (
            "Required fields actually asked",
            "required_clarification_fields_obtained",
            "required_clarification_fields",
        ),
    ]:
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{summary[arm][numerator]}/{summary[arm][denominator]}" for arm in ARMS)
            + " |"
        )
    for label, key in [
        ("Enum-error attempts on any turn ↓", "enum_error_attempts"),
        ("Tasks with a same-turn enum correction", "enum_recovered_tasks"),
        ("Repeated-field question attempts ↓", "repeated_question_attempts"),
        ("Repeated questions delivered ↓", "repeated_questions_delivered"),
        ("All identical-action repeats ↓", "identical_action_repeats"),
        ("Unnecessary handoff attempts ↓", "unnecessary_handoff_attempts"),
        ("Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"),
        ("Rejected record calls ↓", "record_rejections"),
        ("Exact-quote rejections ↓", "quote_rejections"),
        ("Available-exchange tasks with a return attempt ↓", "exchange_as_return"),
        ("Exchange-origin tasks attempting return on another gold outcome ↓", "unauthorized_return"),
        ("Actual accepted confirmation tasks", "accepted_confirmation_tasks"),
        ("Actual declined confirmation tasks", "declined_confirmation_tasks"),
        ("Total model calls", "model_calls"),
        ("Input tokens across all actual calls", "input_tokens"),
        ("Generated tokens including terminal EOS", "output_tokens"),
    ]:
        rows.append("| " + label + " | " + " | ".join(str(summary[arm][key]) for arm in ARMS) + " |")
    rows.append(
        "| Median total calls per task | "
        + " | ".join(f"{summary[arm]['median_model_calls']:g}" for arm in ARMS)
        + " |"
    )
    rows.append(
        "| Median serial task seconds | "
        + " | ".join(f"{summary[arm]['latency_p50_ms'] / 1000:.3f}" for arm in ARMS)
        + " |"
    )
    return "\n".join(rows)


def protocol_table(protocol):
    rows = [
        "| Supplemental exchange-origin consent audit | Predecessor FP16 | Final 1,344 FP16 |",
        "| --- | ---: | ---: |",
    ]
    for label, numerator, denominator in [
        ("Audited original exchanges / all cases", "gold_exchange_tasks", "tasks"),
        (
            "Raw return calls without matching scoped YES / all return calls",
            "return_attempts_without_scoped_yes",
            "raw_return_attempts",
        ),
        (
            "Created proposals without matching scoped YES / all return proposals",
            "return_proposals_without_scoped_yes",
            "return_proposals",
        ),
        (
            "Committed returns without matching scoped YES / all committed returns",
            "committed_returns_without_scoped_yes",
            "committed_returns",
        ),
        (
            "Final returned orders without a verified commit / all final returned orders",
            "final_returns_without_verified_commit",
            "final_return_orders",
        ),
    ]:
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{protocol[arm][numerator]}/{protocol[arm][denominator]}" for arm in ARMS)
            + " |"
        )
    for label, key in [
        ("Rejected unconsented raw return calls", "rejected_return_attempts_without_scoped_yes"),
        ("Declined return proposals", "declined_return_proposals"),
    ]:
        rows.append("| " + label + " | " + " | ".join(str(protocol[arm][key]) for arm in ARMS) + " |")
    return "\n".join(rows)


def main():
    evidence = validated_report(ROOT)
    assert evidence["available"], evidence["reason"] if not evidence["available"] else ""
    values = json.loads((FIGURES / "values.json").read_text())
    assert values["run_id"] == RUN_ID
    quality = json.loads((REPORTS / "data-summary.json").read_text())
    report = json.loads((REPORTS / "evaluation.json").read_text())
    cases = {
        case["id"]: case
        for case in json.loads((ROOT / "training/cuda_eval2/datasets/cases.json").read_text())
    }
    failed = failure_details(report, cases)
    before, after = (evidence["summary"][arm] for arm in ARMS)
    pair, training = evidence["paired"], evidence["training"]
    total = before["tasks"]
    change = (after["task_success"] - before["task_success"]) / total * 100
    moved = "increased" if change > 0 else "decreased" if change < 0 else "stayed unchanged"
    failures = {
        arm: [row["case_id"] for row in values["failed_records"] if row["variant"] == arm] for arm in ARMS
    }
    safety = evidence["protocol_audit"]["candidate_safety_passed"]
    safety_note = (
        "No final-adapter proposal, commit or final returned order lacking the audited evidence was observed in this cohort. This does not close the known bypass."
        if safety
        else "The final adapter produced at least one proposal, commit or returned order lacking the required audited evidence. That failure remains visible independently of business success."
    )
    figure_titles = [
        ("evaluation", "Correctly handled tasks and all core cohorts"),
        ("intent", "First raw interpretation, positive grants and false affirmatives"),
        ("recovery", "Question repeats, identical-action loops and observed efficiency"),
        ("families", "Every authored family and paired case outcomes"),
        ("consent", "Actual scoped-consent evidence, with distinct event units"),
        ("training", "The complete optimizer trace and physical data coverage"),
    ]
    figures = "\n\n".join(f"![{title}](figures/cuda-eval2/{name}.png)" for name, title in figure_titles)
    output = ROOT / "docs/cuda-evaluation2.md"
    text = f"""# ResolveDesk: independent second CUDA evaluation

This is **run002 (`{RUN_ID}`)** on 48 newly authored cases. The interrupted run001 remains separate: its last verified snapshot contained 84/96 completed arm/task records and one ambiguous started call, while later actual progress is unknown. None of its calls is resumed or retried here, and none of its outputs is pooled, replaced or presented as new evidence. This new round reuses the already completed fixed 1,344-update weights; no further training or checkpoint selection occurred.

On **{total} new authored interactions**, correctly handled tasks changed **{before["task_success"]}/{total} → {after["task_success"]}/{total}** ({change:+.1f} percentage points; {moved}). Paired outcomes are **{pair["improved"]} improved, {pair["regressed"]} regressed, {pair["both_pass"]} passed in both arms and {pair["both_fail"]} failed in both**. Explicit-grant recognition changed **{before["true_permission_yes"]}/{before["authorized_tasks"]} → {after["true_permission_yes"]}/{after["authorized_tasks"]}**, alongside unsupported affirmative predictions **{before["unsupported_permission_yes"]}/{before["non_authorized_tasks"]} → {after["unsupported_permission_yes"]}/{after["non_authorized_tasks"]}**.

The baseline is the **already fine-tuned V6.1 predecessor**, and the comparison is its **fixed final 1,344-update CUDA continuation**. Both use the same decoded-4bit-derived FP16 base, FP32 adapter layout, frozen V7 controller, tokenizer, greedy decoding and total-call budget. This evaluates adapter weights under a matched implementation; it does not isolate individual data, weighting or sampler choices, compare Metal with CUDA, or establish production reliability. Historical V7 scores are not pooled.

{authorization_note(evidence["execution_authorization"])}

{compute_observation_note()}

## What was measured

{metric_table(evidence["summary"])}

The cohort has 24 authored families, two cases each, **48 distinct order worlds and initial prompts**, and entities/instruction skeletons disjoint from **{quality["quality"]["historical_cases_checked"]} historical cases**. These are hand-authored synthetic scenarios sharing policy and family structure with development, not independent human annotation or an external retail benchmark. The catalog is limited to one rain-jacket product and two replacement options. Twelve initial requests explicitly grant conditional fallback, 12 refuse and 24 leave permission unknown. The first model output is scored before any accepted record, enum repair or backend normalization. A first output without a complete eight-string `record_request` argument object fails raw-slot extraction; invalid enum values remain visible. Backend acceptance and action-schema validity are separate rows, so correct raw slots do not imply a valid executed action. Later model corrections remain separate counts.

Correct handling includes a fixture's declared acceptance or refusal of the final proposal. The dataset declares 24 accept and 24 decline choices across all cases; eight of each belong to necessary-handoff cases that never reach confirmation. Actual confirmation counts above come from recorded commit/decline events, not those declared choices. The 18 tasks missing information require 20 field answers; asking a field is a process count, while full task success also requires the preserved business grader to pass.

Repeated-question metrics concern repeated fields. Identical-action repeats also include repeated records and reads, so zero repeated questions does not establish that all tool loops disappeared. The historical `unauthorized_return` business field means an exchange-origin task attempted return when its gold outcome was different; it does not certify every execution-consent condition.

## Supplemental execution-consent audit

{protocol_table(evidence["protocol_audit"]["summary"])}

All 96 raw arm/task traces were replayed and all original-exchange returns audited from actual events. A valid grant requires an observed unavailable replacement, a successful scoped fallback question, a subsequent exact customer `YES` and matching active order, replacement, payment, request and revision. Attempts, created proposals, commits and final returned orders have different units and are never summed. Blocked calls and declined proposals remain visible. {safety_note}

The frozen controller still permits a model to rewrite an accepted exchange into a direct return after clarification, potentially bypassing the exchange-consent path. This audit detects observed exposure without changing the controller or business score. Initially direct-return semantic authorization and unrestricted genuine customer operation changes are outside its scope. No promotion or production-safety verdict is inferred from a zero count.

## Training and evaluation controls

The separate CUDA run completed **{training["completed_updates"]} true Adam updates** on **824 unique physical training pairs**, training **{training["parameters"]:,} parameters in 112 rank-8 adapter tensors**. Both initial and final tensors were read independently on CPU; the saved starting values exactly match the predecessor after transpose. Final adapter L2 change is **{training["adapter_l2_change"]:.9f}**. The final 1,344 checkpoint was declared before test inference; the 672 backup was not selected or evaluated on this cohort. All 824 training pairs were sampled during the formal run. Weighted completion loss uses ordinary tokens 1, permission values 8 and reason values 4, with prompt/padding 0 and full input context retained.

Decoding allows **512 generated tokens** and **12 TOTAL model calls per task**, including every rejected action and every customer turn. There is no JSON constraint, hidden warmup, uncounted correction call or retry to obtain a pass. Model-issued repairs are retained and count toward the same 12-call total. The journal retains every started/completed model call, full input/output token IDs and raw text including special tokens; only an actual terminal EOS is removed for text transport. Deterministic replay checks actions, actual replies, proposal confirmation, prompt tokens and output decoding without an LLM. The scoped-consent audit is separate from the business scorer.

Serial evaluation runs predecessor cases before final-adapter cases. Task wall time includes journaling and workflow overhead, model-generation timing uses CUDA synchronization, and later prompts can diverge because actions differ. Observed latency/call distributions are descriptive; no isolated hardware or production speedup is claimed. Sampled training loss is not held-out accuracy, and the small nonrandom family-balanced cohort does not justify population confidence intervals or significance claims.

The retained traces identify three concrete failure types:

- In `{failed["payment_case_id"]}`, both arms recorded the unspecified refund payment as `original`, skipped the required payment question and created a return proposal. Both remained `awaiting_customer` and failed the authored task.
- In `{failed["option_case_id"]}`, both arms recorded `{failed["bad_option"]}` instead of the exact catalog option `{failed["wanted_option"]}`. They handed off without asking the missing payment field.
- The sole new regression, `{failed["regression_case_id"]}`, repeated the identical quote `{failed["bad_quote"]}`. It is not an exact span of the actual customer message. All {failed["quote_rejections"]} record calls were rejected, causing {failed["action_repeats"]} identical-action repeats and exhausting the total-call budget. This is an evidence-quotation failure, not a reason-enum error; correct seven-slot extraction does not establish an accepted record or successful execution.

Current failures remain: predecessor **{", ".join(failures[ARMS[0]]) or "none in this small cohort"}**; final continuation **{", ".join(failures[ARMS[1]]) or "none in this small cohort"}**. [Raw model trajectories](../reports/cuda-eval2/evaluation.json), [call journal](../reports/cuda-eval2/call-journal.json), [pre-inference analysis plan](../reports/cuda-eval2/analysis-plan.json), [sealed checkpoint identity](../reports/cuda-eval2/artifact-identity.json), [scoped-consent rows](../reports/cuda-eval2/protocol-audit.json), [independent replay](../reports/cuda-eval2/verification.json), [figure reconciliation](../reports/cuda-eval2/figure-verification.json) and [per-case CSV](figures/cuda-eval2/per-case.csv) retain the evidence. The read-only app report uses these artifacts; live controller behavior is unchanged.

{figures}

## Reproduction and interpretation

Source inputs, the training proof and every CSV/PNG/SVG are hash-bound. [Training proof](../reports/cuda-eval2/training-proof.json) links the actual complete CUDA training report and independent numeric checks. [Dataset audit](../reports/cuda-eval2/data-summary.json) records denominators, split isolation and historical input hashes. The reference feasibility traces are CPU oracles and are never presented as model scores. Retail policy attribution remains in [third_party/tau-bench](../third_party/tau-bench/README.md).

This is a local prototype result. Keep the comparator, synthetic scope, exact denominators, raw-versus-repaired distinction, residual failures and supplemental consent findings beside any project or résumé claim.
"""
    output.write_text(text)
    print(str(output))


if __name__ == "__main__":
    main()
