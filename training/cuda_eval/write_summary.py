"""Write the new comparison document only after real two-arm evidence is independently verified."""

# ruff: noqa: E501 -- Narrative Markdown stays readable as authored paragraphs.
import json

from resolve_desk.evidence_cuda import validated_report
from training.common import ROOT
from training.cuda_eval.metrics import ARMS, FIGURES, REPORTS


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
    quality = json.loads((REPORTS / "data-summary.json").read_text())
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
    figures = "\n\n".join(f"![{title}](figures/cuda-eval/{name}.png)" for name, title in figure_titles)
    output = ROOT / "docs/cuda-evaluation.md"
    text = f"""# ResolveDesk: matched CUDA evaluation before and after continuation

On **{total} new authored interactions**, correctly handled tasks changed **{before["task_success"]}/{total} → {after["task_success"]}/{total}** ({change:+.1f} percentage points; {moved}). Paired outcomes are **{pair["improved"]} improved, {pair["regressed"]} regressed, {pair["both_pass"]} passed in both arms and {pair["both_fail"]} failed in both**. Explicit-grant recognition changed **{before["true_permission_yes"]}/{before["authorized_tasks"]} → {after["true_permission_yes"]}/{after["authorized_tasks"]}**, alongside unsupported affirmative predictions **{before["unsupported_permission_yes"]}/{before["non_authorized_tasks"]} → {after["unsupported_permission_yes"]}/{after["non_authorized_tasks"]}**.

The baseline is the **already fine-tuned V6.1 predecessor**, and the comparison is its **fixed final 1,344-update CUDA continuation**. Both use the same decoded-4bit-derived FP16 base, FP32 adapter layout, frozen V7 controller, tokenizer, greedy decoding and total-call budget. This evaluates adapter weights under a matched implementation; it does not isolate individual data, weighting or sampler choices, compare Metal with CUDA, or establish production reliability. Historical V7 scores are not pooled.

## What was measured

{metric_table(evidence["summary"])}

The cohort has 24 authored families, two cases each, **48 distinct order worlds and initial prompts**, and entities/instruction skeletons disjoint from **{quality["quality"]["historical_cases_checked"]} historical cases**. These are hand-authored synthetic scenarios sharing policy and family structure with development, not independent human annotation or an external retail benchmark. Twelve initial requests explicitly grant conditional fallback, 12 refuse and 24 leave permission unknown. The first model output is scored before any accepted record, enum repair or backend normalization. A first output without a complete eight-string `record_request` argument object fails raw-slot extraction; invalid enum values remain visible. Backend acceptance and action-schema validity are separate rows, so correct raw slots do not imply a valid executed action. Later model corrections remain separate counts.

Correct handling includes a fixture's declared acceptance or refusal of the final proposal. The dataset declares 24 accept and 24 decline choices across all cases; eight of each belong to necessary-handoff cases that never reach confirmation. Actual confirmation counts above come from recorded commit/decline events, not those declared choices. The 18 tasks missing information require 20 field answers; asking a field is a process count, while full task success also requires the preserved business grader to pass.

Repeated-question metrics concern repeated fields. Identical-action repeats also include repeated records and reads, so zero repeated questions does not establish that all tool loops disappeared. The historical `unauthorized_return` business field means an exchange-origin task attempted return when its gold outcome was different; it does not certify every execution-consent condition.

## Supplemental execution-consent audit

{protocol_table(evidence["protocol_audit"]["summary"])}

All 96 raw arm/task traces were replayed and all original-exchange returns audited from actual events. A valid grant requires an observed unavailable replacement, a successful scoped fallback question, a subsequent exact customer `YES` and matching active order, replacement, payment, request and revision. Attempts, created proposals, commits and final returned orders have different units and are never summed. Blocked calls and declined proposals remain visible. {safety_note}

The frozen controller still permits a model to rewrite an accepted exchange into a direct return after clarification, potentially bypassing the exchange-consent path. This audit detects observed exposure without changing the controller or business score. Initially direct-return semantic authorization and unrestricted genuine customer operation changes are outside its scope. No promotion or production-safety verdict is inferred from a zero count.

## Training and evaluation controls

The separate CUDA run completed **{training["completed_updates"]} true Adam updates** on **824 unique physical training pairs**, training **{training["parameters"]:,} parameters in 112 rank-8 adapter tensors**. Both initial and final tensors were read independently on CPU; the saved starting values exactly match the predecessor after transpose. Final adapter L2 change is **{training["adapter_l2_change"]:.9f}**. The final 1,344 checkpoint was declared before test inference; the 672 backup was not selected or evaluated on this cohort. All 824 training pairs were sampled during the formal run. Weighted completion loss uses ordinary tokens 1, permission values 8 and reason values 4, with prompt/padding 0 and full input context retained.

Decoding allows **512 generated tokens** and **12 TOTAL model calls per task**, including every rejected action and every customer turn. There is no JSON constraint, hidden warmup, correction call or retry to obtain a pass. The journal retains every started/completed model call, full input/output token IDs and raw text including special tokens; only an actual terminal EOS is removed for text transport. Deterministic replay checks actions, actual replies, proposal confirmation, prompt tokens and output decoding without an LLM. The scoped-consent audit is separate from the business scorer.

Serial evaluation runs predecessor cases before final-adapter cases. Task wall time includes journaling and workflow overhead, model-generation timing uses CUDA synchronization, and later prompts can diverge because actions differ. Observed latency/call distributions are descriptive; no isolated hardware or production speedup is claimed. Sampled training loss is not held-out accuracy, and the small nonrandom family-balanced cohort does not justify population confidence intervals or significance claims.

Current failures remain: predecessor **{", ".join(failures[ARMS[0]]) or "none in this small cohort"}**; final continuation **{", ".join(failures[ARMS[1]]) or "none in this small cohort"}**. [Raw model trajectories](../reports/cuda-eval/evaluation.json), [call journal](../reports/cuda-eval/call-journal.json), [pre-inference analysis plan](../reports/cuda-eval/analysis-plan.json), [sealed checkpoint identity](../reports/cuda-eval/artifact-identity.json), [scoped-consent rows](../reports/cuda-eval/protocol-audit.json), [independent replay](../reports/cuda-eval/verification.json), [figure reconciliation](../reports/cuda-eval/figure-verification.json) and [per-case CSV](figures/cuda-eval/per-case.csv) retain the evidence. The read-only app report uses these artifacts; live controller behavior is unchanged.

{figures}

## Reproduction and interpretation

Source inputs, the training proof and every CSV/PNG/SVG are hash-bound. [Training proof](../reports/cuda-eval/training-proof.json) links the actual complete CUDA training report and independent numeric checks. [Dataset audit](../reports/cuda-eval/data-summary.json) records denominators, split isolation and historical input hashes. The reference feasibility traces are CPU oracles and are never presented as model scores. Retail policy attribution remains in [third_party/tau-bench](../third_party/tau-bench/README.md).

This is a local prototype result. Keep the comparator, synthetic scope, exact denominators, raw-versus-repaired distinction, residual failures and supplemental consent findings beside any project or résumé claim.
"""
    output.write_text(text)
    print(str(output))


if __name__ == "__main__":
    main()
