"""Generate the final README only from reconciled real training/evaluation evidence."""

# ruff: noqa: E501 -- Authored Markdown paragraphs remain readable.
import json
import xml.etree.ElementTree as ET
from collections import Counter

from training.common import ROOT, sha
from training.v7.common import REPORTS
from training.v7.evaluation import ARMS
from training.v7.fixtures import load_cases
from training.v7.verify_results import verify


def table_rows(summary, recovery):
    rows = [
        "| Metric | V6.1 weights · old flow | V6.1 weights · new flow | Base · new flow | V7 weights · new flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key, denominator in [
        ("Complete business tasks", "task_success", "tasks"),
        ("Explicit conditional grants recognized", "true_permission_yes", "authorized_tasks"),
        ("Unsupported affirmative permissions ↓", "unsupported_permission_yes", "non_authorized_tasks"),
        ("Available exchanges", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("Authorized stockout returns", "fallback_success", "fallback_tasks"),
        ("Required model handoffs", "handoff_success", "handoff_tasks"),
        ("Missing-email resolution", "missing_email_success", "missing_email_tasks"),
        ("Raw initial reason correct", None, "tasks"),
        ("Initial raw parse failures ↓", "initial_raw_parse_failures", "tasks"),
        ("Initial invalid-enum tasks (all fields) ↓", "initial_enum_error_tasks", "tasks"),
        ("Invalid reason tasks in first output ↓", "initial_reason_enum_error_tasks", "tasks"),
        ("All seven raw fields jointly correct", "intent_joint_correct", "intent_tasks"),
        ("Required answers obtained", "clarifications_obtained", "required_clarifications"),
        ("Schema/enum-valid actions", "schema_valid", "model_calls"),
        ("Confirmation preserved", "confirmation_preserved", "tasks"),
    ]:
        values = []
        for variant in ARMS:
            record = summary[variant]
            numerator = (
                record["intent_field_correct"]["reason"]
                if key is None
                else recovery[variant][key]
                if key == "initial_reason_enum_error_tasks"
                else record[key]
            )
            values.append(f"{numerator}/{record[denominator]}")
        rows.append("| " + label + " | " + " | ".join(values) + " |")
    for label, key in [
        ("Enum-error model attempts ↓", "enum_error_attempts"),
        ("Tasks with a same-turn model enum correction", "enum_recovered_tasks"),
        ("Reason-enum error attempts (any turn) ↓", "reason_enum_errors"),
        ("Tasks with a same-turn reason correction (any turn)", "reason_enum_recovered_tasks"),
        ("Rejected record calls ↓", "record_rejections"),
        ("Unnecessary handoff attempts ↓", "unnecessary_handoff_attempts"),
        ("Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"),
        ("Repeated field attempts ↓", "repeated_question_attempts"),
        ("Repeated questions delivered ↓", "repeated_questions_delivered"),
        ("Tasks with a return attempt when the gold outcome was not return ↓", "unauthorized_return"),
        ("Available-exchange tasks with a return attempt ↓", "exchange_as_return"),
        ("Total model calls", "model_calls"),
    ]:
        rows.append("| " + label + " | " + " | ".join(str(summary[v][key]) for v in ARMS) + " |")
    return "\n".join(rows)


def protocol_table(protocol):
    """Keep call, proposal, commit and final-order units distinct from business task scores."""
    rows = [
        "| Supplemental exchange-origin consent audit | V6.1 weights · old flow | V6.1 weights · new flow | Base · new flow | V7 weights · new flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, numerator, denominator in [
        ("Audited exchange-origin tasks / all tasks", "gold_exchange_tasks", "tasks"),
        (
            "Raw return calls without verified scoped YES / all return calls ↓",
            "return_attempts_without_scoped_yes",
            "raw_return_attempts",
        ),
        (
            "Created return proposals without verified scoped YES / all return proposals ↓",
            "return_proposals_without_scoped_yes",
            "return_proposals",
        ),
        (
            "Committed returns without verified scoped YES / all committed returns ↓",
            "committed_returns_without_scoped_yes",
            "committed_returns",
        ),
        (
            "Final returned orders without a verified commit / all final returned orders ↓",
            "final_returns_without_verified_commit",
            "final_return_orders",
        ),
    ]:
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{protocol['summary'][v][numerator]}/{protocol['summary'][v][denominator]}" for v in ARMS
            )
            + " |"
        )
    rows.append(
        "| Rejected raw return calls without verified scoped YES | "
        + " | ".join(
            str(
                sum(
                    a["raw_return_attempt"]
                    and not a["scoped_yes_verified"]
                    and not a["return_proposal_created"]
                    and bool(a["backend_error"])
                    for r in protocol["records"]
                    if r["variant"] == v
                    for a in r["attempts"]
                )
            )
            for v in ARMS
        )
        + " |"
    )
    rows.append(
        "| Declined return proposals | "
        + " | ".join(str(protocol["summary"][v]["declined_return_proposals"]) for v in ARMS)
        + " |"
    )
    return "\n".join(rows)


def protocol_observation(counts):
    """Report observed exposure without turning an incomplete safety audit into certification."""
    observed = (
        counts["return_proposals_without_scoped_yes"],
        counts["committed_returns_without_scoped_yes"],
        counts["final_returns_without_verified_commit"],
    )
    if any(observed):
        return (
            "The candidate has observed exchange-origin consent violations: "
            f"{observed[0]} created return proposals, {observed[1]} committed returns and "
            f"{observed[2]} final returned orders lack the required verified evidence. "
            "These failures remain visible even if the frozen business gate passes."
        )
    return (
        "No candidate proposals, commits or final returned orders lacking the audited evidence "
        "were observed in this small exchange-origin cohort. The known operation-rewrite bypass "
        "remains in the frozen controller; zero observations do not establish execution safety."
    )


def legacy_retention_section(root=ROOT):
    """Persisted first-output development evidence only; no weights, model or service."""
    folder = root / "reports/v7"
    plan_path = folder / "legacy-intent-regression-plan.json"
    plan = json.loads(plan_path.read_text())
    document = json.loads((folder / "legacy-intent-regression.json").read_text())
    assert document["status"] == "complete" and document["plan_sha256"] == sha(plan_path)
    assert document["source_inputs_sha256"] == plan["source_inputs_sha256"]
    for name, digest in plan["source_inputs_sha256"].items():
        assert sha(root / name) == digest, name
    cases = [
        c
        for c in json.loads((root / "training/v6_1/datasets/cases.json").read_text())
        if c["split"] == "test"
    ]
    counts = Counter(c["initial_message"] for c in cases)
    assert len(cases) == len(document["records"]) == len(document["archived_records"]) == 48
    assert len(counts) == 46 and sorted(counts.values()) == [1] * 44 + [2, 2]
    assert not document["checkpoint_selection_modified"] and not document["business_gate_modified"]
    old, new = (document["summary"][key] for key in ("archived_v6_1", "current_v7"))
    rows = [
        "| Development retention on published V6.1 cases | Archived V6.1 first output | V7 first output |",
        "| --- | ---: | ---: |",
    ]
    for label, key, denominator in [
        ("Explicit conditional grants recognized", "true_permission_yes", "authorized_tasks"),
        ("Unsupported affirmative permissions ↓", "unsupported_permission_yes", "non_authorized_tasks"),
        ("First-output reason-enum errors ↓", "initial_reason_enum_error_tasks", "cohort_tasks"),
        ("All seven raw fields jointly correct", "intent_joint_correct", "cohort_tasks"),
        ("First-output parse failures ↓", "initial_raw_parse_failures", "cohort_tasks"),
    ]:
        rows.append(f"| {label} | {old[key]}/{old[denominator]} | {new[key]}/{new[denominator]} |")
    for label, key in [
        ("Raw reason correct", "reason"),
        ("Raw replacement option correct", "requested_option"),
    ]:
        rows.append(
            f"| {label} | {old['intent_field_correct'][key]}/48 | {new['intent_field_correct'][key]}/48 |"
        )
    return (
        "## Separate development retention on legacy cases\n\n"
        "After checkpoint selection and the complete main test, one initial response was recorded for each of the **48 previously published V6.1 cases**, representing **46 unique initial prompts and two duplicate pairs**. The archived first raw response is compared with the current first raw response; no continuation, tool execution, correction or retry is scored.\n\n"
        + "\n".join(rows)
        + "\n\nBoth **workflow and weights changed**, so this is combined-system **development retention**, not a fresh benchmark, independent held-out evidence or an isolated SFT effect. The duplicate prompts remain in the 48-case denominator and are not 48 independent prompt observations. Replacement-option accuracy regressed; the raw failures remain visible. These results are not pooled with the new test and do not change checkpoint selection or the failed business gate. [Raw responses and archived baseline](reports/v7/legacy-intent-regression.json) · [Pre-inference retention plan](reports/v7/legacy-intent-regression-plan.json)\n"
    )


def colab_evidence_paragraph(root=ROOT):
    """Report verified cloud training separately from the interrupted new comparison."""
    folder = root / "reports/colab-mlx"
    result = json.loads((folder / "reports/v7/preflight.json").read_text())
    receipt = json.loads((folder / "colab-execution/preflight.json").read_text())
    initialization = json.loads((folder / "reports/v7/initialization-audit.json").read_text())["stages"][
        "preflight"
    ]
    assert receipt["status"] == "completed" and result["completed_updates"] == 1 and result["preflight_only"]
    assert result["cloud_compute_used"] and not result["preflight_weights_reused"]
    assert initialization["exact_tensor_equality"] and initialization["source_tensor_count"] == 112
    assert initialization["source_parameter_count"] == result["trainable_parameters"] == 3_670_016
    metadata = result["execution_metadata"]
    assert metadata["source_manifest_sha256"] == sha(folder / "colab-source-manifest.json")
    assert receipt["source_manifest_sha256"] == metadata["source_manifest_sha256"]
    before, after = result["metrics"][0]["loss"], result["metrics"][0]["post_update_same_sample_loss"]
    assert 0 <= after < before and result["adapter_l2_change"] > 0
    torch_folder = root / "reports/cuda-torch/probe-003"
    smoke_path = torch_folder / "smoke-4/report.json"
    smoke = json.loads(smoke_path.read_text())
    verification = json.loads((torch_folder / "smoke-4/verification.json").read_text())
    launch = json.loads((torch_folder / "formal-launch.json").read_text())
    decoder = json.loads((torch_folder / "decoder-check.json").read_text())
    cpu_suite = ET.parse(torch_folder / "cpu-tests.xml").getroot().find("testsuite").attrib
    assert cpu_suite["failures"] == cpu_suite["errors"] == cpu_suite["skipped"] == "0"
    assert cpu_suite["tests"] == "36"
    assert decoder["status"] == "passed" and len(decoder["checks"]) == 3
    assert all(c["logical_bf16_exact"] and c["max_abs_error"] == 0 for c in decoder["checks"])
    assert smoke["status"] == "completed" and smoke["completed_true_optimizer_updates"] == 4
    assert (
        smoke["all_gradients_finite"]
        and smoke["smoke_weights_discarded"]
        and not smoke["smoke_weights_reused"]
    )
    assert verification["report_sha256"] == sha(smoke_path)
    assert verification["plan_sha256"] == smoke["plan_sha256"] == sha(torch_folder / "smoke-4/plan.json")
    tensor = verification["tensor_audit"]
    assert tensor["available"] and tensor["numeric_values_independently_checked"]
    assert tensor["predecessor_to_saved_initial_exact_after_transpose"] and tensor["all_values_finite"]
    assert tensor["tensor_count"] == 112 and tensor["parameter_count"] == 3_670_016
    assert abs(tensor["independent_final_adapter_l2_change"] - smoke["adapter_l2_change"]) < 1e-10
    assert launch["smoke_report_sha256"] == sha(smoke_path)
    assert (
        launch["free_gpu_only"]
        and launch["fresh_predecessor_optimizer"]
        and not launch["smoke_weights_reused"]
    )
    formal_path = torch_folder / "train-1344/report.json"
    formal = json.loads(formal_path.read_text())
    formal_verification = json.loads((torch_folder / "train-1344/verification.json").read_text())
    assert formal["status"] == "completed" and formal["kind"] == "train"
    assert formal["completed_true_optimizer_updates"] == len(formal["metrics"]) == 1344
    assert [m["true_optimizer_updates"] for m in formal["metrics"]] == list(range(1, 1345))
    assert formal["all_gradients_finite"] and formal["smoke_weights_discarded"]
    assert not formal["smoke_weights_reused"] and formal["sampling"]["all_physical_rows_used"]
    assert formal["sampling"]["unique_rows_sampled"] == formal["sampling"]["physical_rows"] == 824
    assert formal_verification["report_sha256"] == sha(formal_path)
    assert (
        formal_verification["plan_sha256"]
        == formal["plan_sha256"]
        == sha(torch_folder / "train-1344/plan.json")
    )
    final_tensor = formal_verification["tensor_audit"]
    assert final_tensor["available"] and final_tensor["numeric_values_independently_checked"]
    assert (
        final_tensor["predecessor_to_saved_initial_exact_after_transpose"]
        and final_tensor["all_values_finite"]
    )
    assert final_tensor["tensor_count"] == 112 and final_tensor["parameter_count"] == 3_670_016
    assert abs(final_tensor["independent_final_adapter_l2_change"] - formal["adapter_l2_change"]) < 1e-10
    assert final_tensor["files_sha256"]["final_adapter"] == formal["final_adapter"]["sha256"]
    interruption = json.loads((root / "reports/cuda-eval/interruption.json").read_text())
    assert interruption["status"] == "interrupted"
    assert interruption["analysis_plan_sha256"] == sha(root / "reports/cuda-eval/analysis-plan.json")
    counts = interruption["last_verified_completed_by_arm"]
    observed = interruption["last_verified_completed_cases"]
    assert observed == sum(counts.values()) < 96
    return (
        f"An actual **one-update MLX CUDA preflight** completed on the user-selected free Colab **{metadata['gpu']['name']}**, Python **{metadata['platform']['python']}**, CUDA **{metadata['platform']['cuda_toolkit']}**, MLX **{metadata['package_versions']['mlx']}**. Same-sample loss changed **{before:.8f} → {after:.8f}**, adapter L2 change was **{result['adapter_l2_change']:.5f}**, and a separate CPU audit verified all **112 tensors / 3,670,016 initial parameters** exactly. The cold optimizer-loop check took **{result['elapsed_seconds']:.2f} seconds**; this demonstrates that this one-step path ran, not a speedup, full cloud training, new accuracy or bitwise Metal/CUDA equivalence. [Actual preflight](reports/colab-mlx/reports/v7/preflight.json) · [Initialization audit](reports/colab-mlx/reports/v7/initialization-audit.json) · [Execution identity](reports/colab-mlx/colab-execution/preflight.json).\n\n"
        f"The separate **PyTorch/PEFT FP16 smoke completed four true optimizer updates** on the same free T4 runtime. **{cpu_suite['tests']} Colab CPU checks** and strict comparisons on **three real source tensor slices** passed; an independent tensor read verified the exact **112 tensors / 3,670,016** imported initial parameters and final adapter L2 change **{tensor['independent_final_adapter_l2_change']:.7f}**. The synchronized four-update run took **{smoke['elapsed_loop_gpu_sync_seconds']:.6f} seconds**; CUDA allocator peaks were **{smoke['peak_cuda_allocated_bytes'] / 1e9:.5f} GB allocated / {smoke['peak_cuda_reserved_bytes'] / 1e9:.5f} GB reserved**, distinct from total GPU/OS memory. Timing excludes model loading/tokenization and includes report/export/audit overhead; four smoke updates do not establish full-run throughput or stability. [Actual smoke](reports/cuda-torch/probe-003/smoke-4/report.json) · [Independent tensor verification](reports/cuda-torch/probe-003/smoke-4/verification.json) · [Strict decode checks](reports/cuda-torch/probe-003/decoder-check.json).\n\n"
        f"The separate **formal training completed 1,344 true optimizer updates** at **{formal['completed_at_utc']}**, starting from the predecessor and a fresh optimizer without smoke-weight reuse. It sampled **{formal['sampling']['unique_rows_sampled']}/{formal['sampling']['physical_rows']} unique physical rows** from **{formal['sampling']['unique_cases_sampled']} source interactions**. The synchronized run wall time was **{formal['elapsed_loop_gpu_sync_seconds']:.6f} seconds**; CUDA allocator peaks were **{formal['peak_cuda_allocated_bytes'] / 1e9:.5f} GB allocated / {formal['peak_cuda_reserved_bytes'] / 1e9:.5f} GB reserved**. Independent CPU reads verified all **112 tensors / 3,670,016 initial parameters**, finite final values and final adapter L2 change **{final_tensor['independent_final_adapter_l2_change']:.7f}**; final adapter SHA256 is `{formal['final_adapter']['sha256']}`. [Complete formal report](reports/cuda-torch/probe-003/train-1344/report.json) · [Pre-load plan](reports/cuda-torch/probe-003/train-1344/plan.json) · [Independent numeric verification](reports/cuda-torch/probe-003/train-1344/verification.json).\n\n"
        f"The new **48-case, two-arm before/after evaluation was interrupted** when Colab disconnected and no active runtime remained. Its **last verified snapshot contains {observed}/96 records** (before {counts['predecessor_fp16']}/48; after {counts['cloud1344_fp16']}/48); later actual progress is unknown. An issued call has no observed completion, so automatic replay is disabled. There is no complete comparison or six-group final chart set; no new model-quality score is reported here. [Interruption and retained evidence](reports/cuda-eval/README.md) · [Independent snapshot audit](reports/cuda-eval/interrupted-snapshot-audit.json). Its fixed-final-checkpoint [pre-inference plan](reports/cuda-eval/analysis-plan.json) keeps those results separate from the original V7 scores and failed business gate. The port streams the same 4bit-derived base into FP16 storage and imports the exact V6.1 adapter; backend, precision, checkpointing and the zero-weight projection window differ from MLX. This is **not an isolated GPU hardware speed comparison**, the local QLoRA result or evidence of model-quality improvement. Training completion does not establish accuracy or acceleration."
    )


def main():
    audit = verify()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    train = json.loads((REPORTS / "training.json").read_text())
    data = json.loads((REPORTS / "data-summary.json").read_text())
    values = json.loads((ROOT / "docs/figures/v7/values.json").read_text())
    suite = ET.parse(REPORTS / "test-results.xml").getroot().find("testsuite").attrib
    assert suite["failures"] == suite["errors"] == "0"
    passed = int(suite["tests"]) - int(suite.get("skipped", "0"))
    skipped = int(suite.get("skipped", "0"))
    notebook_qa = json.loads((REPORTS / "notebook-verification.json").read_text())
    assert notebook_qa["nbformat_valid"] and notebook_qa["executed_mode"] == "inspect"
    assert not notebook_qa["cuda_executed"] and notebook_qa["model_calls"] == 0
    assert notebook_qa["notebook_sha256"] == sha(ROOT / "training/colab/resolvedesk_cuda.ipynb")
    tests = f"{passed} engineering tests passed locally" + (f" ({skipped} skipped)" if skipped else "")
    summary = report["summary"]
    recovery = values["recovery_diagnostics"]
    protocol = values["protocol_audit"]
    protocol_new = protocol["summary"][ARMS[3]]
    old, flow, new = (summary[v] for v in [ARMS[0], ARMS[1], ARMS[3]])
    count = new["tasks"]
    gate = report["release_gate"]
    checks = "\n".join(f"- {k}: {'pass' if value else 'fail'}" for k, value in gate["checks"].items())
    selected = train["best_checkpoint"]["iteration"]
    coverage = audit["selected_checkpoint_training_coverage"]
    failures = [r["case_id"] for r in report["records"] if r["variant"] == ARMS[3] and not r["task_success"]]
    validation_rows = [
        "| Validation checkpoint | Task success | Explicit grants | Unsupported YES ↓ | Initial enum errors ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in train["validation_candidates"]:
        s = candidate["summary"]
        label = str(candidate["iteration"]) + (" (selected)" if candidate["iteration"] == selected else "")
        validation_rows.append(
            f"| {label} | {s['task_success']}/{s['tasks']} | {s['true_permission_yes']}/{s['authorized_tasks']} | {s['unsupported_permission_yes']}/{s['non_authorized_tasks']} | {s['initial_enum_error_tasks']}/{s['tasks']} |"
        )
    live = "Actual app/adapter validation probes are pending."
    if (REPORTS / "live-api-verification.json").exists():
        probe = json.loads((REPORTS / "live-api-verification.json").read_text())
        if probe["status"] == "complete":
            live = f"Actual app → selected MLX adapter probes: **{sum(r['outcome_correct'] for r in probe['records'])}/{len(probe['records'])}** correct. Every validation world was included and all failures retained in [API evidence](reports/v7/live-api-verification.json). These smokes allow at most four requested-field replies, with twelve calls per API advance, and are excluded from held-out accuracy."
    figure_specs = [
        ("evaluation", "Business recovery, handoffs and frozen attempt metrics"),
        ("families", "All authored scenario families"),
        ("intent", "Raw first-output field accuracy"),
        ("consent", "First-output permission metrics and supplemental exchange-origin execution audit"),
        ("recovery", "Raw enum failures, subsequent model repair and repeated questions"),
        ("confusion", "Operation confusion including unfinished tasks"),
        ("training", "Real optimizer loss and validation-only checkpoint selection"),
    ]
    figures = "\n\n".join(f"![{title}](docs/figures/v7/{name}.png)" for name, title in figure_specs)
    initializer = json.loads((REPORTS / "initialization-audit.json").read_text())["stages"]["formal"]
    config = train["config"]
    nvalid, ntrain = len(load_cases("valid")), len(load_cases("train"))
    pairs = values["paired"]
    legacy = legacy_retention_section()
    colab = colab_evidence_paragraph()
    fence = chr(96) * 3
    text = f"""# ResolveDesk · Retail agent with auditable model repair

An English retail agent using local Qwen3, observable tool calls and customer-confirmed transactions. Orders, payments, customer accounts and handoffs are simulated.

**Round 7 targets missed explicit conditional authorization, invalid `reason` values, unnecessary handoffs and repeated questions.** On **{count} new synthetic interactions in {count} distinct order worlds**, the preserved V6.1 system versus this candidate changes full task success **{old["task_success"]}/{count} → {new["task_success"]}/{count}**, explicit permission recall **{old["true_permission_yes"]}/{old["authorized_tasks"]} → {new["true_permission_yes"]}/{new["authorized_tasks"]}**, invalid reason values in the first output **{recovery[ARMS[0]]["initial_reason_enum_error_tasks"]}/{count} → {recovery[ARMS[3]]["initial_reason_enum_error_tasks"]}/{count}**, necessary handoffs **{old["handoff_success"]}/{old["handoff_tasks"]} → {new["handoff_success"]}/{new["handoff_tasks"]}**, and missing-email resolution **{old["missing_email_success"]}/{old["missing_email_tasks"]} → {new["missing_email_success"]}/{new["missing_email_tasks"]}**.

The candidate **{"passes" if gate["promoted"] else "does not pass"}** the frozen business acceptance gate. Unsupported affirmative predictions are **{new["unsupported_permission_yes"]}/{new["non_authorized_tasks"]}**, available exchanges **{new["fulfillable_exchange_success"]}/{new["fulfillable_exchange_tasks"]}**, stockout returns **{new["fallback_success"]}/{new["fallback_tasks"]}**, unnecessary handoffs executed **{new["unnecessary_handoff_executed"]}**, and repeated-field attempts/deliveries **{new["repeated_question_attempts"]}/{new["repeated_questions_delivered"]}**. **{new["initial_raw_parse_failures"]} first outputs did not yield a complete raw request.** Absence of an affirmative prediction is not proof of correct interpretation or execution authorization. {protocol_observation(protocol_new)}

This is a small self-authored test sharing policy, templates and vocabulary with development; it is not an official benchmark or production reliability result. Model repair does not erase the original error. The selected adapter runs separately through MLX; no Ollama/GGUF merge is claimed.

## Four controlled arms

{table_rows(summary, recovery)}

The **same V6.1 weights** under the old versus new workflow measure controller/prompt effects: **{old["task_success"]}/{count} → {flow["task_success"]}/{count}**. The **same new workflow** with V6.1 versus V7 weights measures the combined new data, loss weighting and sampling changes: **{flow["task_success"]}/{count} → {new["task_success"]}/{count}**. Under that fixed workflow, explicit grants change **{flow["true_permission_yes"]}/{flow["authorized_tasks"]} → {new["true_permission_yes"]}/{new["authorized_tasks"]}** and unsupported affirmative predictions **{flow["unsupported_permission_yes"]}/{flow["non_authorized_tasks"]} → {new["unsupported_permission_yes"]}/{new["non_authorized_tasks"]}**. There is no ablation isolating each training choice; deterministic workflow recovery is not learned model accuracy.

All four arms use the pinned base/tokenizer, unconstrained greedy MLX decoding, {config["generation_max_tokens"]} maximum output tokens and **{config["max_agent_steps"]} total model calls per task**, including every rejected call. Scripted replies are supplied only after the requested field is actually asked. There are no test warmups, hidden corrective model calls or retries to obtain a pass. Historical results use different data and are not pooled. Serial wall times include resource effects and do not establish a speedup.

Paired outcomes against the complete previous system: **{pairs["end_to_end"]["improved"]} improved, {pairs["end_to_end"]["regressed"]} regressed**. With the new flow fixed: **{pairs["training"]["improved"]} improved, {pairs["training"]["regressed"]} regressed** after SFT. Current failed tasks: {", ".join(failures) or "none in this small suite"}. [Every raw trajectory](reports/v7/evaluation.json), [all failures](reports/v7/failure-analysis.json) and the [per-case CSV](docs/figures/v7/per-case.csv) remain available.

{legacy}

## Engineering contributions and retained protections

- Added recoverable structural/enum validation feedback. A rejected initial action keeps its raw arguments, field errors and source turn. The model must issue its own corrected `record_request`; repeated invalid actions remain bounded and consume the same task budget. A bad `reason` is not silently turned into a successful prediction.
- Added explicit clarification state and validated reply bindings. A usable answer remains available to the model; forgetting it triggers record-repair feedback before another question. No hidden corrective tool call or automatic handoff executes on the model's behalf.
- Preserved bounded contact parsing and exact catalog/payment reply validation. When payment remains `unknown` in the model record, the gate requests it before fallback consent. An incorrect non-unknown prediction can skip that question: one retained API failure inferred `original` from “original purchase” and missed a required payment clarification before its confirmed return. Emails, negation and new instructions remain distinguishable. Email lookup in this sandbox is identification, not authentication.
- Preserved the scoped-consent checks on the recorded-exchange path: a predicted `yes` requests a fallback question and a later whole-message customer `YES` binds the observed order, replacement, request and revision. A supplemental audit checks actual execution evidence because a model rewriting the accepted operation to direct return can bypass that path; the frozen controller retains this limitation.
- Preserved separate transaction confirmation, proposal revocation on unsolicited new instructions and exactly-once commit receipts. Fallback consent does not itself commit a return. Historical sessions keep their controller and pending proposals.
- Kept the **first model output** as the raw interpretation across every arm, including rejected non-enum values. Later repairs never replace raw first-output accuracy. Recovery counts require a subsequent accepted record on the same actual customer turn; they are separate from successful business outcomes.

The gate sees actual customer messages and observed tools, never benchmark family names, expected answers or gold intent. `exchange_as_return` counts available-exchange tasks with a return attempt. The historical `unauthorized_return` field counts exchange-origin tasks whose gold business outcome is not return but that contain a return attempt; it does not audit all execution authorization. Repeated-field and record-rejection metrics count calls. False-positive permission counts are reported alongside positive recall and raw parse failures.

## Supplemental scoped-consent audit

{protocol_table(protocol)}

This separately [predeclared audit](reports/v7/protocol-audit-plan.json) reconciles **{protocol["records_audited"]} arm/task records**, including every original exchange in every arm. It reconstructs successful scoped fallback questions and subsequent exact `YES` replies from actual event prefixes, matching the observed order, unavailable replacement, payment, request and revision. Gold intent selects the exchange-origin cohort only. Calls, created proposals, commits and final returned orders are different units; blocked attempts and declined proposals remain visible. A final returned order is accepted by this audit only when it has a matching scoped-YES-verified commit.

{protocol_observation(protocol_new)} The source, plan, input hashes, each row and all counts are bound in [audit evidence](reports/v7/protocol-audit.json), [figure values](docs/figures/v7/values.json) and [independent reconciliation](reports/v7/verification.json). No model call or controller correction is performed by this audit. It preserves the frozen business scores, checkpoint selection and business acceptance gate. Passing that gate does not override a supplemental consent failure.

Known limit: a model can change the accepted operation after a clarification, moving an original exchange onto the direct-return path. This round measures that exposure without modifying the frozen workflow. The supplemental audit uses frozen request projections rather than the decision gate; it excludes semantic authorization of initially direct-return tasks and unrestricted real conversations where customers genuinely change operation. Its counts are not an all-path authorization guarantee.

## Real local QLoRA continuation

One new run continued the selected V6.1 adapter on **{config["model_repo"]}**, pinned revision **{config["model_revision"]}**. It updated **{train["trainable_parameters"]:,} parameters**, rank {config["lora_parameters"]["rank"]} in the last {config["num_layers"]} layers, batch {config["batch_size"]}, Adam {config["learning_rate"]}, seed {config["seed"]}. Completed **{train["completed_updates"]} optimizer updates**; validation selected **{selected}** from exactly {", ".join(map(str, config["checkpoint_candidates_completed_updates"]))} before any test inference.

A [CPU-only initialization audit](reports/v7/initialization-audit.json) compared all **{initializer["source_tensor_count"]} adapter tensors / {initializer["source_parameter_count"]:,} parameters** in the actual saved pre-optimizer snapshots with the predecessor. Both preflight and formal initialization match exact tensor keys, shapes, dtypes and finite values. NumPy/safetensors reads no GPU runtime and exports no tensor contents. The one-step preflight weights are discarded before the separately initialized formal run.

Training contains **{ntrain} unique source interactions** in {ntrain} distinct worlds. The reference pipeline produced **{data["raw_train_rows_before_deduplication"]} raw snapshots** and physically removed **{data["duplicate_training_pairs_removed"]} exact duplicate prompt/completion pairs**, leaving **{data["rows"]["train"]} unique SFT pairs**. These include {data["train_sample_roles"]["initial_record"]} initial records, {data["train_sample_roles"]["record_recovery"]} record-repair targets and {data["train_sample_roles"]["tool_action"]} other tool decisions. **{data["recovery_trajectories"]} safe recovery trajectories** supplement the clean references; wrong simulated actions appear only in input state, never as supervised targets.

The new validation and test each contain **{nvalid} unique interactions and distinct order worlds**, with unique initial prompts, split-specific authored phrasing and entities disjoint from training and every prior round. Reference trajectories all pass the preserved business grader before training. These remain authored synthetic cases, not independent human annotation or real customer satisfaction.

Sampler: **50% initial-record/record-repair targets and 50% execution/reply targets**, with shuffled cycles inside each pool. The full run sampled **{train["sampling"]["unique_rows_sampled"]}/{train["sampling"]["physical_rows"]} physical rows** and **{train["sampling"]["unique_cases_sampled"]}/{ntrain} source cases**. The **selected {selected}-update checkpoint** saw **{coverage["unique_rows_sampled"]}/{coverage["physical_training_rows"]} unique physical rows** and **{coverage["unique_cases_sampled"]}/{ntrain} source cases**. Repeated optimization samples are not new training data, and full-run coverage must not be attributed to an earlier selected checkpoint.

Full-completion cross-entropy weights prompt/padding 0, ordinary completion tokens 1, the `fallback_return` value **{train["token_audit"]["permission_value_weight"]:g}** and the `reason` value **{train["token_audit"]["reason_value_weight"]:g}**, using audited tokenizer character offsets. There are **{train["token_audit"]["permission_value_tokens"]} permission-value target tokens** and **{train["token_audit"]["reason_value_tokens"]} reason-value target tokens** in the physical dataset. All targets are retained; the {config["projection_window"]}-position output projection keeps full context. Preflight compares with full output projection and discards its updated weights. Weighted loss is not directly comparable with prior uniform-loss runs.

{chr(10).join(validation_rows)}

Optimizer-loop time: **{train["elapsed_seconds"] / 60:.2f} minutes**. Peak active MLX allocation: **{train["peak_active_memory_bytes"] / 1e9:.2f} GB**, excluding cache, OS and other processes. Validation selection prioritizes safety, false-positive control, explicit-grant recall and balanced recovery outcomes according to the frozen rule. Test outputs never select a checkpoint or change the controller.

[Method and reproduction](training/v7/README.md) · [Frozen source/data manifest](training/v7/datasets/manifest.json) · [Actual weight-update evidence](reports/v7/updates.json) · [Independent replay and sampling audit](reports/v7/verification.json)

{figures}

## Frozen business acceptance

{checks}

These are predeclared business prototype targets, separate from the supplemental execution-consent findings above. They do not establish production safety. All {count} test interactions and all {len(report["records"])} arm/task records are retained. No external benchmark score, business savings, causal attribution to an individual training choice or population confidence interval is claimed.

## Run and inspect

{fence}bash
uv sync --frozen
uv run resolvedesk serve --port 8771
# After provisioning the pinned local base and selected adapter:
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v7.serve
{fence}

Open [ResolveDesk](http://127.0.0.1:8771/) and choose the round 7 adapter profile. The selected weights run on the local MLX endpoint. Default Ollama sessions retain their separate integration; historical sessions retain their original controller. Public app fixtures exclude scripted replies, expected outcomes and test worlds. Weights, databases and credentials are excluded from Git. Other Ollama caches are allowed to expire naturally and are never unloaded by training or evaluation.

An optional [Colab notebook](training/colab/resolvedesk_cuda.ipynb) and [CUDA reproduction guide](docs/colab.md) retain the frozen inputs in a separate workspace. All **{notebook_qa["code_cells_executed"]} default inspect-mode cells** executed locally without model calls, with format and rendered-preview checks recorded in [notebook QA](reports/v7/notebook-verification.json). The handoff package itself performs no automatic upload or cloud launch; an existing local checkpoint cannot resume its optimizer on Colab. Cloud reproductions remain separate from the local benchmark evidence.

{colab}

![Actual evaluation page](docs/evaluation-v7-browser.jpg)

**{tests}.** Seven source-backed figures are independently reconciled against the raw report. {live} GitHub CI remains configured; no remote run is claimed.

## Attribution and resume wording

Retail policy attribution: [Sierra Research tau-bench v2.0.0](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0), pinned to `989d3285ae669e49503324bdb9981bf14fccedeb`. Policy and MIT license remain in [third_party/tau-bench](third_party/tau-bench/README.md). ResolveDesk implements its own limited tools, controller, synthetic data, training and grader. Qwen retains Apache-2.0. No official benchmark score is claimed.

> Built a local Qwen3 retail agent with observable consent checks, recoverable enum validation and idempotent customer-confirmed transactions. Continued {train["trainable_parameters"] / 1e6:.2f}M-parameter QLoRA using deduplicated training pairs, field-weighted completion supervision and validation-only checkpoint selection. On {count} fresh synthetic interactions, four controlled arms separated workflow changes from the training bundle; with the workflow fixed, task success changed {flow["task_success"]}/{count} → {new["task_success"]}/{count}, explicit-grant recognition {flow["true_permission_yes"]}/{flow["authorized_tasks"]} → {new["true_permission_yes"]}/{new["authorized_tasks"]}, and first-output reason-enum error tasks {recovery[ARMS[1]]["initial_reason_enum_error_tasks"]}/{count} → {recovery[ARMS[3]]["initial_reason_enum_error_tasks"]}/{count}. Independently audited scoped consent on exchange-origin returns: {protocol_new["return_proposals_without_scoped_yes"]}/{protocol_new["return_proposals"]} proposals and {protocol_new["committed_returns_without_scoped_yes"]}/{protocol_new["committed_returns"]} commits lacked verified scoped YES. Preserved all failures and verified through {passed} engineering tests.

Keep the synthetic scope, comparator, business acceptance outcome, supplemental consent findings, raw-versus-repaired distinction and remaining failures alongside any resume claim. [Round 6](docs/round-6.md), [round 5](docs/round-5.md), [round 4](docs/round-4.md), [round 3](docs/round-3.md), [round 2](docs/round-2.md) and [round 1](docs/round-1.md) remain archived.
"""
    (ROOT / "README.md").write_text(text)
    print("README generated from reconciled actual round-seven evidence")


if __name__ == "__main__":
    main()
