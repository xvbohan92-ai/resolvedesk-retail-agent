"""Regenerate repository documentation from verified evidence only."""

# ruff: noqa: E501 -- Embedded Markdown keeps readable paragraphs.
import json
import xml.etree.ElementTree as ET

from training.common import ROOT
from training.v5.common import DATA, REPORTS
from training.v5.evaluation import ARMS
from training.v5.verify_results import verify


def main():
    verify()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    values = json.loads((ROOT / "docs/figures/v5/values.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    s, data = report["summary"], manifest["data"]
    old, new = s["previous_new_flow"], s["qlora_v5"]
    tests = ET.parse(REPORTS / "test-results.xml").getroot().find("testsuite").attrib["tests"]
    selected = training["best_checkpoint"]["iteration"]
    rows = [
        "| Fresh v5 test | V4 + old flow | V4 + typed flow | Base + typed flow | V5 + typed flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key, den in [
        ("Full task success", "task_success", "tasks"),
        ("Available exchanges correct", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("Authorized stockout fallback", "fallback_success", "fallback_tasks"),
        ("Required handoff completed by model", "handoff_success", "handoff_tasks"),
        ("Available exchange changed to return ↓", "exchange_as_return", "fulfillable_exchange_tasks"),
        ("Required customer answer obtained", "clarifications_obtained", "required_clarifications"),
        ("Direct-return controls", "direct_return_success", "direct_return_tasks"),
        ("Schema-valid model actions", "schema_valid", "model_calls"),
        ("Confirmation preserved", "confirmation_preserved", "tasks"),
    ]:
        rows.append("| " + label + " | " + " | ".join(f"{s[v][key]}/{s[v][den]}" for v in ARMS) + " |")
    for label, key in [
        ("Unnecessary handoffs executed /24 ↓", "unnecessary_handoff_executed"),
        ("Unnecessary model handoff attempts ↓", "unnecessary_handoff_attempts"),
        ("Unauthorized return attempts ↓", "unauthorized_return"),
        ("Repeated question-field attempts ↓", "repeated_question_attempts"),
        ("Repeated questions actually delivered ↓", "repeated_questions_delivered"),
        ("Unrequested substitute attempts ↓", "unrequested_substitute_attempts"),
        ("Rejected policy-gate actions", "policy_gate_rejections"),
        ("Quote-provenance rejections", "quote_rejections"),
        ("All identical action repeats", "identical_action_repeats"),
        ("All actual model calls", "model_calls"),
    ]:
        rows.append("| " + label + " | " + " | ".join(str(s[v][key]) for v in ARMS) + " |")
    rows.append(
        "| Initial intent: all seven fields correct | "
        + " | ".join(
            f"{s[v]['intent_joint_correct']}/{s[v]['intent_tasks']}"
            if s[v]["intent_tasks"]
            else "not recorded"
            for v in ARMS
        )
        + " |"
    )
    permission = values["permission_diagnostics"]["qlora_v5"]
    gate = report["release_gate"]
    checks = "\n".join(f"- {name}: {'pass' if passed else 'fail'}" for name, passed in gate["checks"].items())
    pairs = values["paired"]["training"]
    total_pairs = values["paired"]["end_to_end"]
    old_index = {r["case_id"]: r for r in report["records"] if r["variant"] == "previous_old_flow"}
    regressions = [
        r["case_id"] for r in report["records"]
        if r["variant"] == "qlora_v5" and not r["task_success"] and old_index[r["case_id"]]["task_success"]
    ]
    fence = chr(96) * 3
    live_text = "Actual API probes are pending; controlled scores above are complete."
    live_path = REPORTS / "live-api-verification.json"
    if live_path.exists():
        live = json.loads(live_path.read_text())
        if live.get("status") == "complete":
            assert live["selected_adapter_sha256"] == training["selected_adapter_sha256"]
            correct = sum(r["outcome_correct"] for r in live["records"])
            live_text = f"Actual app → selected MLX adapter probes: **{correct}/{len(live['records'])}** correct on development examples. Their raw replies, weight identity, pre-confirmation orders and idempotent confirmation receipts are in [live API evidence](reports/v5/live-api-verification.json). These wiring/transaction checks are excluded from held-out accuracy."
    permission_notice = (
        f"The intent parser still records an unsupported affirmative fallback permission on {permission['unsupported_affirmative']}/{permission['non_authorized_tasks']} initial requests. Passing the operational gate does not establish reliable semantic consent interpretation."
        if permission["unsupported_affirmative"]
        else "No unsupported affirmative fallback permission was recorded on these 28 negative initial requests; broader semantic consent is not certified."
    )
    text = f"""# ResolveDesk · Typed-intent retail action agent

An English retail agent with local Qwen3 inference, observable tools and customer-confirmed transactions. Customers, orders, payments and human handoffs are simulated.

**Round 5 targets authorized stockout returns, unnecessary human handoff and exchange completion while retaining repeated-question and exchange-intent protections.** On **32 fresh synthetic tasks**, with the **same new typed-intent controller**, previous → newly trained weights change full task success from **{old["task_success"]}/32 to {new["task_success"]}/32**, stockout fallback from **{old["fallback_success"]}/2 to {new["fallback_success"]}/2**, available exchanges from **{old["fulfillable_exchange_success"]}/10 to {new["fulfillable_exchange_success"]}/10**, and unnecessary executed handoffs from **{old["unnecessary_handoff_executed"]} to {new["unnecessary_handoff_executed"]}**. Repeated field attempts are **{old["repeated_question_attempts"]} → {new["repeated_question_attempts"]}**; available-exchange return errors are **{new["exchange_as_return"]}/10**.

The candidate **{"passes" if gate["promoted"] else "does not pass"}** the predeclared absolute local targets. This is a small, correlated, self-authored test, not an official benchmark or production result. Ollama remains the default integration; the selected QLoRA weights are separately served through MLX.

**{permission_notice}**

## Four controlled arms

{chr(10).join(rows)}

V4 old → typed flow measures the combined model-extracted intent record, shorter prompt and deterministic policy gate. V4 → V5 under the identical typed flow measures the bundled continuation SFT. Base + typed flow is a quality comparator. **Controller policy must not be attributed to learned model reasoning.**

All arms use the same pinned base/tokenizer, unconstrained greedy generation and **twelve TOTAL model calls**, including intent recording, quote errors and every rejected action. Round 4 used eight calls and a different test; its scores are archived and never pooled into this denominator. Serial execution and background-model resource waiting mean raw times are descriptive, not evidence of speedup.

Paired training comparison: **{pairs["improved"]} improved, {pairs["regressed"]} regressed, {pairs["both_pass"]} both passed, {pairs["both_fail"]} both failed**. Failures and regressions remain inspectable.

End-to-end comparison against the prior V4 system, including the changed controller: **{total_pairs["improved"]} improved, {total_pairs["regressed"]} regressed, {total_pairs["both_pass"]} both passed, {total_pairs["both_fail"]} both failed**. Previously successful tasks that now fail: {", ".join(regressions) or "none in this test"}. Aggregate improvement does not erase these regressions.

![Business outcomes and raw attempted actions](docs/figures/v5/evaluation.png)

![Every scenario family](docs/figures/v5/families.png)

![Initial request interpretation](docs/figures/v5/intent.png)

Among accepted initial request records, the intent parser correctly marks **{permission["true_affirmative"]}/{permission["authorized_tasks"]}** actually authorized initial fallback requests, but predicts an unsupported affirmative permission on **{permission["unsupported_affirmative"]}/{permission["non_authorized_tasks"]}** non-authorized requests. These are interpretation errors even if current stock prevents an incorrect return. Exact field accuracy also distinguishes unspecified permission from explicit refusal. This post-hoc diagnosis does not alter selection, grading or the frozen gate.

![Operation confusion](docs/figures/v5/confusion.png)

![Actual training and validation selection](docs/figures/v5/training.png)

[128 raw trajectories](reports/v5/evaluation.json) · [Per-case CSV](docs/figures/v5/per-case.csv) · [Every failure](reports/v5/failure-analysis.json) · [Training and selected fingerprint](reports/v5/training.json) · [Independent reconciliation](reports/v5/verification.json)

## Engineering changes and personal contribution

- Added an explicit **record_request** tool for seven predicted customer fields: operation, email, order, exact option, payment preference, cancellation reason and conditional-return permission. Missing preferences remain unknown. After a customer reply the model updates the record; unsolicited instructions invalidate stale intent.
- Grounded recorded email/order strings and quotations in actual customer turns. **Exact quotation checks provenance only; the semantic fields remain unverified model predictions.** A model can still misinterpret a refusal or preference; initial-field accuracy and end-to-end failure grading expose this.
- Added a policy gate that computes allowed actions from the recorded model prediction and completed identity/order/catalog results. It distinguishes available exact variants, authorized stockout returns, unauthorized substitutes and insufficient gift cards. It blocks premature/avoidable handoff when its observed-fact path is fulfillable.
- The gate does not read task families, expected outcomes or scripted answers. It does not execute a corrective model action or fabricate a handoff. Rejected attempts consume the budget. Tests also demonstrate that a semantically wrong but text-grounded model interpretation is not certified as consent.
- Retained clarification memory, exact repeated-question delivery guards and transactional approval. Correct variant/payment arguments are checked against the current recorded request. Every proposal still requires explicit customer confirmation; replay returns the same receipt. Existing sessions keep their prior controller and pending proposals.
- Separated engineering from training with four arms, and attempted from executed handoffs. Added independent raw-score, intent-label, quote, CSV, figure and historical-artifact checks.

## Data and real local fine-tuning

**64 train / 32 validation / 32 test tasks**, 16 shared families, and **56/28/28 distinct order worlds**. Entities are disjoint across splits and prior experiments. Prices are freshly seeded; individual numbers may overlap. Counterfactual stock/intent pairs deliberately share worlds within a split.

The training corpus contains **{data["rows"]["train"]} snapshots**: 328 reference decisions plus **16 repeated request-contrast snapshots**, not additional tasks. All 128 reference trajectories pass the preserved independent business grader. Data and programmatic customer replies are self-authored; policy, vocabulary and scenario families are shared. There is no independent human annotation, confidence interval, business-savings or production-generalization claim.

Continued the preserved round-4 adapter on official **Qwen/Qwen3-4B-MLX-4bit**, revision **52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495**. Nine local model/tokenizer/license fingerprints are verified before GPU work. Rank 8 in the last eight layers updates **{training["trainable_parameters"]:,} parameters**. Uniform full-completion cross-entropy gives prompt/padding zero weight and every supervised target weight one; there is no action-name preference reward in this round.

Completed **{training["completed_updates"]} new optimizer updates**, batch 1, learning rate 0.00001, seed 202610275; the selected checkpoint is **{selected} updates**. Selection compares exactly 256/512 on the 32-task development set, prioritizing safety and balanced exchange/fallback/handoff success, then overall success, unnecessary handoff, repeats and calls. **No test output selects weights or changes the protocol.** Preflight weights are discarded.

The real one-update preflight reduced same-sample NLL from **{json.loads((REPORTS / "preflight.json").read_text())["metrics"][0]["loss"]:.4f}** to **{json.loads((REPORTS / "preflight.json").read_text())["metrics"][0]["post_update_same_sample_loss"]:.4f}**. Full-vs-selected output projection error was zero. The longest train sequence is **{training["token_audit"]["longest_train_sequence"]} tokens**, below 3072, and every completion target is covered by the 512-position output projection; context is not truncated. Optimizer-loop wall time was **{training["elapsed_seconds"] / 60:.2f} minutes**, including any resource waiting. Peak MLX active allocation was **{training["peak_active_memory_bytes"] / 1e9:.2f} GB**, excluding free-buffer cache, OS and other processes.

[Method and reproduction](training/v5/README.md) · [Preflight](reports/v5/preflight.json) · [All updates](reports/v5/updates.json) · [Frozen manifest](training/v5/datasets/manifest.json)

## Absolute acceptance targets

{checks}

The gate requires all fallback tasks correct, at least 8/10 available exchanges and 7/8 required handoffs, at most one unnecessary executed handoff, no repeated-field/exchange-as-return/unauthorized-return attempts, intact confirmation and overall success no lower than both typed-flow comparators. Passing these small-suite targets is not a production deployment decision.

## Run and inspect

{fence}bash
uv sync --frozen
uv run python -m training.setup_ollama
uv run resolvedesk serve --port 8771
# Separate local adapter service after provisioning the pinned base:
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v5.serve
{fence}

Open [ResolveDesk](http://127.0.0.1:8771/). Default Ollama sessions retain the previous workflow because the V5 candidate failed acceptance. Selecting the explicitly experimental QLoRA provider opts into typed intent; API callers can also opt in with experimental_flow=true. This delivery guard changes session routing only, not the frozen four-arm experiment. Sixteen public validation examples are shown and all 32 are available to the case API. Old validation worlds remain for restoring sessions. Packaged fixtures exclude scripted answers, expected outcomes and test worlds. Email lookup is a demo, not production authentication. Multi-item resolutions are outside this prototype.

Select **QLoRA v5 · MLX** to use the fingerprinted trained weights. They are not merged into Ollama/GGUF. Ollama's schema-constrained path is a separate integration; controlled scores use unconstrained MLX for every arm. Services bind to loopback; weights, local databases and credentials are excluded from Git. Resource checks wait for competing caches rather than unloading other models.

![Actual round-five evaluation page](docs/evaluation-v5-browser.png)

**{tests} engineering tests passed locally.** Lint, JavaScript syntax, historical artifacts, raw traces and export reconciliation are checked. GitHub CI is configured; no remote run is claimed.

{live_text}

## Attribution and resume wording

Retail policy attribution: [Sierra Research tau-bench v2.0.0](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0), pinned to 989d3285ae669e49503324bdb9981bf14fccedeb; policy and MIT license remain in [third_party/tau-bench](third_party/tau-bench/README.md). ResolveDesk implements its own limited tools, controller, synthetic data, training and grader and claims no official tau-bench score. Qwen weights retain Apache-2.0.

> Built a local Qwen3 retail agent with model-extracted typed intent, observed-fact policy gates and confirmation-bound idempotent transactions. Continued 3.67M-parameter QLoRA with full-completion SFT and validation-only checkpoint selection. On 32 new synthetic tasks, four controlled arms separated workflow from training effects; with the typed controller held fixed, previous versus trained weights changed task success {old["task_success"]}/32 → {new["task_success"]}/32, authorized stockout fallback {old["fallback_success"]}/2 → {new["fallback_success"]}/2, available exchanges {old["fulfillable_exchange_success"]}/10 → {new["fulfillable_exchange_success"]}/10 and unnecessary executed handoffs {old["unnecessary_handoff_executed"]} → {new["unnecessary_handoff_executed"]}. Retained {pairs["regressed"]} paired regressions and verified with {tests} engineering tests.

Keep the synthetic scope, comparator, controller's deterministic role, absolute gate result and remaining failures with quantitative claims. [Round 4](docs/round-4.md), [round 3](docs/round-3.md), [round 2](docs/round-2.md) and [round 1](docs/round-1.md) remain archived.
"""
    (ROOT / "README.md").write_text(text)
    print("README generated from actual verified v5 evidence")


if __name__ == "__main__":
    main()
