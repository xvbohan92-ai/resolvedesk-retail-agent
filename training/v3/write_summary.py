"""Generate current project documentation from the completed, reconciled experiment."""

# ruff: noqa: E501 -- Markdown paragraphs and tables retain natural source lines.

import json
import xml.etree.ElementTree as ET
from collections import Counter

from training.common import ROOT
from training.v3.common import REPORTS
from training.v3.verify_results import verify


def main():
    verify()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    train = json.loads((REPORTS / "training.json").read_text())
    values = json.loads((ROOT / "docs/figures/v3/values.json").read_text())
    s = report["summary"]
    base, old, new = [s[v] for v in ("base", "qlora_v2", "qlora_v3")]
    tests = ET.parse(REPORTS / "test-results.xml").getroot().find("testsuite").attrib["tests"]
    pairs = values["paired"]["previous_to_new"]
    table = [
        "| New fixed v3 test · same controller | Base | Previous QLoRA v2 | New QLoRA v3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, den in [
        ("Exchange incorrectly changed to return ↓", "exchange_as_return", "fulfillable_exchange_tasks"),
        ("Available exchanges fulfilled ↑", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("All tasks with prerequisites ↑", "task_success", "tasks"),
        ("Ordinary exchanges", "ordinary_exchange_success", "ordinary_exchange_tasks"),
        ("Direct-return negative controls", "direct_return_success", "direct_return_tasks"),
        ("Required fields requested", "clarifications_obtained", "required_clarifications"),
        ("Schema-valid actions", "schema_valid", "model_calls"),
        ("Confirmation preserved", "confirmation_preserved", "tasks"),
    ]:
        table.append("| " + label + " | " + " | ".join(f"{a[key]}/{a[den]}" for a in (base, old, new)) + " |")
    for label, key in [
        ("Unauthorized return attempts", "unauthorized_return"),
        ("Repeated business reads executed", "duplicate_read_executions"),
        ("Rejected write attempts", "rejected_write_attempts"),
        ("Model calls", "model_calls"),
    ]:
        table.append("| " + label + " | " + " | ".join(str(a[key]) for a in (base, old, new)) + " |")
    table.append(
        "| Median local time, descriptive | "
        + " | ".join(f"{a['latency_p50_ms'] / 1000:.2f} s" for a in (base, old, new))
        + " |"
    )
    table.append(
        "| All identical action repeats | "
        + " | ".join(str(values["identical_action_repeats"][v]) for v in ("base", "qlora_v2", "qlora_v3"))
        + " |"
    )
    handoffs = {
        v: [r for r in report["records"] if r["variant"] == v and r["expected"]["kind"] == "handoff"]
        for v in ("base", "qlora_v2", "qlora_v3")
    }
    table.append(
        "| Expected handoffs completed | "
        + " | ".join(f"{sum(r['task_success'] for r in rows)}/{len(rows)}" for rows in handoffs.values())
        + " |"
    )
    old_by_id = {r["case_id"]: r for r in report["records"] if r["variant"] == "qlora_v2"}
    regressions = Counter(
        r["family"]
        for r in report["records"]
        if r["variant"] == "qlora_v3" and not r["task_success"] and old_by_id[r["case_id"]]["task_success"]
    )
    regression_text = ", ".join(f"{f}: {n}" for f, n in sorted(regressions.items())) or "none"
    repeated_questions = 0
    wrong_fallback_exchange = 0
    for r in report["records"]:
        if r["variant"] != "qlora_v3" or r["task_success"]:
            continue
        fields = Counter(r["questions"])
        repeated_questions += any(n > 1 for n in fields.values())
        wrong_fallback_exchange += r["family"] == "exchange_fallback" and r["predicted_outcome"] == "exchange"
    gate = report["release_gate"]
    decision = "passes" if gate["promoted"] else "does not pass"
    change = new["task_success"] - old["task_success"]
    overall_note = (
        f"**Overall success regressed by {-change} tasks relative to v2.** Reducing this error class did not improve the whole agent."
        if change < 0
        else "**Overall success is unchanged relative to v2.** Corrected intent errors do not by themselves establish a better agent."
        if change == 0
        else f"Overall success improves by {change} tasks relative to v2, with {pairs['regressed']} paired regressions retained."
    )
    checks = "\n".join(f"- `{k}`: {'pass' if v else 'fail'}" for k, v in gate["checks"].items())
    live_text = "Live application verification is separate from held-out model accuracy."
    live_path = REPORTS / "live-api-verification.json"
    if live_path.exists():
        live = json.loads(live_path.read_text())
        if live.get("status") == "complete":
            live_text = (
                f"Actual loopback app/selected-adapter probes produced {sum(r['proposal_correct'] for r in live['records'])} correct proposals in {len(live['records'])} validation worlds after field-keyed scripted customer replies. "
                "The original stops for extra clarification are preserved. Confirmation did not write before approval, and repeated confirmations returned identical receipts. "
                "Each API advance has its own eight-step budget; these are integration checks, not the eight-total-call held-out benchmark or human customer judgments. "
                "[Full API conversations and approvals](reports/v3/live-api-verification.json) · [Initial clarification stops](reports/v3/live-api-initial.json)."
            )
    text = f"""# ResolveDesk · Local Retail Action Agent

An English retail support agent running locally with Qwen3. It reads customer/order/catalog facts, asks for missing details and proposes a cancellation, return or variant exchange. Explicit customer confirmation commits an immutable proposal atomically and idempotently. All customers, payments and orders are simulated.

**Round 3 targets exchange → return confusion.** On a new 48-task synthetic test, mistaken returns on 15 available-exchange requests change from **{old["exchange_as_return"]}/15 to {new["exchange_as_return"]}/15**, available exchanges completed from **{old["fulfillable_exchange_success"]}/15 to {new["fulfillable_exchange_success"]}/15**, and overall task success from **{old["task_success"]}/48 to {new["task_success"]}/48**. The unchanged base resolves **{base["task_success"]}/48**. The new adapter **{decision}** the predeclared local candidate gate. Ollama remains the default integration; v3 MLX is explicitly selectable. This is a local prototype, not a production deployment.

{overall_note}

## Measured outcomes

{chr(10).join(table)}

All three arms use the same frozen v2 controller, prompts, tools, tokenizer, greedy unconstrained decoding and eight-call budget. We change the combined training data/objective/checkpoint-selection recipe; this is not an ablation separating each component. Relative to v2, paired outcomes are **{pairs["improved"]} improved, {pairs["regressed"]} regressed, {pairs["both_pass"]} both passed and {pairs["both_fail"]} both failed**. Mistaken-return counts include any return attempt on an available-exchange request. Authorized unavailable-stock fallbacks are excluded. Task success also requires the right variant, payment, exact amount, required information and confirmed/declined final state. Zero repeated executed reads is a controller invariant, not proof of no repeated model attempts. Serial latency is descriptive.

**Observed regressions:** {regression_text}. Among failed new-adapter runs, **{repeated_questions}** repeat the same clarification field, and **{wrong_fallback_exchange}** unavailable-stock fallback cases propose an exchange instead of the authorized return. These are trajectory observations; they do not isolate a training component as the cause. A zero mistaken-return count on available exchanges does not establish general policy compliance. The new weights remain experimental.

![New same-controller comparison](docs/figures/v3/evaluation.png)

![Business-operation confusion](docs/figures/v3/confusion.png)

![All scenario families](docs/figures/v3/families.png)

![Validation-only business checkpoint selection](docs/figures/v3/training.png)

[Raw 144 trajectories](reports/v3/evaluation.json) · [Per-case CSV](docs/figures/v3/per-case.csv) · [Every failed run](reports/v3/failure-analysis.json) · [Training and adapter fingerprints](reports/v3/training.json) · [Verification](reports/v3/verification.json) · [Full method and reproduction](training/v3/README.md)

## Personal implementation and improvements

- Built the local action workspace, typed tools, progress-aware read cache, revision checks and transactional approvals. Proposals do not mutate orders. Confirmation binds session, immutable proposal and current order revision; replay returns the same receipt. Changed intent revokes pending proposals.
- Diagnosed the previous adapter's semantic failure: valid JSON and accepted arguments still selected returns for every available exchange in its earlier test. That old test is now development evidence, preserved unchanged.
- Added counterfactual training worlds: identical request with stock available/unavailable, and identical catalog/order with explicit exchange/return intent. Added customer corrections after wrong returns and direct returns after catalog inspection; only correct next actions are supervised.
- Changed the actual MLX training loss: exact JSON tool-name token spans receive **8×** weight; other completion tokens receive 1× and prompt/padding tokens 0. Available-exchange terminal decisions are sampled 3× and correction snapshots 2×. This is a parameter update, not a renamed Ollama template.
- Selected exported checkpoints using real **validation business outcomes**, prioritizing completed exchanges over total successes, wrong-return counts and weighted loss. All 16 validation tasks run in a fresh adapter-loading process. The 48 test tasks never select the checkpoint.
- Added semantic confusion matrices, paired regressions, exact trace/CSV reconciliation, hash-verified UI evidence and negative controls for genuine returns. No inference keyword routing, forced exchange action or oracle fallback is added.

## Data and actual training

**128/16/48 train/validation/test tasks**, 16 families, and **480/62/186 reference snapshots**. Within-split counterfactuals intentionally share worlds: **112/14/42 distinct order IDs**. Generated entity IDs and worlds are separated across splits and earlier experiments; phrasing is split-specific and prices are generated independently. The author, policy, schema, domain language and most scenario families are shared. Demonstrations and field-keyed customer replies are programmatic. Tasks are synthetic and correlated; there are no human retail annotations, independent wording judges, production-generalization claims or statistical-significance claims.

The pinned official `Qwen/Qwen3-4B-MLX-4bit` revision is `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. A new adapter starts from this base, not the old adapter or preflight weights. Rank 8 supervised QLoRA fine-tuning in the last eight layers updates **{train["trainable_parameters"]:,} parameters**, batch 1, learning rate 0.00005, seed 202610273, 240 optimizer updates. The selected checkpoint has **{train["best_checkpoint"]["iteration"]} completed updates**, adapter L2 change **{train["adapter_l2_change"]:.5f}**, and weighted validation loss **{train["best_checkpoint"]["weighted_valid_loss"]:.6f}**. Training plus validation took **{train["elapsed_seconds"] / 60:.2f} minutes**, with **{train["peak_memory_bytes"] / 1e9:.2f} GB** peak parent-process MLX allocation; the validation child allocation is not included. All nine base fingerprints are verified. Longest training snapshot: 1,590 tokens under the 2,048-token cap; no truncation. Weighted loss cannot be compared numerically to v2's unweighted loss.

Only the predeclared positive candidates at 119 and 239 completed updates are eligible; final-update 240 weights are retained but excluded. All selection trajectories and failures are retained. The candidate gate is a local prototype rule:

{checks}

Further training on these test failures requires a new held-out test. [Round 1](docs/round-1.md) and [round 2](docs/round-2.md), including their regressions and figures, remain archived.

## Run locally

Use Python 3.12, uv, local Ollama and existing `qwen3:4b` weights:

```bash
uv sync --frozen
uv run python -m training.setup_ollama
uv run resolvedesk serve --port 8771
```

Open [ResolveDesk](http://127.0.0.1:8771/). Choose one of 16 new validation contexts, edit the English request and inspect actual tool actions and exact proposals. Every session has isolated sandbox orders. Model evaluation defaults to the new comparison; two old experiments remain selectable. Validation fixtures exposed to the application contain no expected outcomes or scripted answers; test cases are inaccessible to the case API. Email lookup is demo identity resolution, not production authentication. Multi-item orders, live retail systems and real financial transactions are outside this prototype.

After following the [training setup](training/README.md) and [round-3 reproduction](training/v3/README.md), serve the actual selected adapter separately:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v3.serve
```

The loopback service on 8772 verifies the selected adapter fingerprint. Choose **QLoRA v3 · MLX**. It is not merged into Ollama/GGUF. Ollama's template-only local profile and schema-constrained decoding are a separate integration, not a substitute for the causal MLX comparison. `.env.example` describes process variables; `.env` is not loaded automatically. Model weights, databases and credentials are excluded from Git.

{live_text}

![Actual evaluation page](docs/evaluation-v3-browser.png)

The [browser verification](reports/v3/browser-verification.json) also retains a separate default-Ollama validation exchange: five actual model calls, an initially rejected attempt before catalog inspection, and a correct pending proposal. Its isolated order remains unchanged before confirmation. This is integration evidence, not a trained-model test score.

![Actual default-model exchange proposal](docs/live-v3-ollama-exchange.png)

## Verification and attribution

```bash
uv run pytest -q
uv run ruff check resolve_desk training tests
node --check --input-type=module < resolve_desk/static/app.js
uv run python -m training.v3.verify_results
```

**{tests} Python tests passed locally.** They cover transaction rollback, idempotency, revisions, ownership, payment/variant validation, repeated reads, genuine-return controls, exact action weighting, validation-only selection, frozen data and delivered-result reconciliation. Real MLX/Ollama integration runs are recorded separately; no remote CI run is claimed.

Policy attribution: [Sierra Research τ-bench v2.0.0](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0), pinned to `989d3285ae669e49503324bdb9981bf14fccedeb`; policy and MIT license are retained in [third_party/tau-bench](third_party/tau-bench/README.md). ResolveDesk implements its own limited controller, tools, UI, datasets, training and evaluator. It does not run official tasks, simulator or judges and does not claim an official τ-bench score. Qwen weights retain their Apache-2.0 license.

## Resume wording

> Built a local Qwen3 retail-action agent with revision-aware tools and confirmation-bound idempotent transactions. Diagnosed exchange/return confusion and implemented counterfactual correction data, 8× tool-name weighted QLoRA and validation-business checkpoint selection (3.67M trainable parameters). On a fresh 48-task synthetic test with a fixed controller, changed mistaken returns from {old["exchange_as_return"]}/15 to {new["exchange_as_return"]}/15, completed exchanges from {old["fulfillable_exchange_success"]}/15 to {new["fulfillable_exchange_success"]}/15, and overall success from {old["task_success"]}/48 to {new["task_success"]}/48; retained {pairs["regressed"]} paired regressions and verified with {tests} engineering tests.

Keep the synthetic scope, remaining failures and exact comparison group with every quantitative claim.
"""
    (ROOT / "README.md").write_text(text)
    print("README generated from reconciled v3 results")


if __name__ == "__main__":
    main()
