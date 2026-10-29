"""Render final documentation only from reconciled actual evidence."""

# ruff: noqa: E501 -- Embedded Markdown paragraphs remain readable.
import json
import xml.etree.ElementTree as ET

from training.common import ROOT
from training.v6.common import REPORTS
from training.v6.evaluation import ARMS
from training.v6.verify_results import verify


def main():
    verify()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    train = json.loads((REPORTS / "training.json").read_text())
    values = json.loads((ROOT / "docs/figures/v6/values.json").read_text())
    suite = ET.parse(REPORTS / "test-results.xml").getroot().find("testsuite").attrib
    assert suite["failures"] == suite["errors"] == "0"
    tests = suite["tests"]
    s = values["primary_summary"]
    raw = report["summary"]
    old, new = s["previous_new_flow"], s["qlora_v6"]
    prior = s["previous_old_flow"]
    table = [
        "| 44 unique test requests | V5 original flow | V5 scoped flow | Base scoped flow | V6 scoped flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key, den in [
        ("Full task success", "task_success", "tasks"),
        ("Available exchanges correct", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("Authorized stockout returns", "fallback_success", "fallback_tasks"),
        ("Required model handoffs", "handoff_success", "handoff_tasks"),
        ("Missing-email tasks", "missing_email_success", "missing_email_tasks"),
        ("Unsupported affirmative permission ↓", "unsupported_permission_yes", "non_authorized_tasks"),
        ("Explicit permission recognized", "true_permission_yes", "authorized_tasks"),
        ("Seven initial intent fields jointly correct", "intent_joint_correct", "intent_tasks"),
        ("Required customer answers obtained", "clarifications_obtained", "required_clarifications"),
        ("Schema-valid model actions", "schema_valid", "model_calls"),
        ("Confirmation preserved", "confirmation_preserved", "tasks"),
    ]:
        table.append("| " + label + " | " + " | ".join(f"{s[v][key]}/{s[v][den]}" for v in ARMS) + " |")
    for label, key in [
        ("Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"),
        ("Unnecessary model handoff attempts ↓", "unnecessary_handoff_attempts"),
        ("Unauthorized return attempts ↓", "unauthorized_return"),
        ("Available exchange changed to return ↓", "exchange_as_return"),
        ("Repeated question-field attempts ↓", "repeated_question_attempts"),
        ("Repeated questions delivered ↓", "repeated_questions_delivered"),
        ("Identical action repeats", "identical_action_repeats"),
        ("Scoped consent questions", "consent_questions"),
        ("All actual task-model calls", "model_calls"),
    ]:
        table.append("| " + label + " | " + " | ".join(str(s[v][key]) for v in ARMS) + " |")
    gate = report["release_gate"]
    checks = "\n".join(f"- {k}: {'pass' if v else 'fail'}" for k, v in gate["checks"].items())
    failures = [
        r["case_id"] for r in report["records"] if r["variant"] == "qlora_v6" and not r["task_success"]
    ]
    pairs = values["paired"]
    live = "Actual app/adapter probes are pending."
    if (REPORTS / "live-api-verification.json").exists():
        d = json.loads((REPORTS / "live-api-verification.json").read_text())
        if d.get("status") == "complete":
            live = f"Actual app → selected MLX adapter validation probes: **{sum(r['outcome_correct'] for r in d['records'])}/{len(d['records'])}** correct. Raw replies, adapter identity, pre-confirmation orders and idempotent confirmation receipts are retained in [API evidence](reports/v6/live-api-verification.json). These development smokes are excluded from held-out accuracy."
    figures = "\n\n".join(
        f"![{title}](docs/figures/v6/{name}.png)"
        for name, title in [
            ("evaluation", "Four-arm business outcomes and retained protections"),
            ("families", "All 24 scenario families"),
            ("intent", "All seven initial intent fields"),
            ("consent", "Permission predictions versus actual return attempts"),
            ("confusion", "Operation confusion including unfinished tasks"),
            ("training", "Real weighted training loss and validation-only selection"),
        ]
    )
    fence = chr(96) * 3
    text = f"""# ResolveDesk · Retail agent with scoped fallback consent

An English retail action agent with local Qwen3 inference, observable tools and customer-confirmed transactions. Customers, orders, payments and human handoffs are simulated.

**Round 6 targets fallback-permission interpretation, necessary escalation and missing-email recovery while retaining exchange completion and repeat-question protections.** On **44 unique fresh synthetic requests**, the prior V5 system versus this candidate changes task success **{prior["task_success"]}/44 → {new["task_success"]}/44**, available exchanges **{prior["fulfillable_exchange_success"]}/14 → {new["fulfillable_exchange_success"]}/14**, required handoffs **{prior["handoff_success"]}/14 → {new["handoff_success"]}/14**, and missing-email tasks **{prior["missing_email_success"]}/6 → {new["missing_email_success"]}/6**.

The candidate **{"passes" if gate["promoted"] else "does not pass"}** the predeclared local acceptance targets. Unsupported affirmative initial fallback predictions remain **{new["unsupported_permission_yes"]}/{new["non_authorized_tasks"]}**, with explicit positive permission recognized on **{new["true_permission_yes"]}/{new["authorized_tasks"]}**. Unauthorized return attempts are **{new["unauthorized_return"]}**, repeated-field attempts **{new["repeated_question_attempts"]}**, and available-exchange return attempts **{new["exchange_as_return"]}/14**. Model interpretation and scoped customer consent are separate measurements.

This is a small, self-authored, correlated test, not an official benchmark or production result. The trained adapters are served separately through MLX; no trained Ollama/GGUF merge is claimed.

**Population audit:** the frozen test has 48 rows but four are exact aliases. Before any held-out inference, [the analysis plan](reports/v6/analysis-plan.json) fixed the first source row in each identical interaction as the representative. Headline results use **44 unique requests**; all **192 raw four-arm records** remain retained. Validation checkpoint selection still uses its original 48 rows and frozen rule. Across the raw 48 candidate rows, there were **{raw["qlora_v6"]["unauthorized_return"]} unauthorized return attempts** and **{raw["qlora_v6"]["repeated_question_attempts"]} repeated-field attempts**. Raw candidate task success is **{raw["qlora_v6"]["task_success"]}/48**. Deduplication does not make the 28 shared test worlds independent.

## Four controlled arms

{chr(10).join(table)}

V5 original → new scoped flow holds weights fixed and measures the controller/prompt changes. V5 → V6 under the **identical new flow** measures the combined new-data and permission-weighted SFT: **{old["task_success"]}/44 → {new["task_success"]}/44**. The bare-base arm is another quality comparator. Do not attribute deterministic controller policy to learned model reasoning.

Every arm uses the same pinned base/tokenizer, unconstrained greedy MLX generation, 512 output tokens and **twelve total model calls per task**, including rejected actions. There are no test warmups or retries to obtain a pass. Historical V5 scores use a different test and are not pooled. Raw serial timings include background resource effects and are not speedup evidence.

End-to-end paired outcomes: **{pairs["end_to_end"]["improved"]} improved, {pairs["end_to_end"]["regressed"]} regressed**. Training with the new flow fixed: **{pairs["training"]["improved"]} improved, {pairs["training"]["regressed"]} regressed**. Current failed tasks: {", ".join(failures) or "none in this small suite"}. All failures across all arms remain in [failure analysis](reports/v6/failure-analysis.json) and the [per-case CSV](docs/figures/v6/per-case.csv).

{figures}

## Personal engineering contributions

- Replaced trust in a predicted fallback flag with a scoped customer-consent protocol. A model prediction of `yes` can request one explicit consent question; it cannot grant the return. Only the subsequent whole-message customer `YES` permits that exact order, replacement, request and observed revision. `NO` and unusable replies require a model-issued handoff. Available exchanges skip this consent step.
- Retained separate confirmation-bound transactions. Fallback consent does not commit a return; the actual proposal needs another explicit confirmation. A new unsolicited instruction revokes pending proposals and invalidates old consent. Replayed confirmation returns the same receipt and commits once.
- Projected only literal single-address replies into an actually requested email field. Free-text email replies still need a model update quoting the latest customer turn. Multiple addresses and new instructions cannot be silently swallowed by this narrow projection. Email lookup remains demo identification, not authentication.
- Preserved exact variant/payment checks and clarification memory. The model still emits every action; wrong/repeated attempts remain visible and consume budget. Gate logic reads actual tools and replies, never benchmark family names or expected answers.
- Added four controlled arms, intent-field diagnostics, attempted-versus-executed escalations, raw action replay without an LLM, figure/CSV reconciliation and compatibility with historical sessions.

## Real local fine-tuning

**96 train / 48 validation / 48 test source records**, with **88/44/44 unique interactions**, 24 scenario labels and **56/28/28 distinct order worlds**. Two newly added labels accidentally duplicated existing scenarios. These aliases remain in the frozen training data, raw validation selection and all 192 four-arm test records; they are not independent examples. Several intent/stock counterfactuals intentionally share a world within a split. Entities are disjoint across splits and earlier experiments; policy, vocabulary and families are shared. Numbers and phrases are authored rather than independently sampled from business traffic.

**590 training snapshots (510 unique prompt/completion pairs)** comprise 510 clean reference decisions and 80 safe recovery decisions from 16 simulated false-permission states. Wrong initial predictions occur in input state only, never as training targets. These are not additional independent tasks. All 192 clean reference trajectories and 16 recovery trajectories pass the preserved business grader before training.

Continued the selected V5 adapter on **Qwen/Qwen3-4B-MLX-4bit**, pinned revision **52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495**. Rank 8 in the last eight layers updates **{train["trainable_parameters"]:,} parameters**. Completed **{train["completed_updates"]} new updates** at batch 1, Adam 0.000008, seed 202610286; validation selected **{train["best_checkpoint"]["iteration"]} updates** from exactly 384/768 before any held-out inference.

Full-completion cross-entropy uses prompt/padding weight 0, ordinary completion weight 1 and weight 8 on target tokens overlapping only the permission value. Tokenizer offsets audit all 122 such training targets. This is a combined data/objective change, without an ablation isolating each component. Weighted loss is not directly comparable to prior uniform training loss.

The longest training sequence is **{train["token_audit"]["longest_train_sequence"]} tokens**; all targets are covered by the 512-position output projection and full input context is retained. Preflight weighted full-versus-selected output projection error was zero, and its updated weights were discarded. Optimizer-loop time was **{train["elapsed_seconds"] / 60:.2f} minutes**; peak active MLX allocation **{train["peak_active_memory_bytes"] / 1e9:.2f} GB** excludes cache, OS and other processes.

[Method and reproduction](training/v6/README.md) · [Frozen manifest](training/v6/datasets/manifest.json) · [Real updates](reports/v6/updates.json) · [Independent replay and audit](reports/v6/verification.json)

## Predeclared raw 48-row acceptance

{checks}

These frozen targets use all 48 raw rows, including aliases; the primary table uses 44 unique requests. They are not production certification. No confidence interval, human customer satisfaction, real business savings or official tau-bench score is claimed.

## Run and inspect

{fence}bash
uv sync --frozen
uv run resolvedesk serve --port 8771
# After provisioning the pinned local base and selected adapter:
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v6.serve
{fence}

Open [ResolveDesk](http://127.0.0.1:8771/). Select **Experimental QLoRA v6 · MLX** to inspect the candidate and its scoped YES/NO controls. Default Ollama sessions retain their previous workflow; API callers can opt into the new flow with experimental_flow=true. Historical sessions retain their controller and pending proposals. Twenty-two unique public validation examples are shown; all 48 source rows remain in the case API for compatibility. Packaged fixtures exclude scripted answers, expected outcomes and test worlds. Multi-item resolutions require human assistance.

Services bind to loopback. Weights, local databases and credentials are excluded from Git. Background Ollama caches are allowed to expire naturally and are never unloaded by training or evaluation.

![Actual evaluation page](docs/evaluation-v6-browser.png)

**{tests} engineering tests passed locally.** Six figures are generated from the raw report and independently reconciled. {live} GitHub CI is configured; no remote run is claimed.

## Attribution and resume wording

Retail policy attribution: [Sierra Research tau-bench v2.0.0](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0), pinned to 989d3285ae669e49503324bdb9981bf14fccedeb. Policy and MIT license remain in [third_party/tau-bench](third_party/tau-bench/README.md). ResolveDesk implements its own limited tools, controller, synthetic cases, training and grader and claims no official score. Qwen retains Apache-2.0.

> Built a local Qwen3 retail agent with scoped fallback consent, clarification recovery and idempotent customer-confirmed transactions. Continued 3.67M-parameter QLoRA using offset-audited permission-weighted SFT and validation-only selection. On 44 unique new synthetic requests, four controlled arms separated workflow from training effects; under the same new controller, V5 versus trained weights changed success {old["task_success"]}/44 → {new["task_success"]}/44, required handoff {old["handoff_success"]}/14 → {new["handoff_success"]}/14 and missing-email resolution {old["missing_email_success"]}/6 → {new["missing_email_success"]}/6; unsupported affirmative permission predictions were {new["unsupported_permission_yes"]}/{new["non_authorized_tasks"]} and available exchanges {new["fulfillable_exchange_success"]}/14. Retained all failed cases and verified through {tests} engineering tests.

Keep the synthetic scope, comparison, controller contribution, acceptance outcome and remaining failures alongside quantitative claims. [Round 5](docs/round-5.md), [round 4](docs/round-4.md), [round 3](docs/round-3.md), [round 2](docs/round-2.md) and [round 1](docs/round-1.md) remain archived.
"""
    (ROOT / "README.md").write_text(text)
    print("README generated from reconciled actual V6 results")


if __name__ == "__main__":
    main()
