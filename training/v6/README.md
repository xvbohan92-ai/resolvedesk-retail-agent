# Retired development workflow — no held-out model evaluation

The actual 768-update SFT run in this folder is retained. Its development validation exposed email/payment recovery failures, so the proposed V6 test and reporting plan below were retired **before any held-out model inference**. No `reports/v6/evaluation.json` exists. Use [the final controller and fresh evaluation](../v6_1/README.md), which calibrates the same two weight checkpoints without extra SGD. The following documents the original, unexecuted test plan; it is not a result. See `reports/v6/retirement.json`.

# Round 6: scoped fallback consent and clarification recovery

V5 is preserved as development evidence, including its incorrect return proposals and failed email replies. V6 has its own frozen manifest and an unexecuted test plan. The model-facing files must not be edited to fit those test outputs.

## Controller changes

The seven initial intent fields remain unverified model predictions. Quotation matching proves text provenance, not semantic authorization. On a stockout, a predicted permission of `yes` only permits one scoped consent question. A later, whole-message `YES` from the customer grants that exact order/variant/request/revision scope. `NO` and unusable answers require a model-issued handoff. A model cannot grant consent through a tool argument or a quotation. An available replacement goes straight to an exchange proposal without this question.

Consent does not commit a return. The later proposal still needs its own explicit confirmation. A new unsolicited instruction revokes a pending proposal and invalidates earlier consent. These are separate decisions, not duplicate clarification questions.

Only a whole-message, single email address is projected automatically into an actually requested email field. Extra text, multiple addresses and new instructions are left for the model to interpret with fresh evidence. This deliberately narrow projection does not certify identity. Free-text email replies therefore remain a learned recovery task.

The gate uses current predictions, actual replies and completed tool results. It never reads family names, gold intent, expected operations or scripted answers. The model still emits every tool action, including handoff; rejected attempts consume the total budget. Deterministic controller improvements must be separated from learned behavior.

## Data and objective

- 96 training, 48 validation and 48 test source records across 24 scenario labels. Exact-alias auditing found 88/44/44 unique interactions; two new labels duplicated existing scenarios. The frozen inputs and raw selection remain unchanged. Primary reporting uses a source-first, result-blind set of 44 test representatives fixed in reports/v6/analysis-plan.json before held-out inference.
- 56/28/28 distinct order worlds. Several requests intentionally share a world within each split to contrast permission, stock and missing-field behavior; entities do not cross splits or earlier experiments.
- 590 training snapshots (510 unique prompt/completion pairs): 510 clean reference decisions and 80 safe-recovery decisions from 16 simulated false-positive permission states. The wrong initial prediction appears only as input state, never as a supervised target. These are not additional independent tasks.
- 255 reference snapshots in each validation/test split. All 192 clean trajectories and the 16 training recovery trajectories pass the independently preserved business grader before training.
- Raw test permission labels: 6 grants, 12 explicit refusals/non-authorizations and 30 unspecified requests. The 44 unique interactions contain 4 grants, 10 refusals/non-authorizations and 30 unspecified requests. Report false positives and positive recall separately; always predicting `unknown` is not reliable interpretation.

Continue the selected V5 adapter on pinned `Qwen/Qwen3-4B-MLX-4bit`, revision `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. Rank 8 in the last eight layers updates 3,670,016 parameters. Train 768 new updates, batch 1, Adam at 0.000008, seed 202610286, with candidates at 384/768.

Supervise the complete assistant completion. Prompt and padding receive zero weight; ordinary completion tokens receive weight 1. Tokens overlapping only the `fallback_return` value receive weight 8, checked through tokenizer character offsets. All 122 such training targets and every other completion target remain covered. The 512-position output projection preserves full input context; it is not a context truncation. Preflight compares weighted full versus selected output projection and discards its updated weights.

## Controlled comparison and acceptance

| Arm | Weights | Workflow |
| --- | --- | --- |
| previous_old_flow | Preserved selected V5 | Preserved V5 |
| previous_new_flow | Same V5 | Scoped consent + email projection |
| base_new_flow | Bare pinned base | Same new workflow |
| qlora_v6 | Validation-selected continuation | Same new workflow |

All four arms run the same 48 new tasks, unconstrained greedy MLX generation, 512 output tokens and twelve total model calls per task. All rejected actions count. There is no test warmup generation, retry-to-pass or hidden corrective model call. Scripted replies are supplied only after the model actually asks their field. Archived scores use different tests and are never pooled.

Business scoring preserves operation/variant/payment/amount, identity/order/catalog prerequisites, confirmation and decline checks, and additionally requires all authored missing fields in the two-field email/payment cases. Fresh-consent friction belongs to the workflow comparison. Initial intent fields and raw unauthorized attempts are separately scored.

Validation-only selection prioritizes zero exchange-as-return/unauthorized-return attempts, fewer unsupported affirmative permissions, balanced exchange/fallback/handoff/email outcomes, overall success, unnecessary handoffs, repeats and calls. A tie selects the earlier candidate. Selection precedes all held-out model inference.

The predeclared raw 48-row local gate requires all four stockout returns, all fourteen available exchanges, all sixteen necessary handoffs and all six missing-email cases correct; zero unsupported affirmative initial permission predictions, unnecessary handoffs, repeated-field attempts, exchange-as-return and unauthorized-return attempts; intact confirmation and overall success no lower than both new-flow comparators. Passing is a small-suite result, not production certification.

## Reproduction and artifacts

Inspect `datasets/manifest.json`, `reports/v6` and `docs/figures/v6`. The root verifier replays every raw action and actual customer reply through the frozen controller without an LLM, recomputes business scores, and reconciles CSV/figure/model fingerprints. This is a code/trace audit, not independent human annotation.

Real training needs Apple Silicon, the training lockfile, the pinned base, and the exact ignored selected V5 adapter. Reproduce prior rounds if their local weights are absent. Provisioning, adapters, databases and credentials are excluded from Git. Serving uses `training.v6.serve` through MLX; no merge into Ollama/GGUF is claimed. Ollama remains a separate integration, and historical sessions keep their own controller.

For a separate reproduction copy, retain the manifest and run `training.v6.train --preflight`, `training.v6.train`, `training.v6.select`, then `training.v6.evaluate`. Delivered paths refuse overwrites. Any model-facing change informed by this held-out test needs another manifest and another fresh test. Visualization, verification, serving and publication helpers are excluded from the training-input freeze and record their own fingerprints.

The cases, labels, policy coverage and programmatic replies are self-authored and correlated. There is no official tau-bench result, human customer satisfaction measurement, business savings estimate or statistical generalization claim. Resource checks wait for other Ollama caches naturally and never unload them.

The four exact duplicate groups are retained and disclosed rather than counted as new evidence. Input identity includes message, world, actual scripted replies and confirmation, excluding case ID/family. Use the declared first-source representatives for all primary comparisons; inspect the raw family audit separately. The 28 shared test worlds remain correlated after deduplication. Future dataset revisions should run this uniqueness audit before training, not only before final inference.
