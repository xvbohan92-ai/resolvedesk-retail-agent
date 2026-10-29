# Final round 6 workflow and fresh evaluation

This folder calibrates the **same real V6 SFT weight candidates**, with no additional optimizer run. Development validation motivated the final controller. All model-facing source, data, selection and acceptance rules were then frozen before final validation and test. The retired V6 test was never used for model inference; its records and retired plan are kept separately.

## Actual training lineage

The one V6 training run continues the selected V5 adapter on `Qwen/Qwen3-4B-MLX-4bit`, revision `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. It updates 3,670,016 parameters, rank 8 in the last eight layers, batch 1, Adam 0.000008, seed 202610286. All **768 optimizer updates** completed; checkpoints at exactly 384 and 768 are the only candidates. Local weights are ignored by Git.

The original frozen SFT inputs remain **96 source cases / 88 unique interactions** in 56 order worlds; **590 snapshots / 510 unique prompt/completion pairs**. There are 510 clean reference decisions and 80 safe recovery decisions from 16 simulated false-permission states. An incorrect initial prediction appears only in input state, never as a supervised target. These are not independent new tasks. The original 192 reference trajectories and 16 recovery trajectories passed the preserved business grader.

Prompt/padding weights are zero, ordinary completion tokens weight 1, and tokens overlapping only the `fallback_return` value weight 8. Tokenizer character offsets identify all 122 permission targets. Full context is retained and every completion target is covered by the 512-position output projection; preflight compares it with full output projection and discards its updated weights. This is a combined data/objective change, without a separate ablation isolating either component.

[Actual optimizer evidence](../../reports/v6/updates.json) binds [the original training manifest](../v6/datasets/manifest.json). [The final training/selection report](../../reports/v6_1/training.json) has **two explicit lineages**: `training_dataset` / `training_manifest_sha256` refer to actual SFT; `dataset` / `manifest_sha256` refer to this folder's controller/evaluation freeze. `config` records the optimizer run and `evaluation_config` records the final adapter serving path. These fields must not be interpreted as a second training run.

## Controller contribution

- Initial model slots are predictions. Quote matching proves provenance, not semantic permission. A predicted `yes` on an unavailable replacement permits one scoped consent question; only a subsequent whole-message customer `YES` authorizes that exact fallback. `NO` or unusable answers require the model to issue a handoff. Available replacements remain exchanges without fallback consent.
- Payment is collected before scoped consent, so supplying a missing payment does not invalidate consent and trigger a repeat question. Final transaction confirmation is separate and idempotent; a new unsolicited instruction revokes the pending proposal and scope.
- Requested email replies accept one email address with bounded positive contact boilerplate. Negation, multiple addresses, quoted instructions and new intents fall back to model interpretation. This is demo identification, not authentication.
- Requested replacement, payment and cancellation reason replies accept only the defined exact scalar formats and observed catalog/payment values. A vague or nonliteral initial replacement prediction becomes `unknown` and prompts a clarification; a genuine literal absent option remains absent and requires assistance. Raw arguments and `slot_normalizations` are retained separately.
- No controller emits a hidden corrective tool call or automatic handoff. The model must issue every action, and rejected/repeated attempts consume the budget. The controller never reads family names, gold labels, scripted answers or expected outcomes.

**Initial intent accuracy is scored from raw model action arguments**, before backend normalizations. Thus deterministic recovery is not counted as learned model understanding. Scoped execution safety and neural permission interpretation have separate metrics.

## New evaluation population

Final validation and test each contain **48 unique complete interactions, 24 families and 28 shared order worlds**, with no exact aliases. Every final reference trajectory passes the business grader. [The result-blind population plan](../../reports/v6_1/analysis-plan.json) was written before final validation/test and is bound by [the frozen manifest](datasets/manifest.json). Interaction identity includes initial message, world, actual scripted replies and confirmation, rather than message text or case ID alone.

All training/final-validation/final-test entity IDs are disjoint, and final worlds are disjoint from prior experiments. Within-split counterfactuals intentionally share worlds and common language/policy. The training duplicates are disclosed rather than relabeled as independent evidence. There is no population-level confidence interval or external benchmark claim.

Each 48-case final split includes 14 available exchanges, four authorized stockout returns, 16 necessary handoffs and eight missing-email tasks. There are six explicit initial permission grants and 42 refusals/unspecified requests. Positive permission recall and unsupported affirmative predictions are reported separately; predicting `unknown` everywhere would not solve interpretation.

## Four controlled arms

| Actual evaluator arm | Weights | Workflow |
| --- | --- | --- |
| `previous_old_flow` | Preserved selected V5 | Preserved V5 |
| `previous_new_flow` | Same V5 | Final typed recovery and scoped consent |
| `base_new_flow` | Bare pinned base | Same final workflow |
| `qlora_v6_1` | Validation-selected V6 candidate | Same final workflow |

The frozen config retains the development label `qlora_v6` in its descriptive `evaluation_arms` list; the executable, frozen `evaluation.ARMS` and all final records use `qlora_v6_1` as shown above. The model-facing workflow interface remains `v6` for compatibility with the actual SFT prompts, while runtime/session identity is `v6_1`.

Every arm runs the same new 48 tasks once with unconstrained greedy MLX decoding, 512 maximum output tokens and twelve total model calls per task, including rejected actions. Scripted answers are supplied only after the model asks their field. There are no test warmups, hidden retries or corrective model calls. Historical scores use different data and are not pooled. Serial wall times include resource effects and do not establish a speedup.

Metric grain: `exchange_as_return` and `unauthorized_return` count **tasks with at least one** `propose_return` attempt, not total repeated tool calls. The latter applies to exchange requests whose expected outcome is not a return. Other operation errors remain in full business scoring and the confusion matrices. Repeated-field and unnecessary-handoff attempt metrics count actual calls; executed handoffs count tasks. Initial permission diagnostics include all 48 initial requests and do not treat missing parses as correct explicit grants.

Frozen validation selection prioritizes safety, fewer unsupported affirmative permissions, balanced exchange/fallback/handoff/email success, overall success, unnecessary handoffs, repeats and calls. Ties prefer the earlier checkpoint. Test outputs never select a checkpoint or alter the controller.

The predeclared local acceptance gate requires all 4 fallback, 14 exchange, 16 handoff and 8 email tasks correct; zero unsupported affirmative predictions, unnecessary executed handoffs, repeated-field attempts, available-exchange returns and unauthorized-return attempts; confirmation intact on 48 tasks; overall success no lower than both new-flow comparators. A failed gate remains a failure, even if the scoped controller prevents harmful execution.

## Reproduce and inspect

Provision the pinned base and reproduce prior adapter initialization if weights are absent. Use the separately locked Apple Silicon training environment. The actual training commands are `training.v6.train --preflight` and `training.v6.train`; final commands are:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v6_1.select
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v6_1.evaluate
MPLCONFIGDIR=.mpl-cache .venv/bin/python -m training.v6_1.visualize
.venv/bin/python -m training.v6_1.verify_results
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v6_1.serve
```

Delivered training/selection/test outputs refuse overwrites. Resume a genuinely interrupted test only under identical recorded fingerprints; keep all completed failures. Inputs informed by final test outputs require another freeze and fresh test.

The independent verifier replays all raw actions and actual replies without an LLM, recomputes business/intent metrics, checks data separation and reconciles CSV/figure fingerprints. This is code/trace verification, not independent human annotation. Six quantitative plots live in [docs/figures/v6_1](../../docs/figures/v6_1). Actual application/model probes use public validation cases and are excluded from test accuracy.

Serving helpers, charts, docs and verification are excluded from the model-input freeze and record their own fingerprints. The selected adapter uses a separate MLX endpoint; no Ollama/GGUF merge is claimed. Default Ollama and historical sessions preserve their previous controller. Background Ollama caches are allowed to expire naturally, never unloaded. Databases, credentials and weights are excluded from Git.

Customers, orders, payments and escalations are simulated. There is no official tau-bench result, production certification, real customer satisfaction or business savings measurement.
