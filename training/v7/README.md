# Round 7: conditional-grant recall and model-issued schema recovery

Round 7 continues the selected V6.1 adapter with new deduplicated SFT inputs, new split-specific language and independent order worlds. Its controller, training objective, population, checkpoint-selection rule and acceptance gate were frozen before model inference. Historical results remain in [round 6](../../docs/round-6.md) and are not pooled with this new test.

## Controller and attribution

A `record_request` with an invalid enum is rejected without accepting any executable intent. Its raw arguments remain visible alongside allowed values and unverified valid-field context. The model must issue the full corrected action. The controller never converts an invalid reason into an accepted `unknown`, supplies semantic permission, emits a hidden tool call or automatically hands the task to a human.

The local schema-recovery limit is three invalid actions on the same actual customer turn, including the first failure and two possible model corrections. An accepted action resets that local streak; rejected policy actions cannot bypass it. Every attempt also consumes the unchanged twelve-call total evaluation budget. Exhaustion ends with a visible failure, not a forced handoff. Syntactically valid JSON with an invalid `reason` is distinguished from malformed JSON.

The prompt explicitly limits `reason` to cancellation reasons (`ordered by mistake`, `no longer needed`, or `unknown`); exchanges, returns and handoffs use `unknown`. First-person conditional approval must be interpreted semantically. No backend phrase list or benchmark labels decide whether permission exists.

Actual replies are retained in `clarification_state`. Narrowly validated email, catalog-option and payment/reason replies remain reusable. If a model later forgets one of those supplied values, the gate requires a model-issued corrected record rather than delivering the same question again. Attempts remain in the trace; these deterministic safeguards are not credited as learned model accuracy.

The prior scope protections remain: on a stockout a predicted `yes` only permits a separate, exact YES/NO question. Only the actual subsequent whole-message customer `YES` grants that specific order/variant/request/revision scope. A final return proposal still needs its own transaction confirmation. Replayed accepted confirmations commit once; a new unsolicited instruction revokes the old proposal and scope. Email lookup is demo identification, not authentication.

Historical sessions retain their original controllers. A mismatched fine-tuned service is rejected before a customer message can revoke an existing pending proposal. Default Ollama sessions retain their previous workflow; the new candidate is an explicit profile served by MLX. No Ollama/GGUF weight merge is claimed.

## Data and population audit

- **96 training, 48 validation and 48 test interactions**, each with its own order world. Full interactions and actual initial model-visible prompts are unique. All entity IDs and initial requests are disjoint between splits and from prior experiments.
- The 24 authored families retain cancellation, direct return, available exchange, stockout, gift-card, ownership, missing-field and refusal controls. Each 48-case validation/test split has **14 available exchanges, 16 required handoffs, eight missing-email tasks and four authorized stockout returns**.
- Permission labels in each validation/test split are **12 explicit grants, 12 refusals and 24 unspecified requests**. Grants include available exchanges, so always predicting `unknown` cannot satisfy the semantic evaluation. Training source cases contain 24 grants, 24 refusals and 48 unspecified requests.
- **192 clean reference trajectories and 76 recovery trajectories pass** the preserved independent business grader before training. Recovery input faults cover invalid exchange/cancellation reasons, stale evidence, repeated questions and false affirmative predictions. Wrong actions occur only in input state, never as supervised targets.
- Raw reference generation produces 940 training snapshots. Removing 116 exact prompt/completion duplicates leaves **824 unique pairs**: 96 initial records, 62 record/recovery decisions and 666 execution decisions. Validation and test each have 252 reference snapshots. Snapshots and repeated optimizer samples are not independent tasks.

[The result-blind population plan](../../reports/v7/analysis-plan.json), [reference audit](../../reports/v7/reference-audit.json) and [data summary](../../reports/v7/data-summary.json) are inspectable. The plan precedes all V7 model inference and is bound by [the frozen manifest](datasets/manifest.json). Input identity includes the initial message, full world, actual scripted answers and confirmation choice. Independent worlds remove the earlier shared-order aliases; common authored templates and retail policy still limit generalization. This is not an externally authored industrial benchmark.

## Actual weight-training configuration

Initialize from the selected V6.1 adapter on `Qwen/Qwen3-4B-MLX-4bit`, official revision `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. Rank 8 in the last eight layers updates **3,670,016 parameters**. The configured formal run is **1,344 new optimizer updates**, batch one, Adam 0.000006, seed 202610287, with candidates at exactly 672 and 1,344 updates. Completed counts and losses come from [the actual updates](../../reports/v7/updates.json), not from configuration alone.

Full-completion cross-entropy uses zero prompt/padding weight, ordinary completion weight one, **weight eight on the `fallback_return` value and weight four on the `reason` value**. Tokenizer character offsets audit all 158 permission-value tokens and 194 reason-value tokens in the physical training input. All completion targets remain supervised, including tokens outside those values.

The full input context is retained. The longest training sequence is 2,620 tokens, within the configured 3,072-token limit; the longest completion is 115 tokens. A 512-position output projection retains every supervised target and refuses to run if any target would be dropped. This is not context truncation. The actual one-step preflight matched full versus selected projection loss exactly and decreased same-sample weighted loss from 0.00361033 to 0.00183522. Those updated preflight weights are discarded.

Training alternates **50% record/record-repair targets and 50% execution/reply targets**, using seeded shuffled cycles within each pool. At 1,344 updates both pools receive 672 updates and all 824 physical rows are visited. Repeated samples arise from optimizer sampling, not physical duplicate rows. A selected earlier 672-step candidate has seen only a prefix; [selection evidence](../../reports/v7/training.json) separately reports its actual unique-row and bucket coverage. Do not imply that the selected checkpoint used every physical row when it did not.

The [CPU initialization audit](../../reports/v7/initialization-audit.json) compares all 112 predecessor tensors with the actual saved pre-optimizer snapshots for both preflight and formal training, checking exact keys, shapes, dtypes, values and finiteness. This verifies the continuation instead of merely trusting a permissive loader. Tensor contents and weights are not published.

This experiment changes data, sampling and two field weights together; it does not isolate the effect of any one change. Weighted training loss is not comparable to older uniform loss, and falling loss alone does not establish improved task completion.

## Four controlled arms and scoring

| Arm | Weights | Controller/prompt |
| --- | --- | --- |
| `previous_old_flow` | Selected V6.1 | Preserved V6.1 |
| `previous_new_flow` | Same selected V6.1 | New V7 |
| `base_new_flow` | Bare pinned base | New V7 |
| `qlora_v7` | Validation-selected V7 continuation | New V7 |

All four run the same fresh 48 tasks once through unconstrained greedy MLX generation, up to 512 output tokens and **twelve total model calls per task**, including every rejected action and model-issued repair. Scripted customer replies are supplied only after the model actually asks their field. There are no test warmups, retries to obtain a pass, hidden corrective model calls or checkpoint selection using test results.

**Initial intent is the first model output only**, even if rejected. A valid raw record with an illegal enum retains its original fields for semantic scoring; a malformed/wrong-tool/structurally incomplete first output remains a parse failure. Later accepted corrections never replace the initial prediction. Schema-valid counts exclude enum-invalid actions in every arm, including the new controller that preserves their parsed shape.

`enum_recovered` counts a task with at least one enum failure followed by a valid model-issued record on the **same actual customer turn**. A later new customer turn cannot repair earlier interpretation scores. Initial reason-error tasks, all reason-error attempts, eventual repairs and full task outcomes remain separate.

`exchange_as_return` and `unauthorized_return` count tasks with at least one corresponding return attempt, not all repeated calls. Unauthorized-return scope is exchange requests whose expected outcome is not a return. Repeated-field and unnecessary-handoff attempts count calls; executed handoffs count tasks. Business scoring also checks operation, exact variant, payment, amount, required reads, all authored missing fields, unchanged orders before confirmation and the final confirmation/decline outcome.

Frozen validation selection prioritizes zero unsafe return attempts, fewer unsupported affirmative predictions, explicit-grant recall, balanced business outcomes, overall completion, enum/parse errors, unnecessary handoffs, repeated asks and calls; exact ties prefer the earlier checkpoint. Both candidates are retained. A zero false-positive count does not establish permission understanding without positive recall and parse coverage.

## Predeclared acceptance and delivery

The local acceptance gate requires all four stockout returns, fourteen available exchanges, sixteen required handoffs, eight email tasks and twelve first-output explicit grants correct; zero unsupported affirmative permission, initial parse failures, initial enum errors, reason-enum attempts, unnecessary handoffs, repeated-field attempts, available-exchange returns and unauthorized-return attempts; confirmation preserved for all 48; overall completion no lower than either new-flow comparator. Failed checks remain failed.

[The raw test report](../../reports/v7/evaluation.json), [per-case CSV and figures](../../docs/figures/v7), [independent replay](../../reports/v7/verification.json) and [failure analysis](../../reports/v7/failure-analysis.json) retain all arms and failures. The verifier independently replays actual actions, replies and confirmation without an LLM and reconciles raw metrics, sampling, input/model hashes and figure values. This is code/trace verification, not independent human annotation.

Actual app/API probes use all 48 public validation cases and are reported separately. The app allows twelve decisions per advance, while offline accuracy uses twelve for the entire task; app probes therefore are not pooled with held-out accuracy. Browser YES/NO evidence and offline wheel installation checks are additional engineering validation, not benchmark cases.

For a separate reproduction copy with the pinned local base and predecessor adapter provisioned:

```bash
uv sync --frozen
uv sync --project training --frozen
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v7.train --preflight
training/.venv/bin/python -m training.v7.check_initialization --stage preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v7.train
training/.venv/bin/python -m training.v7.check_initialization --stage formal
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v7.select
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v7.evaluate
MPLCONFIGDIR=.mpl-cache .venv/bin/python -m training.v7.visualize
.venv/bin/python -m training.v7.verify_results
```

Delivered paths refuse overwrites. Reproduce absent predecessor weights from the prior rounds. Model-facing changes informed by this test require another freeze and fresh test. Reporting, visualization, serving and audit helpers are excluded from the input freeze and bind their own fingerprints. Weights, credentials and local databases are ignored by Git. Resource checks wait for other Ollama caches to expire naturally and never unload them.

Customers, orders, payments and escalations are simulated. No production reliability, human customer satisfaction, real business savings, statistical generalization or official tau-bench score is claimed.
