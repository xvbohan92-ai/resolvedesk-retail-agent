# Round 4: recover clarification, stockout fallback and handoff

Round 3 removed available-exchange return errors on its 15 eligible tests, but overall success fell from 23/48 to 20/48. Repeated question-field attempts, authorized fallback and handoff regressed. Those old tests are now development evidence, preserved in the round-3 archive. This round freezes a different test before formal training and model test inference.

## Engineering changes

`resolve_desk/progress.py` extends the preserved v2 workflow. Each real `ask_customer` result records the current message count. The next real customer turn becomes an **unverified** reply in clarification memory. An unsolicited new turn resets old clarification context. No fixture answers, task labels, expected outcome or hidden catalog facts enter the prompt.

An exact repeated field/question with an existing reply is blocked from delivery and returns the cached reply as recovery feedback. The attempted action remains in the trace and consumes the call budget. A different precise followup is allowed for an unusable answer. This guard does not automatically complete a task or force a model handoff. The existing completed-read cache remains unchanged.

Observed order status supplies cancellation/return/exchange eligibility, and an actual order-access denial supplies an access-denied signal. Revised guidance explains missing-field continuation, exact requested variants, explicit stockout-return authorization and payment limits. Code never routes intent from keywords or chooses a gold operation. The original workflow, policy, transactional store and graders remain unchanged for historical experiments.

## New data and training recipe

- **64/32/32 train/validation/test tasks**, 16 shared scenario families, **356/124/124** reference decision snapshots.
- **56/28/28 distinct order worlds**: stock and intent counterfactuals deliberately share worlds within a split, while generated entities and phrasing are separated across splits and older experiments. Price draws are independent; individual numeric amounts can overlap.
- Full clean trajectories preserve reads and clarification transitions. Training adds one extra terminal for each exchange/handoff case, continuation after blocked redundant questions, handoff after an unnecessary question at a policy boundary, correction after an unauthorized substitute exchange, and exchange/return preservation controls. Erroneous injected actions themselves are never supervised.
- Training action counts: 68 exchanges, 44 returns, 48 handoffs, 12 cancellations, 20 clarification questions and 164 read actions. Duplicated terminal weights are not additional unique tasks.
- Exact tool-name token spans receive **2×** cross-entropy weight, compared with 8× in round 3. Other completion tokens receive 1×; prompt and padding receive zero. The longest train example is 1,884 tokens under the 2,560-token cap, with no truncation. The numerical equal-error ratio is approximately 2.
- The same pinned official Qwen3-4B MLX 4-bit base initializes a fresh rank-8 adapter in the last eight layers. No old or preflight adapter is reused. Batch 1, learning rate 0.00003, seed 202610274, 480 updates. The iterator shuffles without replacement within each 356-row cycle. Full training therefore covers all rows; the selected candidate's update count is reported separately.

All reference trajectories execute through the actual validator and pass business grading. Data and scripted customer replies are self-authored and programmatic, not independently human-annotated conversations. The author, policy, schema, domain vocabulary and most scenario families remain shared. Tests are small and correlated; there are no significance or production-generalization claims.

## Validation-guided revisions before new test

The original SFT plan exported 240/480-update candidates. The complete SFT-240 validation scored **13/32**: **1/10 available exchanges, 0/2 authorized fallback and 8/8 handoffs**. It over-escalated normal operations. SFT-480 validation was interrupted after similar exchange/fallback failures. Original SFT records and fingerprints remain; no fabricated complete 480-step score is reported.

A separate, unregularized sequence-preference phase restarted from SFT-480 and trained on 124 original-train-only pairs. Despite increasing teacher-forced preference gaps, it scored **6/32**, including **0/10 exchanges, 0/2 fallback and 6/8 handoffs**, and handed off **23/24** normal tasks. Raw actions show alternative refusal wording instead of correct operations. The complete failed validation, training, weights and protocol are preserved. The new test was still unused; its automatic launch was stopped.

The failed phase also exposed execution issues: an eager loop stopped after six updates, a compiled preflight caught missing mutable-output state capture before updates, and a compiled loop stopped after 18 updates when its buffer cache caused memory pressure. Sources, manifests and metrics are archived in `reports/v4/preference-*`. These failed/preflight weights are never final initialization. Bounded-memory execution subsequently completed the full 124-update failed experiment.

## Final action-span preference + chosen-completion NLL

The final recipe restarts from the **original SFT-480 checkpoint**, never the failed preference adapter. It uses **136 pairs** from the original training worlds: 48 normal operations versus unnecessary handoff, 20 missing-information questions versus handoff, 20 continuations versus repeated questions, 20 exchanges versus wrong returns, 16 required handoffs versus irrelevant questions, and 12 added genuine-return versus wrong-exchange controls. No validation/test examples become training pairs. Every pair has exactly matching visible context on both sides.

Inspired by the [SimPO paper's pairwise objective and supervised regularization discussion](https://arxiv.org/html/2405.14734v2), this application-specific objective is:

```text
z = 2 * (mean action-name logp chosen - mean action-name logp rejected) - 0.5
loss = softplus(-z) + 1.0 * mean chosen-completion NLL
```

Preference rewards use the exact tool-name token spans, so changing a rejected handoff's reason wording cannot satisfy that term. The NLL term covers all chosen arguments, JSON structure and end-of-turn tokens. This differs from standard full-response SimPO and is not an official benchmark method or result. Exact character/token offsets define action masks; prompt and padding targets have zero weight.

Rank 8 in the last eight layers, batch 1, Adam learning rate **0.000003**, seed **2026102742**, one shuffled pass of **136 fixed updates**. Sequential rejected/chosen gradient passes are combined by the exact chain rule; a toy direct-gradient and finite-difference audit checks the implementation. A real one-update preflight must both improve the action preference gap and reduce chosen-completion NLL. Preflight weights are not reused.

The full decoder context is preserved. Only the output projection uses the last **256** positions; every supervised target is asserted inside that window for all train pairs, and a real-sample comparison checks its log probabilities against full projection. No context or supervised completion is silently truncated. Wired memory is capped at 6 GiB, cache at 256 MiB, and freed buffers are cleared after each pair. These limits do not describe total OS or other-process memory.

The action-only candidate (616 total updates) validated at **17/32**, including 5/10 exchanges, 1/2 fallback, 4/8 model handoffs and 2/10 required clarifications. It still guessed missing fields and made unsafe substitutions/returns. This diagnosis prompted a final revision before any new test inference; its records remain frozen.

## Recovery states and observed affordability facts

The final phase initializes from the action-136 adapter and runs **80 fixed updates**: 80 state pairs from **52 original train tasks**, learning rate 0.000003, seed 2026102743, the same exact action-span preference/NLL objective, batch 1, rank 8. The source task families are shared with earlier stages; there are no new independent training tasks.

Twenty pairs require a missing-field question **after actual identity/order/catalog reads** (or failed identity lookup when email is missing); the missing answer is absent from the prompt. Twenty matched continuations instead prefer the correct proposal after an actual answer. Sixteen pairs contrast required handoff with unsafe exchange/return; four preserve authorized stockout returns, eight preserve available exchanges, four preserve genuine returns and eight preserve status/ownership handoffs. Preflight weights are not reused. A first 1e-6 preflight exposed a large unsafe-action margin; its record and the pre-training revision to the earlier-tested 3e-6 rate remain archived.

`decision_facts.py` augments only completed tool results: all exchange candidates retain their exact option, availability and price difference, with a boolean gift-card coverage calculation. It contains no selected candidate, fixture answers, expected outcome or intent classifier. Static guidance distinguishes instrument facts from customer payment authorization and requires a question for missing reason/option/payment preferences. Both the app and all three new-flow test arms receive identical augmentation; the old-flow reference remains unchanged.

Full context and all completion targets remain covered; recovery permits up to 3,072 unpadded input tokens and projects only the last 256 output positions. The gradient, finite-difference, target-coverage and projection audits run before formal updates. Wired/cache limits remain 6 GiB/256 MiB.

The recovery candidate (696 total updates) validated at **24/32**: 4/10 exchanges, 0/2 authorized fallback, 8/8 model handoffs, 6/10 required clarifications and zero available-exchange return attempts. It restored all policy handoffs and missing reason/refund questions, but the return exception and gift-card choice remained weak.

## Final balanced conditional calibration and validation guard

A final trial continues from recovery-80. It uses **24 preference pairs over 20 controlled contexts from 12 original training worlds**, one seeded shuffle repeated for **two epochs / 48 updates**, learning rate 0.000003, seed 2026102744. Source worlds remain training-only. Stock and customer return permission are changed while identity, item and price facts remain fixed across related contexts. There are exactly eight preferred return, eight exchange and eight handoff labels. Reference trajectories for all 20 contexts pass the real validator/grader before training.

Return-fallback examples contrast both handoff and an unrequested substitute exchange. Available-stock controls preserve exchange despite the same conditional return wording. Missing authorization and insufficient-card controls preserve handoff. Same-tool gift-card controls compare the **unquoted payment-value token span**, since comparing identical action names would yield no signal. The NLL term still covers the full chosen completion. An initial mask included quote characters; the one-update prefix, source, preflight and mask repair are archived, and those weights were discarded. An independent audit checks all eight chosen/rejected payment-value masks.

The final choice compares the fixed recovery-696 and calibration-744 checkpoints using **validation only**, under `calibration_choice.py`, frozen before calibration training. Calibration must have zero exchange-as-return and no losses in overall, exchange, model handoff, required-clarification or direct-return successes, and no increase in question repeats. Among eligible candidates, prefer fallback successes, overall successes, fewer repeated questions and fewer calls. Otherwise retain recovery-696. The new test does not choose weights.

The five-stage validation figure combines pre-test engineering/training revisions; it does not isolate each change causally. The four-arm fresh test separately measures the latest combined flow with previous weights and the complete new training recipe under that same flow. All prior candidates and failures remain available.

SFT, sequence, action, recovery and calibration manifests bind their own inputs. Inherited `metrics` and initialization/weighting fields describe SFT; nested `preference`, `recovery` and `calibration` describe each later trial. `calibration_adopted` states whether its weights enter the final selected lineage. `selected_adapter_sha256` identifies the actual evaluated/served model. A calibration trial rejected by the guard still retains its raw training/validation records, but is excluded from selected-lineage time/update totals.

## Four-arm fresh test

| Arm | Weights | Workflow | Purpose |
| --- | --- | --- | --- |
| previous_old_flow | Preserved V3 adapter | V2 | New-test pre-change reference |
| previous_new_flow | Same V3 adapter | V4 | Combined engineering effect |
| base_new_flow | Unchanged MLX base | V4 | Base-model quality comparator |
| qlora_v4 | Validation-guarded recovery696 or calibration744 adapter | V4 | Bundled training effect |

Every arm uses the same 32 test tasks, pinned base/tokenizer, unconstrained greedy generation and **eight total model calls**. One deterministic trial per task; arms run serially and latency is descriptive. There are 10 available-exchange, 2 stockout-return, 8 handoff, 8 direct-return control and 4 cancellation tasks; these five groups are disjoint because direct returns exclude the separate fallback. Expected final operations are 10 exchanges, 10 returns, 8 handoffs and 4 cancellations. Twelve non-handoff operations are confirmed, twelve declined; all eight handoff tasks must leave orders unchanged.

Task success requires the correct operation, exact variant/payment/amount, identity/order/catalog prerequisites, required clarification and final confirmed/declined state. Wrong but schema-valid arguments fail. A return **attempt** on an available-exchange task counts as exchange-as-return even when rejected. Correct authorized stockout returns are excluded from that error metric.

Repeated question-field attempts count every additional model ask of the same field. Delivered repeated questions count only successful question results; blocked attempts remain visible separately. In these synthetic cases the scripted replies are usable and unambiguous; the metric does not judge whether every repeated field in unrestricted human conversation is unnecessary. Actual model-handoff success additionally requires a model `handoff` action. Controller-only recovery remains observable in raw outcomes and is not credited to the model-handoff metric.

Additional post-hoc handoff diagnostics divide verified handoffs by all actual model handoffs and count handoffs on tasks whose expected outcome was not handoff. They expose over-escalation, are reconciled from fixed raw traces, and do not alter checkpoint selection or promotion.

The predeclared local gate requires overall, exchange, fallback and handoff performance at least both new-flow comparators, zero available-exchange return errors, intact confirmation, and no increase in repeated-field attempts (a strict reduction when the previous comparator has repeats). All per-case regressions remain in the report. No test output selects checkpoints or changes these rules. Further tuning after these tests requires a different frozen test set.

## Reproduce

Use the root and training lockfiles, Apple Silicon MLX environment and pinned base provisioning in `training/README.md`. In a fresh checkout, archive delivered `reports/v4` and `docs/figures/v4`, create a working `reports/v4`, and retain the frozen `preference-manifest.json`, `action-manifest.json`, `recovery-manifest.json`, `calibration-manifest.json` and development evidence (`validation-240.json`, `validation-604.json`, `training-unregularized.json`, `training-action-only.json`, `validation-616.json`, `validation-696.json`, `training-recovery-only.json` and `preference-updates.json`) there. Work in a separate copy and archive only the final action-preference outputs before replay. Move any existing ignored V4 run folders aside. Preserve existing checkpoints and original frozen data.

These are stage entry points, not an unattended one-command replay. Each selector writes `training.json`; archive it as `training-action-only.json` after action selection and as `training-recovery-only.json` after recovery selection before the next selector. Delivered manifests intentionally bind exact preceding weight/validation identities, including report bytes. In a separate new experiment, changed weights or report hashes require that stage's `prepare` entry point and newly archived manifests; do not relabel a replay with the delivered fingerprints or mix its outputs into this completed experiment. Preserve the original failed-development evidence separately.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.audit
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.train --preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.select --preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.train
# Reuse the frozen preference pairs/protocol; do not re-run prepare over delivered artifacts.
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.action_preference preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.action_preference train
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.action_select
# Archive this working run's training.json as training-action-only.json first.
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.recovery preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.recovery train
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.recovery_select
# Archive the reproduced recovery training.json as training-recovery-only.json first.
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.calibration preflight
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.calibration train
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.resource_runner training.v4.calibration_select
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v4.resource_runner training.v4.facts_evaluate
MPLCONFIGDIR=.mpl-cache uv run --group viz python -m training.v4.visualize
uv run python -m training.v4.verify_results
```

Supplied frozen data are not regenerated during normal reproduction. `datasets/manifest.json` binds original facts, labels, SFT objective, runner, workflow, selection and scoring sources. `reports/v4/preference-manifest.json` and `reports/v4/action-manifest.json` and `reports/v4/recovery-manifest.json` bind the validation-guided revisions before held-out inference. Preserve all five manifests when reproducing in a fresh reports folder; the SFT-480 initialization fingerprint must match exactly. Original `select.py` remains an archived SFT selection procedure and is not the final preference-stage entry point. Reading reference labels for schema/token audits is not model test inference. Training and selection refuse overwrites; partial test evaluation resumes only with identical source/training identities. Ollama remains a separate GGUF/JSON-Schema integration; its live demonstrations are excluded from controlled MLX scores.

The four-arm evaluation also needs the preserved V3 adapter fingerprint; reproduce it with the round-3 method if its ignored local weights are absent. Historical failed development evidence stays frozen during final-stage replay. Exact checkpoint identity guards are intentional: a replay with different numerical weights requires its own recorded run and manifests, not edited delivered fingerprints.

The final calibration validation initially failed with a Metal allocation error before completing any case. The selected weights received no further update. `resource_runner.py` limits wired allocation to 6 GiB and free-buffer cache to 256 MiB before calling the unchanged frozen final selector/evaluator. The same allocator settings apply to all four test arms; decoding, inputs and selection rules are unchanged. See `reports/v4/inference-resource-repair.json`.

`reports/v4/runtime-environment.json` records background Ollama contention observed during the serial test. Other models were not stopped. Timing is retained for inspection only; this run supports no controlled latency/throughput improvement claim.
