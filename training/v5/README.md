# Round 5: typed intent, stockout returns and bounded human handoff

This round continues the preserved V4 QLoRA adapter and separates a model's customer-intent interpretation from observed-fact business rules. The controller computes a permitted next action from the recorded prediction and completed tools. The model must still interpret the customer, record/update slots and issue every tool action. Errors consume calls; no corrective action is silently executed.

An exact quotation proves only that text occurred in a customer turn. It does **not** semantically certify consent. The controller can act on a wrong but well-formed model interpretation; independent intent labels, business grading and explicit proposal review expose this limitation. This is a simulated prototype, not production authentication or a payment system.

## Frozen experiment

- 64 training, 32 development and 32 new held-out tasks over 16 shared families.
- 56/28/28 distinct order worlds; deliberate within-split stock/intent counterfactuals share worlds. Fresh eight-character entities and seeded prices are separated across splits and earlier rounds.
- 328 clean reference decisions plus 16 repeated initial request contrasts, totaling 344 training snapshots; repeated rows are not extra tasks.
- Every reference trajectory executes through actual tools and passes the preserved independent business grader. Labels and customer replies are self-authored, programmatic and correlated; there is no independent human annotation or generalization claim.
- Actual model interpretation has seven fields: operation, email, order ID, replacement option, payment preference, cancellation reason and conditional-return permission. Missing values remain unknown. After an actual reply the model merges prior slots and the new reply; unsolicited instructions invalidate stale intent.
- The policy gate reads only predicted slots and completed identity/order/catalog results. Neither runtime prompt nor gate reads expected outcomes, case family or scripted answers. Unit tests mutate those hidden fields and verify unchanged runtime behavior.

The original four pre-training validation probes exposed guessed payment preferences and quotation paraphrases. A whole-message-copy prototype was revised before formal freeze to allow exact contiguous customer-text quotations. Earlier probe/source evidence is archived in reports/v5. Handoff explanations need not match a fixed sentence when the observed-fact gate permits handoff. These development probes are not controlled test results.

## Actual training

Base: official Qwen/Qwen3-4B-MLX-4bit, pinned revision 52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495. The nine local model/tokenizer/license files are verified. Initialization is the exact selected V4 adapter fingerprint, not a preflight or rejected calibration adapter.

Rank 8 in the final eight layers, 3,670,016 trainable parameters, batch 1, Adam learning rate 0.00001, seed 202610275 and 512 fixed updates. Training shuffles without replacement per 344-row cycle. All completion tokens have weight one; prompt/padding have zero. There is no action-name reward or preference loss this round.

The full decoder context is retained. Only the output projection uses the last 512 positions; every supervised target is asserted inside it. The real preflight compares selected against full output projection and requires a same-sample loss reduction after one actual gradient update. Preflight weights are discarded. The 3072-token input cap rejects oversized data rather than truncating it.

Two checkpoints are saved after exactly 256 and 512 completed updates. Selection uses the 32-task validation set only: first zero available-exchange return and unauthorized-return attempts, then maximize the minimum exchange/fallback/handoff success fraction, overall success, fewer executed/attempted unnecessary handoffs, repeated fields and calls. Ties retain the earlier candidate. No test output selects weights or changes a prompt or rule.

MLX wired allocation is capped at 6 GiB, free-buffer cache at 256 MiB. The resource helper waits when competing Ollama caches exceed 6 GiB and never unloads them. Timing includes resource waiting where present and is descriptive only. Peak active MLX memory excludes cache, OS and other processes.

## Four-arm test

| Arm | Adapter | Controller |
| --- | --- | --- |
| previous_old_flow | Preserved selected V4 | Preserved V4 |
| previous_new_flow | Same V4 | Typed intent + policy gate |
| base_new_flow | No adapter | Same typed flow |
| qlora_v5 | Validation-selected continuation | Same typed flow |

All four arms share the same 32 tasks, base/tokenizer, greedy unconstrained decoding, 512-token generation cap and **twelve total calls**, including intent records and every rejected action. Original V4 scores used eight calls and a different test; those scores are archived, not pooled. One trial per case, scripted customer answers and confirmations; no confidence intervals or speedup claims.

Task success uses the preserved independent scorer: correct kind, item/variant, payment, amount, identity/order/catalog reads, required answer, and confirmed/declined state. A model attempt to return an available exchange counts as an error even if blocked. Authorized unavailable-stock returns are excluded from that error. Handoff success requires an actual model handoff action and complete prerequisites. Both unnecessary model handoff attempts and actual executed handoffs remain visible.

Initial intent is assessed against seven independently authored case fields; missing/invalid initial records fail. Old-flow arms do not expose a comparable intent record and are reported as unavailable, not zero. This field-level metric never substitutes for business correctness. For example, an extracted conditional return may be correct while the requested option or payment is wrong.

The predeclared release rule requires all two stockout fallback tasks correct, at least 8/10 available exchanges, at least 7/8 required handoffs, at most one executed unnecessary handoff, zero repeated-field/available-exchange return/unauthorized-return attempts, all confirmation invariants and overall success at least both typed-flow comparators. Relative baseline equality cannot excuse 0/2 fallback. This gate is a small local prototype criterion, not production certification.

## Audit or reproduce

Run root engineering tests and training.v5.verify_results to audit delivered source/trace/CSV/figure identities; ignored model files are not required for those checks. Real training needs Apple Silicon, the training lockfile, the pinned base provisioning in training/README.md and the exact ignored V4 initialization adapter. Reproduce V4 first if those local weights are absent.

Delivered paths refuse overwrites. Use a separate experiment copy, archive its reports/v5 and ignored V5 run directories, and retain the frozen datasets/manifest. Run training.v5.train --preflight, then training.v5.train, training.v5.select and training.v5.evaluate in separate GPU processes. The selected service entry point is training.v5.serve. Training does not reuse preflight weights. Partial test evaluation resumes only when source/training fingerprints match exactly.

A new initialization, data, protocol or implementation requires its own prepared/frozen manifest and a fresh test if it is tuned after this test. Do not edit delivered fingerprints to relabel a changed run. The reproduction target is inspectable code and actual recorded artifacts; identical numerical weights across different hardware/software are not guaranteed.

Visualization, independent verification, serving, API probes and documentation helpers are excluded from the training-input manifest because they do not change model inputs/selection/grading. Their outputs record their own source or raw-data fingerprints. Model-facing code, config, datasets, runner, selection rule and core policy are frozen. Historical V1–V4 inputs and reports remain unchanged.
