# Round 3: distinguish exchange, return and conditional fallback

The second adapter produced valid JSON and accepted tool arguments but changed all 12 available-exchange requests into returns on its test. Those failures are now development evidence. This round freezes new data and compares the unchanged MLX base, the preserved v2 adapter and a new v3 adapter on exactly the same new test tasks.

## Changes being evaluated

1. **Counterfactual data.** Available/unavailable fallback pairs have identical customer requests, identities, orders and prices; only the requested variant's stock changes. Direct return versus exchange pairs use the same catalog/order and different customer intent, including an explicit “do not exchange” negative control. Prices include charges and partial credits. This prevents a blanket exchange or blanket return policy from passing every task.
2. **Targeted training snapshots.** Correct available-exchange terminal decisions are oversampled threefold. Customer corrections after a deliberately wrong return are oversampled twofold; the erroneous model action itself is not supervised. Additional direct returns after catalog inspection prevent “catalog read means exchange” from becoming a shortcut. Required clarifications and upstream tools remain represented.
3. **Actual loss change.** Fast-tokenizer character offsets locate the JSON tool-name value. Only target tokens overlapping that value receive weight 8; other completion tokens receive weight 1, and prompt/padding tokens receive zero. The audited low-level MLX trainer uses this weight tensor in cross-entropy. A numerical equal-error check gives a tool/argument loss ratio of approximately 8. This is not a renamed prompt or a template-only model profile.
4. **Business-based checkpoint selection.** Positive callback checkpoints at 119 and 239 completed updates are exported and loaded in a separate clean process. All 16 validation tasks run through the real tools. Selection lexicographically maximizes available-exchange successes, total task successes, negative exchange-as-return cases and negative weighted completion loss. The initial state is diagnostic. Training completes 240 updates; final-update weights are retained but are not an eligible selection candidate under this predeclared schedule.

All inference arms keep the **same v2 WorkflowStore, system prompt, model-visible state, tools, tokenizer, greedy decoding and eight-call budget**. No new keyword routing, forced exchange tool, JSON grammar, repair or oracle fallback is added. The measured difference is the combined data/objective/selection training recipe; this experiment does not isolate each component's individual effect.

## Frozen data

There are **128 training / 16 validation / 48 test tasks** in 16 families, with **480/62/186 reference snapshots**. The training set contains 200 correct exchange decisions, 64 return decisions, 32 handoffs, 16 cancellations, 40 clarification requests and 128 read-tool decisions. Duplicated weighting snapshots are not additional unique tasks.

The counterfactuals intentionally share worlds within each split: 112/14/42 distinct order IDs for 128/16/48 tasks. Every session has an isolated order key. Worlds and six-character IDs are disjoint across splits and previous experiments; customer phrasing is split-specific and prices are generated independently. The author, schema, policy, domain language and most task families remain shared; all data are self-authored synthetic examples, not independently annotated human retail conversations. Counterfactual tasks are correlated and no statistical significance or production generalization is claimed.

The test contains 15 requests that should complete an available exchange. The primary misclassification numerator is the number of those tasks with any `propose_return` action, even if it has valid arguments. Correctly authorized returns after an unavailable requested exchange are excluded. Direct-return negative controls and unauthorized returns on exchange/handoff requests are reported separately. Task success still requires the exact operation, option, payment, amount, final state and required customer information. Field-keyed scripted replies do not independently judge the wording of every question.

The longest training snapshot is 1,590 tokens, below the 2,048-token cap. Both prompts and exact name-token offsets are audited, and no examples are truncated. Reference and correction trajectories execute against the actual validator; every reference outcome passes.

## Local training and comparison

The pinned official `Qwen/Qwen3-4B-MLX-4bit` base/revision is unchanged. Training starts from that base rather than continuing the rejected adapter. QLoRA uses rank 8, scale 16, last eight layers, dropout 0, batch 1, learning rate 0.00005, seed 202610273 and 240 optimizer updates. Only the new training/validation data can affect selection. New test model outputs are generated after selection finishes.

The predeclared local gate requires exchange completion to improve over v2 and be no worse than base, no exchange-as-return cases on available exchanges, all ordinary exchange cases to pass, overall success not below either comparator, direct-return success no worse than base, no paired non-exchange regressions against base, and intact confirmation invariants. It is a prototype criterion, not deployment approval or a statistical guarantee. All failed tasks and paired regressions remain in the report regardless of the gate.

A one-update preflight checks the weighted trainer and clean-process adapter reload using train/validation data only. Its weights are not reused for formal training, its result is not test accuracy, and the formal runner initializes independently. The v1/v2 weights, scores, raw traces and charts are preserved unchanged.

## Reproduction

Use the existing root/training lockfiles and pinned base-provisioning instructions. In a fresh source checkout, first move supplied `reports/v3` and `docs/figures/v3` to reference folders and create an empty `reports/v3`. Existing ignored run folders must also be moved aside; never overwrite original checkpoints or frozen data.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v3.audit
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v3.train
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v3.evaluate
MPLCONFIGDIR=.mpl-cache uv run --group viz python -m training.v3.visualize
uv run python -m training.v3.verify_results
```

Validation children load exported weights in an independent process so inference uses the same adapter-loading path as serving. Exact candidate fingerprints and all validation trajectories are saved. The training run and completed evaluation refuse overwrite; partial evaluation resumes only if all source and model identities match. `datasets/manifest.json` freezes training/validation/test facts, labels, masks, objective, runner, selection and scoring source hashes before test inference. Reading reference test labels for schema/token audits is not model inference.

Further tuning on these results requires another frozen test set. Ollama remains a separate GGUF/JSON-Schema integration; its scores must not be substituted for the controlled MLX comparison. All payments and approvals remain simulated.
