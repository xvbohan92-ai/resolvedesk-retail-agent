# ResolveDesk: Correct task handling increased on the paired40 subset under V8 flow

> **Public projection:** These reports publish checked numerical results. New raw model outputs, prompts, token arrays, cloud snapshots, weights and account authorizations remain local. Public checks reconcile CSVs, denominators, paired outcomes and figure hashes; they do not repeat the original token/business replay or recertify its private source seal.

This is a **separately authorized retrospective stage analysis** of the fixed common prefix: 40 cases × 3 arms, 120 complete traces and 657 calls. The original144 remains incomplete; its terminal summary and full-result gate are unchanged. This is not the complete48 evaluation and no model call was rerun. The prior comparison adapter was already fine-tuned.

The independent CPU audit covers all 136 complete records, 765 retained responses and the one unknown call's request. Effect metrics below use only the 120 paired rows, excluding the 16 complete tail records.

Correct business handling includes the customer's approval or decline; a correctly declined proposal preserves the order. Exchange/fallback success counts are not executed transaction counts.

| Metric | Prior / V7 | Prior / V8 | Saved V8 / V8 |
| --- | --- | --- | --- |
| Correct business handling (including correct declines) | 30/40 (75.0%) | 30/40 (75.0%) | 33/40 (82.5%) |
| Available-exchange subset | 5/10 (50.0%) | 7/10 (70.0%) | 8/10 (80.0%) |
| Stockout-fallback subset | 1/2 (50.0%) | 2/2 (100.0%) | 2/2 (100.0%) |
| Necessary handoff | 13/16 (81.2%) | 10/16 (62.5%) | 13/16 (81.2%) |
| Missing-email subset | 2/4 (50.0%) | 2/4 (50.0%) | 3/4 (75.0%) |
| Joint first raw fields | 24/40 (60.0%) | 24/40 (60.0%) | 27/40 (67.5%) |
| Initial explicit-YES recall | 8/8 (100.0%) | 7/8 (87.5%) | 8/8 (100.0%) |
| Unsupported first YES ↓ | 1/32 (3.1%) | 1/32 (3.1%) | 1/32 (3.1%) |
| First-output parse failures ↓ | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| First reason-enum errors ↓ | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Repeated-question attempts | 0 | 0 | 0 |
| Repeated questions delivered | 0 | 0 | 0 |
| All identical-action repeats | 48 | 15 | 9 |
| Unnecessary handoff attempts | 5 | 0 | 0 |
| Unnecessary handoffs executed | 5 | 0 | 0 |
| Model calls | 236 | 208 | 213 |

workflow holds prior weights fixed and compares V7 with V8 controller/prompt flow. training_under_v8 holds V8 flow fixed and compares prior with saved V8 continuation weights. combined changes both. These do not isolate a particular data, sampling or loss choice, and do not compare vanilla Qwen against fine-tuning.

| Contrast | Both pass | Both fail | Improved | Regressed |
| --- | --- | --- | --- | --- |
| workflow | 25 | 5 | 5 | 5 |
| training_under_v8 | 28 | 5 | 5 | 2 |
| combined | 27 | 4 | 6 | 3 |

Coverage of key conditions (counts, not correctness rates):

| Condition | Included40 | Original48 |
| --- | --- | --- |
| natural_initial_literal_quote_provenance | 40 | 48 |
| missing_payment | 4 | 8 |
| stockout_return_fallback_with_missing_payment | 0 | 2 |
| available_exchange_with_missing_email | 0 | 2 |
| two_missing_email_and_payment_fields | 0 | 2 |
| available_exchange_explicit_return_refusal | 0 | 2 |
| ambiguous_option_needs_customer_choice | 2 | 2 |
| initial_known_option_exchange | 22 | 28 |
| quoted_permission_hard_negative | 2 | 2 |
| necessary_handoff_gold_cases | 16 | 16 |
| gold_non_handoff_cases | 24 | 32 |

Omitted branches (not zero-scored):

| Family | Included / original | Interpretation |
| --- | --- | --- |
| exchange_available_refusal | 0/2 | The available-exchange explicit-refusal branch is omitted; other NO predictions cannot establish its improvement. |
| exchange_fallback_missing_payment | 0/2 | Stockout fallback with a missing payment method is not covered; this cannot establish that fallback fix. |
| exchange_missing_email | 0/2 | The dedicated available-exchange missing-email and operation-preservation branch is omitted. |
| missing_email_payment | 0/2 | The combined missing-email/payment multi-turn recovery branch is omitted. |

The common40 prefix results from fixed arm-block order and a deadline, not random truncation or a new independent benchmark. Of48/48/40 completed records,16 old-weight tail records are excluded. Seven new-weight tasks were unstarted; one task has an issued call with unknown response status. Eight missing records are neither successes nor failures; no automatic replay or resume is allowed. Do not extrapolate subset rates to 48 cases or claim coverage of omitted V8 fixes.

Case mix shifted: all16 necessary-handoff cases remain (16/40 versus16/48 originally); all eight missing cases are non-handoff, reducing that group from32 to24. Matched contrasts within the three40-case arms remain descriptive, but their overall rates cannot be directly compared with the original48 or historical96-trace evaluation.

Quote correction, product-suffix/option grounding and the distinction between original order and payment method are not separately randomized contrasts; discuss only observed selected traces, not a claim that all these fixes work from aggregate business handling alone.

Only first raw outputs earn initial-field credit; later repair cannot erase them. Unsupported initial YES is not an executed unauthorized transaction. Attempted/delivered repeated questions and all repeated actions use distinct units. Failed/corrective calls share the 12-call task budget. Same-author synthetic scenarios, finite catalogs and scripted replies limit generalization. Serial timing includes fixed arm order, trajectories and logging; it is not isolated hardware speed. Zero observed scoped-consent violations do not certify production safety.

The following concern selected exchange-origin tasks; units must not be summed:

| Metric | Prior / V7 | Prior / V8 | Saved V8 / V8 |
| --- | --- | --- | --- |
| Exchange-origin tasks | 24 | 24 | 24 |
| Raw return attempts | 1 | 2 | 2 |
| Return proposals | 1 | 2 | 2 |
| Return commits | 1 | 1 | 1 |
| Final return orders | 1 | 1 | 1 |
| Attempts without scoped YES | 0 | 0 | 0 |
| Proposals without scoped YES | 0 | 0 | 0 |
| Commits without scoped YES | 0 | 0 | 0 |
| Final orders without verified commit | 0 | 0 | 0 |
| Rejected attempts without scoped YES | 0 | 0 | 0 |

The saved historical V8 run completed 1344 updates and sampled 888/1100 physical pairs across 96 source worlds. Rank 8, 112 FP32 tensors and 3,670,016 parameters; this evaluation performed 0 updates. Historical loss is not new training or accuracy.

Every failed selected record:

| Arm | Case | Final status | Calls |
| --- | --- | --- | --- |
| prior_weights_old_flow | v8-eval2-test-exchange_fallback-01 | handoff | 6 |
| prior_weights_old_flow | v8-eval2-test-foreign_order-01 | running | 12 |
| prior_weights_old_flow | v8-eval2-test-missing_email-01 | running | 12 |
| prior_weights_old_flow | v8-eval2-test-exchange_missing_payment-01 | handoff | 5 |
| prior_weights_old_flow | v8-eval2-test-exchange_gift-00 | handoff | 5 |
| prior_weights_old_flow | v8-eval2-test-exchange_gift-01 | handoff | 6 |
| prior_weights_old_flow | v8-eval2-test-exchange_shortfall-00 | running | 12 |
| prior_weights_old_flow | v8-eval2-test-exchange_fallback_available-00 | handoff | 5 |
| prior_weights_old_flow | v8-eval2-test-exchange_fallback_available-01 | running | 12 |
| prior_weights_old_flow | v8-eval2-test-exchange_stockout_missing_email-00 | running | 12 |
| prior_weights_new_flow | v8-eval2-test-return-01 | failed | 6 |
| prior_weights_new_flow | v8-eval2-test-forbidden_status-00 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-foreign_order-01 | failed | 6 |
| prior_weights_new_flow | v8-eval2-test-exchange_missing_option-00 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_gift-00 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_shortfall-00 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_shortfall-01 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_fallback_available-01 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_stockout_missing_email-00 | failed | 3 |
| prior_weights_new_flow | v8-eval2-test-exchange_stockout_missing_email-01 | failed | 9 |
| v8_weights_new_flow | v8-eval2-test-return-01 | failed | 6 |
| v8_weights_new_flow | v8-eval2-test-missing_payment-00 | failed | 6 |
| v8_weights_new_flow | v8-eval2-test-foreign_order-01 | failed | 6 |
| v8_weights_new_flow | v8-eval2-test-exchange_gift-01 | failed | 3 |
| v8_weights_new_flow | v8-eval2-test-exchange_shortfall-00 | failed | 3 |
| v8_weights_new_flow | v8-eval2-test-exchange_fallback_available-01 | failed | 3 |
| v8_weights_new_flow | v8-eval2-test-exchange_stockout_missing_email-01 | failed | 9 |

![Stage-only evaluation](figures/v8-eval2-partial40/evaluation.png)

![Stage-only intent](figures/v8-eval2-partial40/intent.png)

![Stage-only recovery](figures/v8-eval2-partial40/recovery.png)

![Stage-only families](figures/v8-eval2-partial40/families.png)

![Stage-only consent](figures/v8-eval2-partial40/consent.png)

![Stage-only training](figures/v8-eval2-partial40/training.png)

[CSV120](figures/v8-eval2-partial40/per-case.csv) · [Paired40](figures/v8-eval2-partial40/case-mapping.csv) · [Metric denominators](figures/v8-eval2-partial40/metrics.csv) · [Historical training](figures/v8-eval2-partial40/training.csv)

[Public result and local-verification summary](../reports/v8-eval2-partial40/public-summary.json) · [120 public case rows](figures/v8-eval2-partial40/per-case.csv) · [Published scope](../PUBLICATION_STATUS.md) · [Chart values and public hashes](figures/v8-eval2-partial40/values.json) · [Failure analysis](v8-eval2-partial40-failure-analysis.md)
