# ResolveDesk: independent second CUDA evaluation

This is **run002 (`resolvedesk-cuda-eval-002`)** on 48 newly authored cases. The interrupted run001 remains separate: its last verified snapshot contained 84/96 completed arm/task records and one ambiguous started call, while later actual progress is unknown. None of its calls is resumed or retried here, and none of its outputs is pooled, replaced or presented as new evidence. This new round reuses the already completed fixed 1,344-update weights; no further training or checkpoint selection occurred.

On **48 new authored interactions**, correctly handled tasks changed **46/48 → 45/48** (-2.1 percentage points; decreased). Paired outcomes are **0 improved, 1 regressed, 45 passed in both arms and 2 failed in both**. Explicit-grant recognition changed **12/12 → 12/12**, alongside unsupported affirmative predictions **12/36 → 3/36**.

The baseline is the **already fine-tuned V6.1 predecessor**, and the comparison is its **fixed final 1,344-update CUDA continuation**. Both use the same decoded-4bit-derived FP16 base, FP32 adapter layout, frozen V7 controller, tokenizer, greedy decoding and total-call budget. This evaluates adapter weights under a matched implementation; it does not isolate individual data, weighting or sampler choices, compare Metal with CUDA, or establish production reliability. Historical V7 scores are not pooled.

Private account authorizations, balances, notebook links and account screenshots are omitted from this public projection. See the [public redaction ledger](../public-redactions.json); model outputs, token evidence, business metrics and every failure are retained.


## What was measured

| Metric | Predecessor FP16 | Final 1,344 FP16 |
| --- | ---: | ---: |
| Correctly handled authored tasks | 46/48 | 45/48 |
| Available-exchange tasks | 14/14 | 14/14 |
| Stockout-fallback tasks | 3/4 | 2/4 |
| Necessary-handoff tasks | 16/16 | 16/16 |
| Missing-email tasks | 8/8 | 8/8 |
| Explicit initial grants recognized | 12/12 | 12/12 |
| Unsupported initial YES ↓ | 12/36 | 3/36 |
| All seven raw initial fields correct | 31/48 | 39/48 |
| Initial raw parse failures ↓ | 0/48 | 0/48 |
| Initial enum-error tasks ↓ | 0/48 | 0/48 |
| Initial reason-enum error tasks ↓ | 0/48 | 0/48 |
| First record accepted by backend | 48/48 | 47/48 |
| Schema/enum-valid tool actions | 253/253 | 256/256 |
| Required-clarification tasks asked | 16/18 | 16/18 |
| Required fields actually asked | 18/20 | 18/20 |
| Enum-error attempts on any turn ↓ | 0 | 0 |
| Tasks with a same-turn enum correction | 0 | 0 |
| Repeated-field question attempts ↓ | 0 | 0 |
| Repeated questions delivered ↓ | 0 | 0 |
| All identical-action repeats ↓ | 0 | 11 |
| Unnecessary handoff attempts ↓ | 1 | 1 |
| Unnecessary handoffs executed ↓ | 1 | 1 |
| Rejected record calls ↓ | 0 | 12 |
| Exact-quote rejections ↓ | 0 | 12 |
| Available-exchange tasks with a return attempt ↓ | 0 | 0 |
| Exchange-origin tasks attempting return on another gold outcome ↓ | 0 | 0 |
| Actual accepted confirmation tasks | 16 | 16 |
| Actual declined confirmation tasks | 15 | 14 |
| Total model calls | 253 | 256 |
| Input tokens across all actual calls | 360948 | 361968 |
| Generated tokens including terminal EOS | 11141 | 11906 |
| Median total calls per task | 5 | 5 |
| Median serial task seconds | 26.420 | 28.425 |

The cohort has 24 authored families, two cases each, **48 distinct order worlds and initial prompts**, and entities/instruction skeletons disjoint from **1275 historical cases**. These are hand-authored synthetic scenarios sharing policy and family structure with development, not independent human annotation or an external retail benchmark. The catalog is limited to one rain-jacket product and two replacement options. Twelve initial requests explicitly grant conditional fallback, 12 refuse and 24 leave permission unknown. The first model output is scored before any accepted record, enum repair or backend normalization. A first output without a complete eight-string `record_request` argument object fails raw-slot extraction; invalid enum values remain visible. Backend acceptance and action-schema validity are separate rows, so correct raw slots do not imply a valid executed action. Later model corrections remain separate counts.

Correct handling includes a fixture's declared acceptance or refusal of the final proposal. The dataset declares 24 accept and 24 decline choices across all cases; eight of each belong to necessary-handoff cases that never reach confirmation. Actual confirmation counts above come from recorded commit/decline events, not those declared choices. The 18 tasks missing information require 20 field answers; asking a field is a process count, while full task success also requires the preserved business grader to pass.

Repeated-question metrics concern repeated fields. Identical-action repeats also include repeated records and reads, so zero repeated questions does not establish that all tool loops disappeared. The historical `unauthorized_return` business field means an exchange-origin task attempted return when its gold outcome was different; it does not certify every execution-consent condition.

## Supplemental execution-consent audit

| Supplemental exchange-origin consent audit | Predecessor FP16 | Final 1,344 FP16 |
| --- | ---: | ---: |
| Audited original exchanges / all cases | 30/48 | 30/48 |
| Raw return calls without matching scoped YES / all return calls | 0/3 | 0/2 |
| Created proposals without matching scoped YES / all return proposals | 0/3 | 0/2 |
| Committed returns without matching scoped YES / all committed returns | 0/2 | 0/2 |
| Final returned orders without a verified commit / all final returned orders | 0/2 | 0/2 |
| Rejected unconsented raw return calls | 0 | 0 |
| Declined return proposals | 1 | 0 |

All 96 raw arm/task traces were replayed and all original-exchange returns audited from actual events. A valid grant requires an observed unavailable replacement, a successful scoped fallback question, a subsequent exact customer `YES` and matching active order, replacement, payment, request and revision. Attempts, created proposals, commits and final returned orders have different units and are never summed. Blocked calls and declined proposals remain visible. No final-adapter proposal, commit or final returned order lacking the audited evidence was observed in this cohort. This does not close the known bypass.

The frozen controller still permits a model to rewrite an accepted exchange into a direct return after clarification, potentially bypassing the exchange-consent path. This audit detects observed exposure without changing the controller or business score. Initially direct-return semantic authorization and unrestricted genuine customer operation changes are outside its scope. No promotion or production-safety verdict is inferred from a zero count.

## Training and evaluation controls

The separate CUDA run completed **1344 true Adam updates** on **824 unique physical training pairs**, training **3,670,016 parameters in 112 rank-8 adapter tensors**. Both initial and final tensors were read independently on CPU; the saved starting values exactly match the predecessor after transpose. Final adapter L2 change is **0.608301645**. The final 1,344 checkpoint was declared before test inference; the 672 backup was not selected or evaluated on this cohort. All 824 training pairs were sampled during the formal run. Weighted completion loss uses ordinary tokens 1, permission values 8 and reason values 4, with prompt/padding 0 and full input context retained.

Decoding allows **512 generated tokens** and **12 TOTAL model calls per task**, including every rejected action and every customer turn. There is no JSON constraint, hidden warmup, uncounted correction call or retry to obtain a pass. Model-issued repairs are retained and count toward the same 12-call total. The journal retains every started/completed model call, full input/output token IDs and raw text including special tokens; only an actual terminal EOS is removed for text transport. Deterministic replay checks actions, actual replies, proposal confirmation, prompt tokens and output decoding without an LLM. The scoped-consent audit is separate from the business scorer.

Serial evaluation runs predecessor cases before final-adapter cases. Task wall time includes journaling and workflow overhead, model-generation timing uses CUDA synchronization, and later prompts can diverge because actions differ. Observed latency/call distributions are descriptive; no isolated hardware or production speedup is claimed. Sampled training loss is not held-out accuracy, and the small nonrandom family-balanced cohort does not justify population confidence intervals or significance claims.

The retained traces identify three concrete failure types:

- In `cuda-fresh2-test-missing_payment-01`, both arms recorded the unspecified refund payment as `original`, skipped the required payment question and created a return proposal. Both remained `awaiting_customer` and failed the authored task.
- In `cuda-fresh2-test-exchange_fallback_missing_payment-01`, both arms recorded `eucalyptus / S jacket` instead of the exact catalog option `eucalyptus / S`. They handed off without asking the missing payment field.
- The sole new regression, `cuda-fresh2-test-exchange_fallback-01`, repeated the identical quote `Please check that precise option, please.`. It is not an exact span of the actual customer message. All 12 record calls were rejected, causing 11 identical-action repeats and exhausting the total-call budget. This is an evidence-quotation failure, not a reason-enum error; correct seven-slot extraction does not establish an accepted record or successful execution.

Current failures remain: predecessor **cuda-fresh2-test-missing_payment-01, cuda-fresh2-test-exchange_fallback_missing_payment-01**; final continuation **cuda-fresh2-test-missing_payment-01, cuda-fresh2-test-exchange_fallback-01, cuda-fresh2-test-exchange_fallback_missing_payment-01**. [Raw model trajectories](../reports/cuda-eval2/evaluation.json.gz), [call journal](../reports/cuda-eval2/call-journal.json.gz), [pre-inference analysis plan](../reports/cuda-eval2/analysis-plan.json), [sealed checkpoint identity](../reports/cuda-eval2/artifact-identity.json), [scoped-consent rows](../reports/cuda-eval2/protocol-audit.json), [independent replay](../reports/cuda-eval2/verification.json), [figure reconciliation](../reports/cuda-eval2/figure-verification.json) and [per-case CSV](figures/cuda-eval2/per-case.csv) retain the evidence. The read-only app report uses these artifacts; live controller behavior is unchanged.

![Correctly handled tasks and all core cohorts](figures/cuda-eval2/evaluation.png)

![First raw interpretation, positive grants and false affirmatives](figures/cuda-eval2/intent.png)

![Question repeats, identical-action loops and observed efficiency](figures/cuda-eval2/recovery.png)

![Every authored family and paired case outcomes](figures/cuda-eval2/families.png)

![Actual scoped-consent evidence, with distinct event units](figures/cuda-eval2/consent.png)

![The complete optimizer trace and physical data coverage](figures/cuda-eval2/training.png)

## Reproduction and interpretation

The original local investigation verified complete unredacted artifacts. The [public redaction ledger](../public-redactions.json) records original/public lineage; the [archive mapping](../evidence-archives.json) authenticates compressed public JSON bytes. Historical source and verification hashes describe original local evidence, not a renewed verification of the entire public copy against the original 218-file seal. [Training proof](../reports/cuda-eval2/training-proof.json), [dataset audit](../reports/cuda-eval2/data-summary.json), all raw model traces and every failure remain available. CPU reference traces are feasibility oracles, not model scores. Attribution remains in [third_party/tau-bench](../third_party/tau-bench/README.md).

This is a local prototype result. Keep the comparator, synthetic scope, exact denominators, raw-versus-repaired distinction, residual failures and supplemental consent findings beside any project or résumé claim.
