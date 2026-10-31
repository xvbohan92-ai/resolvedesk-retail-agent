# Publication status

The latest addition documents **completed V8 training and a separately authorized retrospective common40 analysis**. It supersedes the earlier description of V8 as a zero-update formal run; the original bootstrap failure remains historical evidence.

| Evidence | Actual status | Scope |
| --- | --- | --- |
| Saved V8 continuation | Completed 1,344 optimizer updates | 112 rank-8 FP32 adapter tensors, 3,670,016 trainable parameters; 888/1,100 unique SFT pairs sampled from 96 source scenarios |
| Original three-arm V8 evaluation | **Incomplete: 136/144 complete records** | 48/48/40 by arm; 765 retained responses and one issued call without a retained response; no retry or full-result claim |
| Common40 stage analysis | 120 matched records / 657 calls | Retrospectively authorized, fixed nonrandom prefix, 20/24 families; local CPU evidence audit and six PNG/SVG chart groups |
| Earlier complete CUDA run002 | 96 matched records / 509 calls | A separate 48-case cohort; its extraction gains and overall business regression are preserved |

On the common40 subset, correct business handling is **30/40 → 30/40 → 33/40** for prior weights/V7 flow, prior weights/V8 flow and saved V8 weights/V8 flow. With V8 flow fixed, five cases improve and two regress. Initial payment accuracy declines from 39/40 to 37/40, unsupported initial YES remains 1/32, and calls increase from 208 to 213. Correct declines can count as business success; these are not executed-transaction rates.

The eight missing complete records are not scored as model failures: seven tasks were unstarted and one contains the ambiguous issued call, which must not be replayed. Four recovery/consent families are absent, all 16 handoff cases remain, and the truncated case mix differs from the original 48. This is not a complete 144-record result, an external benchmark or evidence of production readiness. No new inference, retraining, default-provider promotion or pooling with historical experiments is claimed by this analysis.

See the [English stage report](docs/v8-eval2-partial40.md), [中文报告](docs/v8-eval2-partial40.zh-CN.md), [failure analysis](docs/v8-eval2-partial40-failure-analysis.md) and [public summary](reports/v8-eval2-partial40/public-summary.json).

## Public projection boundary

For this V8 update, the repository publishes sanitized aggregates, per-case derived results, CSVs, PNG/SVG figures and narrative failure analysis. **V8 raw traces, token arrays, call journals, model weights, databases, credentials, account/budget/authorization records, operator files and cloud snapshots are excluded.** The original evidence remains in the local archive and is not replaced by this export.

The local audit verified retained token/tool evidence, but the public update supports only inspection and numerical reconciliation of the published projection. It does not distribute everything needed to repeat the private token/event replay. Public metadata and hashes describe the redacted projection; the original local byte seal does not certify changed public files. The latest [public reconciliation script](scripts/reconcile_partial40_public_results.py) is distinct from model inference or replay.

Previously published historical synthetic trajectories and token evidence stay in their own experiment paths. Large historical JSON/JSONL artifacts remain losslessly gzip-compressed; `python scripts/unpack_evidence.py` verifies their public SHA256 before restoring them. Portable interpreter markers replace private build paths in public receipts. Account screenshots, private authorizations, balances and machine-specific paths are not part of the public projection.
