# Saved-weight evaluation source and public delivery

This namespace retains implementation source and synthetic fixtures for the local V8 saved-weight experiment. `run.py` records a request before each model call, preserves all failed/corrective calls within a12-call task budget, and rejects reopening an issued run. `resource_guard.py` reads a separately supplied authorization receipt; this repository does not supply an active cloud authorization. Notebook builders containing operator-specific resource observations are omitted.

The original48-case, three-arm experiment stopped at136/144 completed records. Its local raw evidence and weight files are intentionally absent from this public update. The original and partial verification entry points remain strict: without the original inputs, they cannot issue private full-source/token/business verification receipts. Do not replace those missing inputs with a public projection or use the published results to reopen the interrupted run.

`training/v8_eval2_partial` contains the separate retrospective analysis implementation and fixture-based CPU contracts. Its local verification gate also requires the original archive. The common40 public summary does not make that original144-record experiment complete.

For the independently runnable **public** inspection paths, use:

```bash
python scripts/reconcile_partial40_public_results.py
uv run python -m resolve_desk.public_partial40 --port 8774
```

These paths check published numerical CSV outcomes, paired counts, denominators and figure hashes. They do not generate model outputs, load weights, reconstruct private prompts, rescore absent raw business evidence or activate a cloud runtime. The48 authored synthetic inputs are included for inspecting the case definitions and fixed common-prefix selection. Outcome labels in the public CSV are inputs to this consistency check.

[Stage report](../../docs/v8-eval2-partial40.md) · [Publication scope](../../PUBLICATION_STATUS.md) · [Public summary](../../reports/v8-eval2-partial40/public-summary.json)
