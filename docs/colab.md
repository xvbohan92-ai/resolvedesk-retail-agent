# Colab CUDA reproduction

The local round-seven experiment continues separately. This optional bundle reproduces
its frozen training and authored test in a new directory; it is not a new held-out
benchmark and cannot resume the current optimizer. Existing checkpoints contain adapter
weights, but no Adam, RNG or sampler state. No upload or cloud job is performed by
preparing the bundle.

**Validation status:** the export, safety checks, report adaptation and notebook code
can be checked on CPU. CUDA installation, model loading, numerical equivalence, memory
fit, optimizer updates and GPU inference have **not** been executed here. There is no
measured GPU speedup claim.

## Prepare locally

From the ResolveDesk source checkout, choose a new destination outside the checkout:

```bash
.venv/bin/python -m training.colab.bundle --destination /private/tmp/resolvedesk-colab-001
```

This writes `resolvedesk-colab-source.zip`, its SHA receipt, a notebook and an initializer
manifest. Source export uses explicit directories, suffixes and exact report paths.
It excludes `.env`, databases, model weights, user state, and original session/evaluation
reports. The only historical report inputs are the exact synthetic planning/training
receipts required by frozen verification: `reports/v6/updates.json`,
`reports/v6_1/analysis-plan.json`, `reports/v6_1/training.json`, and
`reports/v7/analysis-plan.json`. Frozen V6, V6.1 and V7 verification closures are included.
No current V7 training, validation or evaluation result is exported.

To stage the existing predecessor adapter for a **user-managed** upload, choose another
new destination and add `--stage-initializer`. Its two declared files and SHA manifest
go into a **separate** `resolvedesk-initializer.zip`; weights never enter the source ZIP.
This copies local files only. Inspect the archive inventory and receipt before uploading.
The adapter must equal frozen SHA
`041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08`.

## Open the notebook

Open `resolvedesk_cuda.ipynb` in Colab yourself. Choose a Linux **Python 3.12** GPU runtime.
Upload the source ZIP and, when needed, the separate initializer ZIP yourself. Neither
the notebook nor the export command calls Drive, GitHub, an upload API, or a billing API.
The notebook defaults to `MODE = "inspect"`; running its default cells loads no model,
installs no package and starts no training. Copy the source ZIP SHA from the local export
receipt into `EXPECTED_SOURCE_SHA256` before using a compute mode.

The frozen training uses MLX 0.32.4, MLX-LM 0.32.2, Transformers 5.21.0, NumPy 2.5.4 and
safetensors 0.8.0. Linux requires MLX's CUDA12 extra; the Mac-only lock is included as
provenance, not used as a Linux installation lock. The notebook explicitly pins those
core versions and records the actual installed versions. If a matching CUDA wheel cannot
be installed, stop and record the incompatibility; do not silently upgrade the run.

The [official MLX requirements](https://github.com/ml-explore/mlx/blob/main/docs/src/install.rst)
include NVIDIA compute capability >=7.5, driver >=550.54.14, CUDA >=12.0 and glibc >=2.35.
The installed CUDA stack and selected GPU still need an actual one-step preflight.
The base snapshot includes BF16 scales; hardware eligibility alone does not prove this
specific quantized training path works or fits in memory.

## Choose each stage explicitly

Use one stable, new `WORKSPACE` directory, such as `/content/resolvedesk-cuda-repro-001`.
Existing exports are verified rather than merged; changed sources, unknown ZIP files,
symlinks and traversal paths are rejected. The runner refuses an original Git checkout.

1. `setup`: inspect Python, glibc, driver, selected NVIDIA GPU, free VRAM and CUDA toolkit
   **before installing** the explicitly pinned dependencies. An absent or unusable GPU
   stops setup; there is no CPU training fallback or automatic paid-runtime selection.
   Set `PROVISION_BASE = True` to download the public base
   `Qwen/Qwen3-4B-MLX-4bit` at revision
   `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. Download is restricted to the existing
   base-model specification; every expected size and SHA is then verified. Set the
   initializer archive path to import its exact two files. No model inference occurs.
2. `preflight`: the actual frozen one-update training preflight checks full versus tail
   output-projection loss, a finite decreasing sample loss, changed adapter weights and
   target coverage. A separate CPU NumPy/safetensors process then checks exact predecessor
   initialization. Its updated adapter is discarded; it is never used to initialize formal
   training. This stage performs real GPU/model work only after explicitly selecting it.
3. `train`: starts the configured **1,344-update** run from the exact predecessor, not from
   preflight. The unchanged 50/50 sampler, field weights, full input context and checkpoint
   candidates remain in force. A passing preflight initializer audit is required.
4. `select`: runs the unchanged validation-only selection over the two configured
   candidates. No test result is used for selection.
5. `evaluate`: reproduces the existing four-arm, 48-task-per-arm test once, with the same
   controller, unconstrained greedy decoding and 12-call budget. It is a cross-backend
   reproduction, not 192 new independent benchmark tasks. Do not pool it with local scores.
   The wrapper writes a separate scoped-consent audit plan before CUDA test inference,
   then scans the completed raw records; findings remain separate from business scores.

The wrapper does not change model prompts, loss, data, optimizer, seeds, budgets, controller,
selection or metric definitions. It transparently adapts only these backend/report details:

- Metal wired-memory configuration becomes a recorded no-op on CUDA.
- Local Ollama cache waiting becomes a bounded `nvidia-smi` free-memory probe of device 0
  (10-second timeout, no model unloading, fail on insufficient free memory). The default
  free-memory guard is **8,192 MiB (8 GiB)**, a conservative prerequisite rather than a
  guarantee that this model's CUDA training path fits.
- Root training/validation/evaluation reports identify `cloud_compute_used: true`, the
  GPU, backend, actual core versions, wrapper SHA and source-manifest SHA. Hardcoded local
  identity from the frozen trainer is corrected only in its reporting JSON binding.

All weights, reports and stage receipts stay under the new reproduction directory. Every
stage has its own `colab-execution/<stage>.json` receipt; failed stages retain errors and
cannot be overwritten by rerunning. Start another clean reproduction to retry a failed
stage. There is no optimizer resume. The frozen evaluation supports partial continuation,
but this wrapper intentionally does not retry a started stage or combine evidence.

## Check before trusting results

Confirm the CUDA preflight and independent initialization audit pass on the actual GPU,
then inspect all completed updates, both validation candidates and the four-arm raw test
records. Check `colab-execution` metadata and the unchanged input hashes. Differences in
CUDA kernels can change numerical results despite identical inputs; no bitwise equality
with Metal is claimed. The supplemental local scoped-consent audit remains relevant;
primary business success alone does not certify every execution-consent condition.

The notebook format and all Python code cells should be validated before handoff:

```bash
python -m training.colab.build_notebook --validate
.venv/bin/python -m pytest tests/test_colab_bundle.py
```

The first command uses `nbformat` when installed, and always compiles every code cell.
Default inspect-mode notebook execution needs no CUDA or model. Compute-mode end-to-end
execution remains pending an actual Colab runtime. Download results and adapter files
yourself; keep their reproduction identity when copying them back. Moving a CUDA adapter
into local serving is a separate compatibility check, not an automatic replacement.
