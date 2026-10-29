# Separate CUDA FP16 LoRA experiment

This optional performance experiment uses PyTorch/PEFT on an explicitly selected free
Colab GPU. It reuses the frozen V7 training data and exact V6.1 initializer. The original
MLX run, its reports, its selected weights and its evaluation remain separate.

**The separate formal run completed all 1,344 true optimizer updates on a free Tesla T4.**
It covered all 824 physical SFT pairs, and an independent CPU tensor audit verified the
exact 112-tensor initialization and changed final adapter. The new 48-case before/after
evaluation was interrupted by termination of the accessible Colab runtime. Its last
locally verified snapshot has 84/96 arm/case records, not a complete comparison. No new
quantified before/after effect, six-group final chart set or acceleration result is
available to publish. Cloud results remain separate from the original local V7 evaluation
and its failed business gate.

## Actual smoke evidence

The third isolated probe passed **36 CPU contract tests** in the actual Colab environment
([JUnit](../reports/cuda-torch/probe-003/cpu-tests.xml)). Three real source tensor slices,
covering **29,696 values**, matched MLX logical BF16 decoding exactly after the two-stage
rounding correction ([slice audit](../reports/cuda-torch/probe-003/decoder-check.json)).
These three slices do not establish numerical equivalence of the entire base.

| Actual four-update smoke | Observed result |
| --- | --- |
| Runtime | Tesla T4, Python 3.12.13, CUDA 12.8 |
| True Adam updates | 4/4, each with finite gradients and positive adapter change |
| Synchronized run wall time | 19.692511 seconds |
| Individual update times | 4.512597, 4.944884, 3.402891, 6.646229 seconds |
| Steps 2–4 median | 4.944884 seconds, only three observations |
| Peak CUDA allocated | 10.76595 GB / 10.02657 GiB |
| Peak CUDA reserved | 12.89329 GB / 12.00781 GiB |
| Independently recomputed adapter L2 change | 0.028329403722431663 |
| Training rows actually sampled | 4/824 physical rows, from four source interactions |

The [actual report](../reports/cuda-torch/probe-003/smoke-4/report.json),
[pre-load plan](../reports/cuda-torch/probe-003/smoke-4/plan.json) and
[independent CPU verification](../reports/cuda-torch/probe-003/smoke-4/verification.json)
retain the source/input hashes, samples, timing scope and allocation counters. The CPU
verifier reread all **112 tensors / 3,670,016 parameters**, confirmed exact predecessor
values after transpose in the saved pre-optimizer snapshot, checked finite initial/final
values and recomputed the final L2 change. It loaded no model or GPU runtime.

The final smoke adapter SHA256 is:

```text
c6fbb980f285c82e257191c64fa2beb8992e47a6bfa61e40d2bd10c5ab6999ae
```

GB denotes decimal bytes; GiB denotes powers of 1,024. Allocated and reserved counters
are different from total GPU/OS memory. The 19.692511-second run includes report,
checkpoint/export and audit overhead, while model loading and tokenization are excluded.
For context, the original local Mac run's first four updates took about **8.1 seconds**
([local update log](../reports/v7/updates.json)). This smoke does **not** show acceleration.
The precision, projection, kernels and measurement overhead differ, so these totals do
not isolate a hardware effect. Four updates do not establish full-run stability,
throughput or model quality.

Earlier failures remain available:

- [Probe 001 decoder failure](../reports/cuda-torch/probe-001/decoder-first-failure.json):
  the real source-slice check rejected single-round BF16 decoding before training;
  **zero optimizer updates**. Its [diagnostic](../reports/cuda-torch/probe-001/decoder-diagnostic.json)
  was retained and the implementation was corrected before a new source export.
- [Probe 002 failed smoke](../reports/cuda-torch/probe-002/smoke-4/report.json):
  preinstalled TorchAO 0.10.0 was incompatible with PEFT's optional import;
  **zero optimizer updates**. The report and
  [runtime cleanup log](../reports/cuda-torch/probe-002/environment-fix.log) remain intact.
  Probe 003 used a new source plan and process after uninstalling the unused dependency.

## Completed formal training; matched evaluation interrupted

The separate **1,344-update** formal run was actually launched on the free T4 at
**2026-10-28 20:00:14 UTC**
([launch receipt](../reports/cuda-torch/probe-003/formal-launch.json)) and completed at
**2026-10-28 21:52:40 UTC**. It started again from the exact predecessor and a fresh
optimizer; smoke weights were not reused.

| Actual formal training | Observed result |
| --- | --- |
| True Adam updates | 1,344/1,344, all recorded gradients finite |
| Unique physical rows sampled | 824/824, from 96 source interactions |
| Balanced bucket updates | 672 record/repair + 672 execution/reply |
| Synchronized run wall time | 6,589.486070 seconds / 109.82 minutes |
| Steps 2–1,344 median update time | 4.287824 seconds |
| Peak CUDA allocated | 12.45404 GB / 11.59873 GiB |
| Peak CUDA reserved | 15.11627 GB / 14.07813 GiB |
| Independently recomputed final adapter L2 change | 0.6083016451749128 |

The [completed report](../reports/cuda-torch/probe-003/train-1344/report.json),
[pre-load plan](../reports/cuda-torch/probe-003/train-1344/plan.json) and
[independent CPU verification](../reports/cuda-torch/probe-003/train-1344/verification.json)
bind every update/sample, source hash and memory/timing scope. Numeric reads confirmed
all **112 tensors / 3,670,016 parameters** in the saved starting adapter exactly match the
predecessor after transpose, all final values are finite, and the reported final L2
change agrees with an independent calculation. No model or GPU was loaded by that audit.

The final 1,344-update adapter SHA256 is:

```text
33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053
```

Checkpoints at 672 and 1,344 updates are retained. The new comparison predeclares the
**final 1,344 checkpoint**, with no selection using its test outputs. Its new 48 authored
cases have a [frozen pre-inference plan](../reports/cuda-eval/analysis-plan.json); the two
matched arms compare the predecessor and final adapter under the same FP16 backend and
V7 workflow. The Colab runtime subsequently disconnected and its session manager reported
**no active sessions**. The original runtime is no longer accessible, and the current
matched evaluation **cannot continue from the recovered evidence**.

The **last locally verified backup snapshot contains 84/96 completed arm/case records**:
48 predecessor records and 36 final-adapter records. The snapshot was observed at
**2026-10-28 23:34:30 UTC** and retains a pending case and unfinished/in-flight call.
That timestamp is **not** the termination time; 84 is **not** a known final actual
completion count. The exact termination time and any progress after the last verified
backup are unknown. No complete 96-record evaluation was recovered.

All backed-up raw outputs, partial reports, call journals, logs and pending entries must
remain intact. **Ambiguous model calls must not be automatically rerun**, and incomplete
records must not be dropped or replaced to obtain a successful result. The separate
atomic report/journal snapshots cannot establish that an unfinished call did not run.
The [interrupted evaluation status](../reports/cuda-eval/README.md) and
[backup inventory](../reports/cuda-eval/continuation.json) retain the provenance.

There is no complete paired accuracy result, quantified before/after effect or set of
six final comparison figure groups to publish. The 48-before/36-after snapshot cannot
substitute for the declared complete paired cohort. Formal training completion verifies
this run's execution and measured resource use, not generalization, production
reliability or acceleration. Original V7 results, the failed business gate and the known
operation-rewrite consent limitation remain unchanged.

## Model and objective

The loader reads only the pinned local snapshot `Qwen/Qwen3-4B-MLX-4bit`, revision
`52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. It streams affine 4-bit groups into their
logical BF16 decoded values, rounding the multiplication to BF16 before rounding the
addition to BF16, then stores the base in FP16. The two primitive rounding steps match
the source MLX expression; one fused FP32 multiply/add followed by a single BF16 cast
can produce different values. It records casting
mismatches, underflows, nonfinite values and maximum error. This is **FP16 LoRA on a
decoded-4-bit-derived base**. It uses no bitsandbytes/NF4 quantized training, and it makes
no bitwise equivalence claim with MLX computation.

Only the last eight layers' seven projection types receive rank-8 LoRA. All **112
tensors / 3,670,016 parameters** start at the exact selected V6.1 values, after the
required transpose into PEFT layout. FP32 adapter masters preserve their numerical
values. MLX's direct scale 16 becomes PEFT alpha 128 divided by rank 8. The base stays
frozen and the input/output embeddings share one allocation. The loader requires at
least **12 GiB free VRAM** before allocating the approximately 8 GB FP16 base. This run's
smoke and formal reports demonstrate their observed memory fit; that prerequisite alone
does not guarantee fit on another runtime.

The frozen **824 physically unique SFT pairs** and Python seed `202610287` are unchanged.
The alternating sampler uses 50% initial-record/record-repair targets and 50% execution
targets, with the same shuffled cycles. The 1,344-step schedule covers all 824 rows;
its 672-step prefix covers 494. Sampling repetitions are optimization updates, not new
examples.

Existing `encode_row` performs exact tokenizer-offset weighting: prompts and padding 0,
ordinary completion 1, permission values 8 and reason values 4. The backbone sees the
entire input context; only the vocabulary projection uses the final **192 positions**
instead of MLX's 512. An offline CPU audit of all 824 rows found a longest sequence of
2,620 tokens, longest completion 115 and maximum supervised tail including actual padding
141. No input or target is dropped. Every actual run repeats this audit and fails if a
target would be lost. A shorter zero-weight projection preserves the defined objective,
while backend, precision, attention, checkpointing and execution costs differ.

Causal SDPA uses FP16 autocast, nonreentrant gradient checkpointing and `use_cache=False`.
Embedding outputs are not forced to require gradients, avoiding unnecessary backward
computation through the frozen prefix. Adam uses learning rate `6e-6`, betas `(0.9,0.999)`,
epsilon `1e-8`, no weight decay and no gradient clipping. GradScaler starts at 128.
Every update requires all 112 gradients to exist and be finite, a nonzero gradient norm,
a real Adam step verified by its optimizer hook, finite adapter weights and positive
adapter change. An overflow or skipped update fails visibly and cannot be counted as a
completed update.

## Source and runtime

The source ZIP includes `training/cuda_torch/{model,train}.py`, both corresponding CPU
test files and this guide. Weights, credentials, databases, customer sessions and current
local evaluation reports remain outside the source ZIP. The separate predecessor
initializer is unchanged:

```text
041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08
```

Use a **new source workspace**. In the existing notebook, update the ZIP SHA and
`WORKSPACE`, then run its source verification/extraction cell alone. It validates every
archive entry and refuses to merge different sources. The notebook's remaining compute
cells invoke the separate MLX reproduction; launch the PyTorch commands below directly
from the new workspace.

The already observed core Colab environment needs no package installation:

| Package | Required version |
| --- | --- |
| Python | 3.12, with the actual patch recorded |
| torch | 2.11.0+cu128 |
| peft | 0.19.1 |
| accelerate | 1.14.0 |
| transformers | 5.21.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.4 |
| torchao | Absent as both an installed distribution and an importable module |

The trainer records and enforces these package versions. It uses Linux, one actual CUDA
GPU with compute capability at least 7.5, local tokenizer/model files and no model download.
No paid-runtime selection or billing action is implemented.

An unused preinstalled `torchao` can trigger an incompatible optional PEFT import even
though this experiment does not use it. In the **isolated temporary Colab runtime**, remove
that package before the next new smoke run:

```bash
python -m pip uninstall -y torchao
```

Use a fresh CLI process after removal. The trainer's stdlib-only precheck runs before any
Torch/model import or allocation. It requires both `find_spec("torchao")` and its installed
distribution metadata to be absent; import masking or a stub in `sys.modules` is rejected.
The plan/report retain Python/sys identity, required package versions and relevant optional
dependency versions, including TorchAO, bitsandbytes, Triton and alternative attention stacks.
Failed plans/reports remain intact; cleanup does not authorize resuming their output directory.

Reuse the already verified base and initializer by copying their existing directories
into the same relative paths below the new workspace. For example, after adapting the
two workspace names to the actual notebook:

```python
from pathlib import Path
import shutil

old = Path("/content/resolvedesk-cuda-repro-001")
new = Path("/content/resolvedesk-torch-repro-001")
for relative in (
    "data/models/qwen3-4b-mlx",
    "data/finetuning/retail-qlora-v6_1/best",
):
    target = new / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old / relative, target)  # Refuses an existing destination.
```

This copies local runtime files and makes no network request. Both the trainer and loader
verify the exact base file sizes/hashes and predecessor hash again. Keep old workspaces
and reports intact. Source validation rejects symlink redirections.

## Explicit smoke and formal stages

From the new workspace, run the discarded four-update performance smoke:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m training.cuda_torch.train --kind smoke --steps 4
```

The first four samples come from the frozen schedule, not hand-picked easy examples.
Before importing Torch or constructing the model, the trainer writes a plan binding its
source, loader, frozen config/data, base specification, initializer and complete planned
sample sequence. Outputs go to `reports/cuda-torch/smoke-4/`. The report retains the exact
initializer audit, per-step loss, sample index, synchronized update wall time, peak CUDA
allocation/reservation, gradient/scaler checks and adapter change. Saved tensors contain
only adapter parameters, including the pre-optimizer PEFT snapshot; no full base is saved.

Inspect the completed smoke and its memory/time evidence before starting the formal run:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m training.cuda_torch.train --kind train --steps 1344
```

Formal training requires the passed four-update smoke receipt with matching source/input
hashes. It loads the exact predecessor **again**, creates a fresh optimizer/scaler and
uses no smoke weights or optimizer state. Its separate output is
`reports/cuda-torch/train-1344/`, with PEFT checkpoints at 672 and 1,344 true updates.
There is no validation, checkpoint selection or test inference in this CLI.

Existing output directories are never overwritten or resumed. Failed runs retain their
plan, any completed metrics, pending sample/error and any already saved adapter evidence.
To retry an explicitly approved failed run, choose another new directory with `--output`
under `reports/cuda-torch/`; for a custom smoke location, pass its JSON to the formal run
with `--smoke-report`. The CLI also permits one-to-three-step disposable probes, but only
a completed four-update smoke can authorize the formal stage.

Per-update synchronized wall times include gradient/parameter audit overhead. Overall
run time also includes report/checkpoint/export overhead, while model loading and
tokenization are outside the update-loop timer. Peak allocated and reserved CUDA memory
are distinct from total GPU/OS memory. These observations would describe this complete
implementation and precision path, not an isolated hardware speedup.

## Verification and interpretation

```bash
python -m pytest -q tests/test_cuda_torch_training.py tests/test_cuda_torch_model.py
```

Dependency-free CPU tests verify the actual tail-loss call path, full-context/target
alignment, weighting, projection rejection, sampler coverage and smoke/formal identity
guards. Loader tests verify strict key/shape/scale mappings and CPU dequantization;
Torch-specific CPU tests run when Torch is present. None of these fixtures is GPU or
model-quality evidence.

A successful four-update smoke establishes a short functioning update path on that
runtime. It does not establish full-run throughput, 1,344-step stability or model quality.
Any later accuracy comparison needs its own declared evaluation, raw failures and
separate provenance. Do not replace the original V7 results or label reused authored
tests as new held-out data.
