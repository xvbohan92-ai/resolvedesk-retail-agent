"""Dependency-free CPU contracts for the CUDA trainer; no Torch/model/GPU execution."""

import itertools
import json
import math
from types import SimpleNamespace

import pytest

from training.cuda_torch import train
from training.v7.common import read_rows
from training.v7.train import balanced_sample_indices


class Tensor:
    """Tiny scalar-array double for testing the trainer's real tensor slicing/objective path."""

    def __init__(self, values, shape):
        self.values, self.shape = list(values), tuple(shape)
        assert math.prod(shape) == len(self.values)

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        indices += (slice(None),) * (len(self.shape) - len(indices))
        axes, shape = [], []
        for size, index in zip(self.shape, indices, strict=True):
            if isinstance(index, slice):
                axis = list(range(size))[index]
                axes.append(axis)
                shape.append(len(axis))
            else:
                axes.append([index % size])
        strides = [math.prod(self.shape[n + 1 :]) for n in range(len(self.shape))]
        values = [
            self.values[sum(i * s for i, s in zip(point, strides, strict=True))]
            for point in itertools.product(*axes)
        ]
        return Tensor(values, shape)

    def float(self):
        return self

    def reshape(self, *shape):
        if -1 in shape:
            missing = len(self.values) // -math.prod(shape)
            shape = tuple(missing if size == -1 else size for size in shape)
        return Tensor(self.values, shape)

    def reshape_as(self, other):
        return self.reshape(*other.shape)

    def __mul__(self, other):
        assert self.shape == other.shape
        return Tensor([a * b for a, b in zip(self.values, other.values, strict=True)], self.shape)

    def sum(self):
        return math.fsum(self.values)


def cross_entropy(logits, targets, reduction):
    assert reduction == "none"
    rows, width = logits.shape
    assert targets.shape == (rows,)
    losses = []
    for row, target in enumerate(targets.values):
        values = logits.values[row * width : (row + 1) * width]
        maximum = max(values)
        losses.append(maximum + math.log(sum(math.exp(v - maximum) for v in values)) - values[target])
    return Tensor(losses, (rows,))


def encoded(size, prompt):
    return {
        "tokens": list(range(size)),
        "weights": [0.0] * (prompt - 1) + [1.0] * (size - prompt),
        "prompt_length": prompt,
        "permission_tokens": 0,
        "reason_tokens": 0,
    }


def test_actual_tail_loss_keeps_full_context_but_projects_only_weight_width():
    seen = {}
    logits = [
        [20, -20, -20],
        [-10, 15, -10],
        [10, 0, -5],
        [100, -100, -100],
        [1, -1, 3],
        [2, 1, -1],
    ]
    flattened = list(itertools.chain.from_iterable(logits))

    def backbone(*, input_ids, use_cache, return_dict):
        assert input_ids.values == [0, 1, 2, 0, 1, 2]
        assert use_cache is False and return_dict is True
        seen["context_positions"] = input_ids.shape[1]
        return SimpleNamespace(last_hidden_state=Tensor(flattened, (1, 6, 3)))

    def head(hidden):
        seen["projected_positions"] = hidden.shape[1]
        return hidden

    causal = SimpleNamespace(model=backbone, get_output_embeddings=lambda: head)
    model = SimpleNamespace(get_base_model=lambda: causal)
    fake_torch = SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(cross_entropy=cross_entropy)))
    tokens, weights = Tensor([0, 1, 2, 0, 1, 2, 0], (1, 7)), Tensor([0, 8, 4], (1, 3))
    got = train.tail_loss(model, tokens, weights, fake_torch)
    full_ce = cross_entropy(Tensor(flattened, (6, 3)), Tensor([1, 2, 0, 1, 2, 0], (6,)), "none")
    expected = (full_ce.values[4] * 8 + full_ce.values[5] * 4) / 12
    assert got == pytest.approx(expected)
    assert seen == {"context_positions": 6, "projected_positions": 3}
    # The very wrong, zero-weight target at position 3 cannot affect the objective.
    assert full_ce.values[3] > 100 and got < 1


def test_projection_retains_every_original_input_and_weighted_target():
    example = encoded(210, 200)
    example["weights"][-2:] = [8.0, 4.0]
    tokens, weights = train.tail_batch(example, 3072)
    assert tokens[:210] == example["tokens"] and len(tokens) == 225
    assert len(weights) == 192 and sum(weights) == sum(example["weights"])
    assert not any(weights[: 192 - 25])
    audit = train.token_audit([example, encoded(33, 30), encoded(3072, 3000)], 3072)
    assert audit["all_targets_covered"] and audit["all_physical_rows_checked"] == 3
    assert audit["maximum_padding_tokens"] == 32
    assert audit["maximum_tail_including_padding"] < 192
    assert audit["longest_sequence"] == 3072


def test_projection_rejects_any_supervised_target_outside_tail():
    with pytest.raises(AssertionError, match="supervised completion token would be dropped"):
        train.tail_batch(encoded(260, 5), 3072)
    with pytest.raises(AssertionError):
        train.tail_batch(encoded(3073, 3000), 3072)


def test_schedule_is_exact_frozen_python_sampler_and_full_run_covers_all_rows():
    rows = read_rows("train")
    schedule = balanced_sample_indices(rows, 202610287, 1344)
    assert [index for _, index in schedule[:4]] == [364, 370, 523, 612]
    assert [bucket for bucket, _ in schedule[:4]] == ["intent_and_record_repair", "execution_and_reply"] * 2
    full = train.schedule_audit(rows, schedule)
    prefix = train.schedule_audit(rows, schedule[:672])
    assert full["unique_rows_sampled"] == full["physical_rows"] == 824 and full["all_physical_rows_used"]
    assert prefix["unique_rows_sampled"] == 494 and not prefix["all_physical_rows_used"]
    assert full["bucket_updates"] == {"intent_and_record_repair": 672, "execution_and_reply": 672}
    assert full["unique_cases_sampled"] == 96


def test_prepare_refuses_overwrite_or_wrong_formal_steps_before_reading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "REPORT_ROOT", tmp_path)
    existing = tmp_path / "old-run"
    existing.mkdir()
    (existing / "preserve.txt").write_text("retain evidence")
    with pytest.raises(FileExistsError):
        train.prepare_plan(existing, "smoke", 4)
    with pytest.raises(ValueError, match="exactly 1344"):
        train.prepare_plan(tmp_path / "wrong-steps", "train", 4)
    assert (existing / "preserve.txt").read_text() == "retain evidence"


def test_formal_requires_four_true_finite_updates_and_same_source_initializer(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    report = {
        "status": "completed",
        "kind": "smoke",
        "completed_true_optimizer_updates": 4,
        "adapter_l2_change": 0.1,
        "initializer_audit": {"exact_tensor_equality": True},
        "all_gradients_finite": True,
        "initializer_sha256": "same-init",
        "source_inputs_sha256": {"source.py": "same-source"},
        "smoke_weights_discarded": True,
        "smoke_weights_reused": False,
    }
    path = tmp_path / "smoke.json"
    plan = {"initializer_sha256": "same-init", "source_inputs_sha256": {"source.py": "same-source"}}
    path.write_text(json.dumps(report))
    assert train.check_smoke(plan, path)["weights_reused"] is False
    for changes in [
        {"completed_true_optimizer_updates": 3},
        {"all_gradients_finite": False},
        {"initializer_sha256": "wrong-init"},
        {"source_inputs_sha256": {"source.py": "changed"}},
        {"smoke_weights_reused": True},
    ]:
        path.write_text(json.dumps({**report, **changes}))
        with pytest.raises(AssertionError):
            train.check_smoke(plan, path)


@pytest.mark.parametrize("module_present,distribution_version", [(True, None), (False, "0.10.0")])
def test_torchao_guard_rejects_module_or_distribution_independently(
    monkeypatch, module_present, distribution_version
):
    def version(name):
        if name == "torchao" and distribution_version:
            return distribution_version
        if name in train.PACKAGE_PINS:
            return train.PACKAGE_PINS[name]
        raise train.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(train.importlib.metadata, "version", version)
    monkeypatch.setattr(
        train.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin="CPU fixture") if module_present else None,
    )
    snapshot = train.dependency_snapshot()
    assert snapshot["torchao"]["module_discoverable"] is module_present
    assert snapshot["torchao"]["distribution_version"] == distribution_version
    with pytest.raises(RuntimeError, match="must be uninstalled"):
        train.dependency_precheck(snapshot)


def test_torchao_guard_requires_real_absence_and_rejects_module_masking(monkeypatch):
    def version(name):
        if name in train.PACKAGE_PINS:
            return train.PACKAGE_PINS[name]
        raise train.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(train.importlib.metadata, "version", version)
    monkeypatch.setattr(train.importlib.util, "find_spec", lambda name: None)
    snapshot = train.dependency_snapshot()
    assert train.dependency_precheck(snapshot) == snapshot
    assert snapshot["torchao"]["distribution_version"] is None
    monkeypatch.setitem(train.sys.modules, "torchao", None)
    with pytest.raises(RuntimeError, match="must be uninstalled"):
        train.dependency_precheck()
