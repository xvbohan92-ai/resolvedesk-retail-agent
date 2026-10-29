"""CPU integration checks actual112-tensor serialization without importing a GPU runtime."""

import importlib.util

import pytest

from training.common import ROOT

PRIOR = ROOT / "reports/cuda-torch/probe-003/train-1344/adapter/adapter_model.safetensors"
pytestmark = pytest.mark.skipif(
    not PRIOR.is_file() or importlib.util.find_spec("safetensors") is None,
    reason="Optional CPU weight audit uses the locally saved predecessor and safetensors",
)


@pytest.fixture
def weights(tmp_path):
    from safetensors.numpy import load_file, save_file

    source = load_file(str(PRIOR))
    initial = {
        k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        ): v.copy()
        for k, v in source.items()
    }
    after = {k: v.copy() for k, v in source.items()}
    after[next(iter(after))].flat[0] += 0.001
    start, end = tmp_path / "initial.safetensors", tmp_path / "after.safetensors"
    save_file(initial, str(start))
    save_file(after, str(end))
    return start, end, initial, after


def test_cpu_proof_checks_real_initialization_and_final_delta(weights):
    from training.v8.verify_training import tensor_audit

    initial, final, _, _ = weights
    proof = tensor_audit(PRIOR, initial, final)
    assert proof["parameters_checked"] == 3_670_016
    assert proof["tensors_checked"] == 112
    assert proof["exact_prior_initialization"]
    assert 0 < proof["adapter_l2_change"] < 0.002


def test_wrong_initializer_is_rejected(weights):
    from safetensors.numpy import save_file

    from training.v8.verify_training import tensor_audit

    initial, final, start, _ = weights
    start[next(iter(start))].flat[0] += 0.001
    save_file(start, str(initial))
    with pytest.raises(ValueError, match="initialization"):
        tensor_audit(PRIOR, initial, final)


def test_nonfinite_final_weights_are_rejected(weights):
    from safetensors.numpy import save_file

    from training.v8.verify_training import tensor_audit

    initial, final, _, end = weights
    end[next(iter(end))].flat[0] = float("nan")
    save_file(end, str(final))
    with pytest.raises(ValueError, match="Nonfinite"):
        tensor_audit(PRIOR, initial, final)
