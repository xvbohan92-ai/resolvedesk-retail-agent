"""A completed prior allowance, expired clock or missing new approval cannot start V8."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from training.common import sha
from training.v8 import resource_guard as guard


@pytest.fixture
def approved(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    now = datetime(2026, 10, 30, tzinfo=UTC)
    root = tmp_path / "reports/v8"
    root.mkdir(parents=True)
    plan = root / "pre-inference-plan.json"
    plan.write_text('{"fixture":true}\n')
    value = {
        "round": "v8",
        "explicit_human_approval": True,
        "cap_compute_units": 10,
        "approval_text": "CPU test fixture only, not real approval",
        "purchases_allowed": False,
        "renewals_allowed": False,
        "gpu_type": "T4",
        "google_drive_mount_allowed": False,
        "permitted_work": ["discarded_smoke4", "fixed_final1344", "validation48", "three_arm_test144"],
        "analysis_plan_sha256": sha(plan),
        "allocated_at_utc": now.isoformat(),
        "process_deadline_utc": (now + timedelta(hours=5)).isoformat(),
        "runtime_release_deadline_utc": (now + timedelta(hours=5, minutes=5)).isoformat(),
        "starting_balance_units": 78,
        "reserve_units": 2,
        "observed_units_per_hour": 1.07,
        "operator_stop_balance_units": 70,
    }
    path = root / "test-only-resource.json"
    path.write_text(json.dumps(value))
    return path, value, now


def test_missing_approval_cannot_allocate(tmp_path):
    with pytest.raises(PermissionError):
        guard.verify_resource_receipt(tmp_path / "absent.json")


def test_new_explicit_scope_and_clock_are_required(approved):
    path, value, now = approved
    assert guard.verify_resource_receipt(path, now=now) == value
    with pytest.raises(PermissionError):
        guard.verify_resource_receipt(path, now=now + timedelta(hours=6))
    assert guard.verify_resource_receipt(path, active=False) == value


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("round", "resolvedesk-cuda-eval-002"),
        ("explicit_human_approval", False),
        ("cap_compute_units", 11),
        ("cap_compute_units", True),
        ("cap_compute_units", float("nan")),
        ("purchases_allowed", True),
        ("renewals_allowed", True),
        ("gpu_type", "A100"),
        ("google_drive_mount_allowed", True),
        ("approval_text", ""),
        ("permitted_work", ["fixed_final1344"]),
        ("analysis_plan_sha256", "0" * 64),
        ("reserve_units", 0),
        ("operator_stop_balance_units", 68),
        ("observed_units_per_hour", 3),
        ("starting_balance_units", 9),
        ("allocated_at_utc", "2026-10-30T00:00:00"),
        ("runtime_release_deadline_utc", "2026-10-30T07:00:00+00:00"),
    ],
)
def test_rejects_stale_or_over_budget_approval(approved, field, bad):
    path, value, now = approved
    value[field] = bad
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, PermissionError)):
        guard.verify_resource_receipt(path, now=now)


def test_receipt_cannot_be_redirected_through_symlink(approved, tmp_path):
    path, _, now = approved
    alias = tmp_path / "redirect.json"
    alias.symlink_to(path)
    with pytest.raises(PermissionError):
        guard.verify_resource_receipt(alias, now=now)


def test_deadline_stops_even_if_audit_disk_write_fails(tmp_path, monkeypatch):
    callbacks, stopped = [], []

    class FakeTimer:
        def __init__(self, delay, callback):
            callbacks.append(callback)

        def start(self):
            pass

    monkeypatch.setattr(guard.threading, "Timer", FakeTimer)
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    now = datetime.now(UTC)
    receipt = {"process_deadline_utc": (now + timedelta(minutes=1)).isoformat()}
    guard.arm_process_deadline(receipt, "fixture", stop=stopped.append)

    def disk_failure(*args, **kwargs):
        raise OSError("test fixture: disk full")

    monkeypatch.setattr(type(tmp_path), "mkdir", disk_failure)
    with pytest.raises(OSError, match="disk full"):
        callbacks[0]()
    assert stopped == [124]
