"""Offline watchdog tests using mocked telemetry and checkpoint writers."""

from types import SimpleNamespace

import pytest
from thermal_stop import REQUEST, STATUS, ThermalStop, checkpoint_if_requested, publish, read_record
from thermal_watchdog import NvidiaTelemetry, monitor


@pytest.mark.parametrize("fail_checkpoint", [False, True])
def test_spikes_missing_samples_and_targeted_shutdown(tmp_path, fail_checkpoint):
    targets = [tmp_path / name for name in ("seed1", "seed2", "finished")]
    unrelated = tmp_path / "unrelated"
    for run in [*targets, unrelated]:
        run.mkdir()
    (targets[-1] / "complete.json").touch()
    args = SimpleNamespace(
        run=targets, warn_c=70, stop_c=80, sustain_seconds=20, checkpoint_timeout=40, gpu="0"
    )
    temperatures = iter([83, 65, 83, None, 83, 83, 83, 65])
    elapsed = 0
    events = []

    def log(event, **data):
        events.append((elapsed, event, data))

    def save(path):
        if fail_checkpoint:
            raise OSError("mock disk failure")
        path.write_bytes(b"mock checkpoint")

    trainer = SimpleNamespace(completed_updates=5, total_env_steps=630, save_checkpoint=save)

    def sleep(seconds):
        nonlocal elapsed
        assert seconds == 10
        elapsed += seconds
        if elapsed == 70:
            for run in targets[:2]:
                with pytest.raises(OSError if fail_checkpoint else ThermalStop):
                    checkpoint_if_requested(run, trainer)

    result = monitor(
        args, lambda: {"temperature_c": next(temperatures)}, log, clock=lambda: elapsed, sleep=sleep
    )
    assert result == int(fail_checkpoint)
    requests = [(time, data) for time, event, data in events if event == "stop_requested"]
    assert [(time, data["run"]) for time, data in requests] == [(60, str(run)) for run in targets[:2]]
    assert [time for time, event, _ in events if event == "warning"] == [60]
    for run in targets[:2]:
        request = read_record(run / REQUEST)
        status = read_record(run / STATUS)
        assert status["request_id"] == request["request_id"]
        assert status["reason"]["type"] == "sustained_gpu_temperature"
        assert status["checkpoint_status"] == ("failed" if fail_checkpoint else "saved")
        assert (run / "latest.pt").exists() is not fail_checkpoint
    assert not (targets[-1] / REQUEST).exists()
    assert not (unrelated / REQUEST).exists()


def test_timeout_does_not_count_old_checkpoint(tmp_path):
    (tmp_path / "latest.pt").touch()
    publish(tmp_path / STATUS, {"request_id": "old", "checkpoint_status": "saved"})
    args = SimpleNamespace(run=[tmp_path], warn_c=70, stop_c=80, sustain_seconds=10, checkpoint_timeout=20, gpu="0")
    ticks = iter([0, 10, 20, 30])
    elapsed = next(ticks)
    events = []

    def sleep(seconds):
        nonlocal elapsed
        elapsed = next(ticks)

    assert monitor(
        args, lambda: {"temperature_c": 82}, lambda event, **data: events.append((event, data)),
        clock=lambda: elapsed, sleep=sleep,
    ) == 1
    assert events[-1][1]["checkpoint_status"] == "unconfirmed"


def test_unified_memory_and_thermal_flags(monkeypatch):
    import thermal_watchdog

    def query(gpu, fields):
        assert gpu == "0"
        return dict(zip(fields, ["55", "30", "[N/A]", "Active", "Not Active"], strict=True))

    monkeypatch.setattr(thermal_watchdog, "gpu_query", query)
    monkeypatch.setattr(thermal_watchdog, "host_available_kib", lambda: 123)
    assert NvidiaTelemetry("0")() == {
        "temperature_c": 55,
        "gpu_utilization_pct": 30,
        "gpu_memory_free_mib": None,
        "host_memory_available_kib": 123,
        "sw_thermal_slowdown": True,
        "hw_thermal_slowdown": False,
    }
