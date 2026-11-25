"""Monitor a DGX Spark GPU and request checkpoint-and-exit from selected runs."""

import argparse
import csv
import json
import math
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from thermal_stop import REQUEST, STATUS, publish, read_record

POLL_SECONDS = 10
CORE_FIELDS = ("temperature.gpu", "utilization.gpu", "memory.free")
THERMAL_FIELDS = ("sw_thermal_slowdown", "hw_thermal_slowdown")


def gpu_query(gpu, fields):
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    (row,) = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    return dict(zip(fields, row, strict=True))


def optional_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def host_available_kib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return optional_number(line.split()[1])
    except OSError:
        pass
    return None


class NvidiaTelemetry:
    def __init__(self, gpu):
        self.gpu = gpu
        self.fields = None

    def __call__(self):
        if self.fields is None:
            for prefix in ("clocks_event_reasons.", "clocks_throttle_reasons."):
                fields = CORE_FIELDS + tuple(prefix + name for name in THERMAL_FIELDS)
                try:
                    readings = gpu_query(self.gpu, fields)
                except subprocess.CalledProcessError:
                    continue
                self.fields = fields
                break
            else:
                readings = gpu_query(self.gpu, CORE_FIELDS)
                self.fields = CORE_FIELDS
        else:
            readings = gpu_query(self.gpu, self.fields)
        sample = dict(zip(
            ("temperature_c", "gpu_utilization_pct", "gpu_memory_free_mib"),
            (optional_number(readings[field]) for field in CORE_FIELDS),
            strict=True,
        ))
        sample["host_memory_available_kib"] = host_available_kib()
        for name in THERMAL_FIELDS:
            sample[name] = next(
                ({"Active": True, "Not Active": False}.get(value)
                 for field, value in readings.items() if field.endswith("." + name)),
                None,
            )
        return sample


def monitor(args, read_sample, log, clock=time.monotonic, sleep=time.sleep):
    pending = {}
    warning_since = stop_since = deadline = None
    next_poll = clock()
    while True:
        try:
            sample = read_sample()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            sample = {"temperature_c": None, "error": str(error)}
        log("sample", **sample)
        now = clock()
        temperature = sample["temperature_c"]

        if deadline is None:
            if all((run / "complete.json").exists() for run in args.run):
                log("runs_complete")
                return 0
            if temperature is None or temperature < args.warn_c:
                warning_since = None
            elif warning_since is None:
                warning_since = now
            elif now - warning_since >= args.sustain_seconds and math.isfinite(warning_since):
                log("warning", temperature_c=temperature, threshold_c=args.warn_c)
                warning_since = -math.inf

            if temperature is None or temperature < args.stop_c:
                stop_since = None
            elif stop_since is None:
                stop_since = now
            elif now - stop_since >= args.sustain_seconds:
                reason = {
                    "type": "sustained_gpu_temperature",
                    "gpu": args.gpu,
                    "temperature_c": temperature,
                    "threshold_c": args.stop_c,
                    "observed_seconds": now - stop_since,
                }
                failed = False
                for run in args.run:
                    if (run / "complete.json").exists():
                        log("already_complete", run=str(run))
                        continue
                    request = dict(request_id=uuid.uuid4().hex, reason=reason)
                    try:
                        try:
                            publish(run / REQUEST, request)
                        except FileExistsError:
                            request = read_record(run / REQUEST)
                    except OSError as error:
                        log("request_failed", run=str(run), reason=reason, error=str(error))
                        failed = True
                        continue
                    pending[run] = request["request_id"]
                    log("stop_requested", run=str(run), **request)
                deadline = now + args.checkpoint_timeout

        if deadline is not None:
            for run, request_id in list(pending.items()):
                status = read_record(run / STATUS)
                if status and status["request_id"] == request_id:
                    log("final_checkpoint", run=str(run), **status)
                    failed |= status["checkpoint_status"] != "saved"
                elif (run / "complete.json").exists():
                    log("completed_during_shutdown", run=str(run), request_id=request_id)
                elif now >= deadline:
                    log("final_checkpoint", run=str(run), request_id=request_id, checkpoint_status="unconfirmed")
                    failed = True
                else:
                    continue
                del pending[run]
            if not pending:
                return int(failed)

        next_poll = max(next_poll + POLL_SECONDS, clock())
        sleep(max(0, next_poll - clock()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="Existing training run directory")
    parser.add_argument("--log", type=Path, required=True, help="New JSONL log path")
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument("--warn-c", required=True, type=float)
    parser.add_argument("--stop-c", required=True, type=float)
    parser.add_argument("--sustain-seconds", type=int, default=30)
    parser.add_argument("--checkpoint-timeout", type=int, default=300)
    args = parser.parse_args()
    if not 0 < args.warn_c < args.stop_c < math.inf:
        parser.error("require 0 < warn-c < stop-c")
    if args.sustain_seconds < POLL_SECONDS or args.checkpoint_timeout <= 0:
        parser.error("require sustain-seconds >= 10 and checkpoint-timeout > 0")
    args.run = list(dict.fromkeys(run.resolve() for run in args.run))
    if any(not run.is_dir() for run in args.run):
        parser.error("all --run directories must already exist")
    with args.log.open("x", buffering=1) as stream:
        def log(event, **data):
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=event, **data)) + "\n")

        log("started", **{**vars(args), "run": [str(run) for run in args.run], "log": str(args.log)})
        return monitor(args, NvidiaTelemetry(args.gpu), log)


if __name__ == "__main__":
    raise SystemExit(main())
