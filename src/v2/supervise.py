"""Unattended expanded study with bounded runtime and resumable stages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PYTHON = sys.executable
SCRIPT = str(ROOT / "runner.py")

def read(name, default=None):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else default

def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)

def main():
    previous = read("supervisor.json", {})
    if previous.get("status") == "running":
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Expanded supervisor already running")
    state = {"status": "running", "pid": os.getpid(), "started": time.time()}
    write("supervisor.json", state)
    run = read("run.json", {"started": time.time()})
    deadline = run["started"] + 12 * 3600
    stages = read("completed_stages.json", [])

    def step(command):
        key = " ".join(command)
        if key in stages:
            return
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Expanded study reached configured wall-clock limit")
        disk = int(subprocess.check_output(["du", "-sk", str(BASE)], text=True).split()[0]) * 1024
        if disk > read("protocol.json")["max_disk_gb"] * 10 ** 9:
            raise RuntimeError("Combined experiment directory exceeded the configured disk limit")
        state.update(stage=key, updated=time.time())
        write("supervisor.json", state)
        filename = "_".join(s.lstrip("-") for s in command) + ".log"
        with (ROOT / "logs" / filename).open("a") as log:
            result = subprocess.run([PYTHON, SCRIPT, *command], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        if result.returncode:
            raise RuntimeError(f"Stage {key} failed; see logs/{filename}")
        stages.append(key)
        write("completed_stages.json", stages)

    try:
        if not (ROOT / "pilot_verified.json").exists():
            step(["pilot"])
        step(["quantize", "--label", "gptq_4", "--pilot"])
        step(["quantize", "--label", "awq_4", "--control", "--pilot"])
        step(["sensitivity"])
        step(["quantize", "--label", "gptq_4"])
        step(["quantize", "--label", "gptq_4", "--control"])
        step(["quantize", "--label", "awq_4", "--control"])
        # Protocol budget is now fixed; record implementation before any search begins.
        if not (ROOT / "source_hashes.json").exists():
            write("source_hashes.json", {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ["runner.py", "operators.py", "gptq_mixed.py", "prepare.py", "supervise.py", "protocol.json"]})
        step(["search"])
        step(["freeze"])
        step(["quantize", "--label", "combined_gptq"])
        baselines = ["higher_precision", "uniform_3", "uniform_4", "uniform_6", "uniform_8",
                     "loss_sensitivity", "gptq_4", "combined_gptq"]
        controls = ["higher_precision", "uniform_4", "gptq_4", "awq_4"]
        for label in baselines:
            step(["evaluate", "--label", label, "--validation-only"])
        for label in controls:
            step(["evaluate", "--label", label, "--control", "--validation-only"])
        step(["freeze_choices"])
        for label in baselines + [r["label"] for r in read("finalists.json")]:
            step(["evaluate", "--label", label])
        for label in controls:
            step(["evaluate", "--label", label, "--control"])
        step(["export"])
        protocol = read("protocol.json")
        for method in protocol["methods"]:
            for seed in protocol["seeds"]:
                search = read(f"search_{method}_{seed}.json")
                if not search["complete"] or len({r["id"] for r in search["history"]}) != protocol["evaluations_per_method_seed"]:
                    raise RuntimeError(f"Incomplete search: {method} {seed}")
        outcomes, control = read("outcomes.json"), read("awq_control/outcomes.json")
        if len(outcomes) != 38 or len(control) != 4:
            raise RuntimeError("Incomplete held-out comparisons")
        budget = read("space.json")["baseline_parameter_bytes"]
        if any(r["parameter_bytes"] > budget for r in outcomes if r.get("seed") is not None or r["label"] in ["loss_sensitivity", "gptq_4", "combined_gptq"]):
            raise RuntimeError("A matched-budget candidate exceeded the baseline")
        run = read("run.json")
        run.update(status="complete", completed=time.time())
        run.pop("error", None)
        write("run.json", run)
        step(["report"])
        state.update(status="complete", completed=time.time())
        write("supervisor.json", state)
        print("PASS — expanded searches, quantizers, held-out tests, speed and export verified.", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated=time.time())
        write("supervisor.json", state)
        run = read("run.json", {})
        run.update(status="failed", error=state["error"], updated=time.time())
        write("run.json", run)
        subprocess.run([PYTHON, SCRIPT, "report"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        print(state["error"], file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
