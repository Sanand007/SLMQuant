"""Checkpointed first v3 stratum, using the unchanged v2 search operators."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


V3 = Path(__file__).resolve().parent
BASE = V3.parent
STUDY = V3 / "studies/mlx_affine_qwen3_1_7b"
os.environ.update(
    HF_HOME=str(BASE / "cache/huggingface"),
    HF_HUB_OFFLINE="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    TOKENIZERS_PARALLELISM="false",
    MPLCONFIGDIR=str(V3 / "cache/matplotlib"),
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    inputs = read(BASE / "v2/inputs.json")
    manifest = read(V3 / "data_manifest.json")
    assert manifest["status"] == "frozen_inputs"
    folder = STUDY / "data"
    folder.mkdir(parents=True, exist_ok=True)
    source = V3 / "data/qwen3_1_7b"
    for name in ("calibration", "train", "validation"):
        target = folder / f"{name}_tokens.npy"
        if not target.exists():
            shutil.copyfile(source / f"{name}_tokens.npy", target)
    target = folder / "test_tokens.npy"
    if not target.exists():
        import numpy as np

        news = np.load(source / "news_test_tokens.npy")
        imdb = np.load(source / "imdb_test_tokens.npy")
        combined = np.stack([row for pair in zip(news, imdb) for row in pair])
        np.save(target, combined)
    arc = folder / "arc_easy.json"
    if not arc.exists():
        shutil.copyfile(V3 / "data/arc_challenge.json", arc)
    model_inputs = {
        "model_id": inputs["buddy_id"],
        "model_path": inputs["buddy_model_path"],
        "model_revision": inputs["buddy_revision"],
        "v3_data_manifest": str(V3 / "data_manifest.json"),
    }
    if not (STUDY / "inputs.json").exists():
        write(STUDY / "inputs.json", model_inputs)


def source_hashes():
    files = [
        Path(__file__),
        STUDY / "protocol.json",
        V3 / "protocol_frozen.md",
        V3 / "data_manifest.json",
        BASE / "experiment.py",
        BASE / "v2/runner.py",
        BASE / "v2/operators.py",
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def limits(first_started):
    if time.time() > first_started + 12 * 3600:
        raise TimeoutError("V3 MLX-affine stratum reached its original 12-hour deadline")
    size_kib = int(subprocess.check_output(["du", "-sk", str(V3)], text=True).split()[0])
    if size_kib * 1024 > 30 * 10**9:
        raise RuntimeError("Combined v3 directory exceeded its 30-GB cap")


def load_runner(first_started):
    sys.path.insert(0, str(BASE / "v2"))
    import runner as legacy

    legacy.ROOT = STUDY
    legacy.v1.ROOT = STUDY
    legacy.v1.ident = legacy.ident

    class BoundedRunner(legacy.Runner):
        last_disk_check = 0

        def guard(self):
            super().guard()
            if time.monotonic() - self.last_disk_check > 60:
                limits(first_started)
                self.last_disk_check = time.monotonic()

    return BoundedRunner(), legacy


def selected_choices(rows, runner):
    by_method = {}
    for method in ("random", "tournament", "alternis"):
        subset = [row for row in rows if row["method"] == method]
        best = min(subset, key=lambda row: (row["validation_nll"], row["id"]))
        by_method[method] = best
    conventional = runner.evaluate("uniform_4", "validation")
    by_method["conventional"] = {
        "label": "uniform_4", "method": "conventional", "genome": list(runner.base),
        "id": "uniform_4", "validation_nll": conventional["nll"]
    }
    deployable = min(by_method.values(), key=lambda row: (row["validation_nll"], row["id"]))
    return {"by_method": by_method, "deployable": deployable,
            "decision_source": "validation only; frozen before v3 text or ARC-Challenge tests"}


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("PASS — MLX-affine supervisor already running")
            return
        run_path = STUDY / "run.json"
        run = read(run_path, {"started": time.time(), "status": "initializing"})
        if not run_path.exists():
            write(run_path, run)
        started = run["started"]
        completed_path = STUDY / "completed_stages.json"
        completed = read(completed_path, [])

        def stage(name, action):
            if name in completed:
                return
            limits(started)
            write(STUDY / "supervisor.json", {"status": "running", "stage": name,
                                               "pid": os.getpid(), "first_started": started,
                                               "updated": time.time()})
            action()
            completed.append(name)
            write(completed_path, completed)

        try:
            stage("prepare", prepare)
            hashes_path = STUDY / "source_hashes.json"
            hashes = source_hashes()
            if hashes_path.exists() and read(hashes_path) != hashes:
                raise RuntimeError("Source/config hash changed after this stratum started; comparability audit required")
            if not hashes_path.exists():
                write(hashes_path, hashes)
            runner, legacy = load_runner(started)
            stage("sensitivity", runner.sensitivity)
            stage("searches", runner.searches)
            stage("validation", runner.freeze)

            def freeze():
                if not (STUDY / "choices_frozen.json").exists():
                    rows = read(STUDY / "finalists.json")
                    write(STUDY / "choices_frozen.json", selected_choices(rows, runner))

            stage("choices_frozen", freeze)

            def text_tests():
                import numpy as np

                folder = STUDY / "data"
                runner.data["news_fresh"] = np.load(folder / "news_test_tokens.npy")
                runner.data["imdb_fresh"] = np.load(folder / "imdb_test_tokens.npy")
                finalists = read(STUDY / "finalists.json")
                rows = finalists + [read(STUDY / "choices_frozen.json")["by_method"]["conventional"]]
                path = STUDY / "heldout_text_fresh.json"
                recorded = read(path, {})
                for row in rows:
                    for split in ("news_fresh", "imdb_fresh"):
                        key = f"{row['label']}:{split}"
                        if key not in recorded:
                            result = runner.evaluate(tuple(row["genome"]), split)
                            recorded[key] = {"label": row["label"], "method": row["method"],
                                             "seed": row.get("seed"), "split": split,
                                             "genome_id": row["id"], "nll": result["nll"],
                                             "tokens": result["tokens"],
                                             "parameter_bytes": result["parameter_bytes"]}
                            write(path, recorded)

            stage("heldout_text_fresh", text_tests)

            def task_tests():
                choices = read(STUDY / "choices_frozen.json")["by_method"]
                path = STUDY / "heldout_task.json"
                recorded = read(path, {})
                for method, row in choices.items():
                    if method not in recorded:
                        recorded[method] = runner.task(tuple(row["genome"]))
                        write(path, recorded)

            stage("heldout_task", task_tests)

            def export():
                choice = read(STUDY / "choices_frozen.json")["deployable"]
                target = STUDY / "models/selected"
                if not (target / "reload_check.json").exists():
                    genome = tuple(choice["genome"])
                    runner.export(genome, target)
                    runner.check_export(target, genome)

            stage("export", export)

            def benchmarks():
                choices = read(STUDY / "choices_frozen.json")["by_method"]
                for method, row in choices.items():
                    prior = read(STUDY / "benchmarks.json", [])
                    if not any(item["label"] == method for item in prior):
                        legacy.v1.benchmark(method, read(STUDY / "inputs.json")["model_path"], row["genome"])

            stage("benchmarks", benchmarks)
            limits(started)
            write(STUDY / "supervisor.json", {"status": "complete", "stage": "complete",
                                               "first_started": started, "completed": time.time()})
            print("PASS — v3 MLX-affine stratum complete")
        except Exception as exc:
            write(STUDY / "supervisor.json", {"status": "failed", "stage": completed[-1] if completed else "setup",
                                               "first_started": started, "updated": time.time(),
                                               "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
