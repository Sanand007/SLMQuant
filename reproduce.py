#!/usr/bin/env python3
"""Verify archived evidence or launch an isolated, checkpointed reproduction."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDY = "studies/wide30_mlx_affine_qwen3_1_7b"
SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
METHODS = ["random", "tournament", "alternis"]
MODELS = {
    "primary": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "control": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify():
    manifest = read(ROOT / "MANIFEST.json")
    for rel, expected in manifest.items():
        require(digest(ROOT / rel) == expected, f"Checksum mismatch: {rel}")
    for version, expected in [("v2", 90), ("v3", 500)]:
        size_limit = read(ROOT / f"evidence/{version}/space.json")["baseline_parameter_bytes"]
        for method in METHODS:
            for seed in SEEDS:
                p = ROOT / f"evidence/{version}/search_{method}_{seed}.json"
                d = read(p)
                rows = d["history"]
                require(d["complete"], f"Incomplete search: {p.name}")
                require(len(rows) == expected, f"Wrong score count: {p.name}")
                require(len({tuple(r["genome"]) for r in rows}) == expected,
                        f"Duplicate genome: {p.name}")
                require(all(math.isfinite(r["nll"]) for r in rows), f"Invalid NLL: {p.name}")
                require(all(r["parameter_bytes"] <= size_limit for r in rows),
                        f"Candidate exceeds model-size budget: {p.name}")
                if version == "v3":
                    require(d["generation"] == 30, f"Missing generations: {p.name}")
                    gens = read(p.with_name(p.stem + "_generations.json"))
                    require([g["generation"] for g in gens] == list(range(31)),
                            f"Incomplete generation trace: {p.name}")
                    require(all(g["evaluated"] == 18 + 16 * g["generation"] +
                                int(g["generation"] >= 15) + int(g["generation"] >= 30)
                                for g in gens), f"Wrong generation allowance: {p.name}")
        require(read(ROOT / f"evidence/{version}/selected_reload_check.json")["pass"],
                f"Archived {version} export reload check did not pass")
    require(len(read(ROOT / "evidence/v2/outcomes.json")) == 38, "V2 result count")
    require(len(read(ROOT / "evidence/v2/awq_control/outcomes.json")) == 4, "AWQ result count")
    fresh = read(ROOT / "evidence/v3/heldout_text_fresh.json")
    require(len(fresh) == 62, "V3 fresh text result count")
    require(all(r["split"] in {"news_fresh", "imdb_fresh"} for r in fresh.values()),
            "Old text outcomes mixed into primary evidence")
    require(all(r["parameter_bytes"] == 968001536 for r in fresh.values()),
            "V3 selected result size mismatch")
    require(len(read(ROOT / "evidence/v3/heldout_task.json")) == 4, "V3 task result count")
    require(len(read(ROOT / "evidence/v3/benchmarks.json")) == 4, "V3 benchmark result count")
    print("PASS — checksums, 2,700 V2 scores, 15,000 V3 scores, generations and result counts.")


def results(folder=None):
    folder = Path(folder) if folder else ROOT / "evidence/v3"
    rows = read(folder / "heldout_text_fresh.json")
    print("V3 corrected held-out NLL gains: positive means lower loss than random.")
    for corpus in ["news_fresh", "imdb_fresh"]:
        for method in ["tournament", "alternis"]:
            differences = [rows[f"random_{s}:{corpus}"]["nll"] -
                           rows[f"{method}_{s}:{corpus}"]["nll"] for s in SEEDS]
            mean = statistics.mean(differences)
            half = 2.2621571628540993 * statistics.stdev(differences) / math.sqrt(10)
            wins = sum(d > 1e-12 for d in differences)
            print(f"{corpus:11} {method:10} {mean:.8f} "
                  f"95% CI [{mean-half:.8f}, {mean+half:.8f}], {wins}/10 improved seeds")
    print("These are fixed-corpus paired-seed intervals, not general accuracy claims.")


def download_models():
    from huggingface_hub import snapshot_download
    paths = {}
    for key, (repo_id, revision) in MODELS.items():
        path = ROOT / "models" / repo_id.split("/")[-1] / revision
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(path))
        paths[key] = str(path)
    write(ROOT / ".local_models.json", paths)
    print("PASS — both pinned model snapshots downloaded; weights stay outside Git.")


def runtime_hashes(run):
    files = list(run.rglob("*.py")) + list(run.rglob("*.md"))
    for base in [run / "v2", run / "v3" / STUDY]:
        files.extend(base.glob("data/*"))
        files.extend(base.glob("protocol.json"))
        files.extend(base.glob("inputs.json"))
    files.extend((run / "v2/awq_control/data").glob("*"))
    files.extend([run / "v3/data_manifest.json", run / "v3" / STUDY / "fresh_data_manifest.json"])
    return {str(p.relative_to(run)): digest(p) for p in sorted(set(files)) if p.is_file()}


def prepare(version, name, max_hours=0, max_disk_gb=0):
    require(version != "v3" or (max_hours == 0 and max_disk_gb == 0),
            "V3 wide30 is uncapped; optional resource-limit flags are supported only for V2")
    require(name and all(c.isalnum() or c in "_-" for c in name),
            "Run name must contain only letters, digits, underscores or hyphens")
    run = ROOT / "runs" / name
    if run.exists():
        meta = read(run / "reproduction.json")
        require(meta["study"] == version, "Run name already belongs to another study")
        for rel, expected in meta["hashes"].items():
            require(digest(run / rel) == expected, f"Frozen runtime file changed: {rel}")
        return run
    paths = read(ROOT / ".local_models.json")
    for key in MODELS:
        require((Path(paths[key]) / "config.json").exists(), f"Missing {key} model; download models first")
    require(max_hours >= 0 and max_disk_gb >= 0, "Resource limits must be nonnegative")
    shutil.copytree(ROOT / "src", run, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "inputs/v2", run / "v2", dirs_exist_ok=True)
    shutil.copytree(ROOT / "inputs/v3", run / "v3", dirs_exist_ok=True)
    # Preserve prior-task exclusions for the fresh-data preparation audit.
    shutil.copytree(ROOT / "inputs/exclusions", run, dirs_exist_ok=True)
    inp = read(run / "v2/inputs.json")
    inp.update(model_path=paths["primary"], buddy_model_path=paths["control"])
    write(run / "v2/inputs.json", inp)
    inp3 = read(run / "v3" / STUDY / "inputs.json")
    inp3.update(model_path=paths["control"], v3_data_manifest=str(run / "v3/data_manifest.json"))
    write(run / "v3" / STUDY / "inputs.json", inp3)
    for base in [run / "v2", run / "v3" / STUDY]:
        p = read(base / "protocol.json")
        p.update(max_runtime_hours=max_hours or 1e9, max_disk_gb=max_disk_gb or 1e9)
        if base == run / "v2":
            p["evaluations_per_method_seed"] = 90
            p["runtime_rule"] = "Published 90 distinct scores fixed; never adapt to runtime."
        write(base / "protocol.json", p)
        (base / "logs").mkdir(exist_ok=True)
    write(run / "reproduction.json", {
        "study": version, "name": name, "max_hours": max_hours, "max_disk_gb": max_disk_gb,
        "hashes": runtime_hashes(run),
        "note": "New independent output; original evidence was not copied into this run."
    })
    return run


def run_study(args):
    require(sys.platform == "darwin", "MLX reruns require macOS on Apple Silicon")
    import fcntl
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/.scorer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another reproduction owns the scorer lock") from None
        run = prepare(args.study, args.name, args.max_hours, args.max_disk_gb)
        script = run / ("v2/supervise.py" if args.study == "v2" else "v3/wide30_study.py")
        log = run / "launcher.log"
        print(f"Running {args.study}; log: {log}. Ctrl-C stops; same command resumes.", flush=True)
        with log.open("a") as out:
            child = subprocess.Popen([sys.executable, str(script)], cwd=run,
                                     stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
            try:
                code = child.wait()
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
                print("Stopped; committed checkpoints retained. Resume with the same command.")
                return
        require(code == 0, f"Study exited {code}; inspect {log} and stage logs")
        print(f"PASS — {args.study} runner completed; outputs: {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    result_parser = sub.add_parser("results")
    result_parser.add_argument("--folder", help="Directory containing a completed heldout_text_fresh.json")
    sub.add_parser("download-models")
    for command in ["prepare", "run"]:
        p = sub.add_parser(command)
        p.add_argument("study", choices=["v2", "v3"])
        p.add_argument("--name", required=True)
        p.add_argument("--max-hours", type=float, default=0, help="V2 first-start wall-clock limit; 0 disables")
        p.add_argument("--max-disk-gb", type=float, default=0, help="V2 per-run directory limit; 0 disables")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify()
        elif args.command == "results":
            results(args.folder)
        elif args.command == "download-models":
            download_models()
        elif args.command == "prepare":
            print(f"PASS — prepared {prepare(args.study, args.name, args.max_hours, args.max_disk_gb)}")
        else:
            run_study(args)
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"ERROR — {exc}\n")


if __name__ == "__main__":
    main()
