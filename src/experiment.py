"""Resumable local MLX study with an automatic HTML report and final export."""
import argparse
import copy
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HOME=str(ROOT / "cache/huggingface"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                  HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", MPLCONFIGDIR=str(ROOT / "cache/matplotlib"))
from search import Space, alternis_pool, tournament_pool

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default

def ident(g):
    return "bf16" if g is None else hashlib.sha256(bytes(g)).hexdigest()[:20]

def log(event, **fields):
    row = {"time": time.time(), "event": event, **fields}
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")

class Runner:
    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm import load
        self.mx, self.nn, self.np, self.flatten = mx, nn, np, tree_flatten
        self.unflatten = tree_unflatten
        self.p = read_json(ROOT / "protocol.json")
        self.inputs = read_json(ROOT / "inputs.json")
        self.run = read_json(ROOT / "run.json", {"started": time.time(), "status": "initializing"})
        write_json(ROOT / "run.json", self.run)
        self.deadline = self.run["started"] + self.p["max_runtime_hours"] * 3600
        self.db = sqlite3.connect(ROOT / "results.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS evaluations (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.mx.set_cache_limit(1024 ** 3)
        self.model, self.tokenizer, self.config = load(self.inputs["model_path"], return_config=True)
        self.model.eval()
        self.original = dict(tree_flatten(self.model.leaf_modules(), is_leaf=nn.Module.is_module))
        self.original = {k: v for k, v in self.original.items()
                         if hasattr(v, "to_quantized") and v.weight.shape[-1] % self.p["group_size"] == 0}
        self.names = sorted(k for k, v in self.original.items() if ".layers." in k and isinstance(v, nn.Linear))
        if len(self.names) < 10:
            raise RuntimeError(f"Unexpected search space: {len(self.names)} modules")
        self.float_bytes = self.model_bytes()
        self.bank = {}
        for name, module in self.original.items():
            variants = (3, 4, 6) if name in self.names else (4,)
            self.bank[name] = {}
            for bits in variants:
                q = module.to_quantized(group_size=self.p["group_size"], bits=bits)
                mx.eval(q.parameters())
                self.bank[name][bits] = q
        self.space = Space([{b: self.module_bytes(self.bank[n][b]) for b in (3, 4, 6)} for n in self.names])
        self.base = tuple([4] * len(self.names))
        self.apply(self.base)
        self.baseline_bytes = self.model_bytes()
        self.fixed_bytes = self.baseline_bytes - self.space.budget
        self.data = {}
        for split, tokens in [("train", self.p["search_tokens"]), ("validation", self.p["validation_tokens"]), ("test", self.p["test_tokens"])]:
            token_path = ROOT / "data" / f"{split}_tokens.npy"
            if token_path.exists():
                blocks = np.load(token_path)
            else:
                ids = self.tokenizer.encode((ROOT / "data" / f"{split}.txt").read_text(), add_special_tokens=False)
                length = self.p["sequence_length"]
                starts = list(range(0, len(ids) - length, length))
                random.Random(20261017).shuffle(starts)
                starts = starts[:tokens // length]
                blocks = np.array([ids[s:s + length + 1] for s in starts], dtype=np.int32)
                np.save(token_path, blocks)
            if blocks.shape != (tokens // self.p["sequence_length"], self.p["sequence_length"] + 1):
                raise RuntimeError(f"Unexpected token shape for {split}: {blocks.shape}")
            self.data[split] = blocks
        self.arc = read_json(ROOT / "data/arc_easy.json")
        write_json(ROOT / "space.json", {"names": self.names, "costs": self.space.costs,
            "float_parameter_bytes": self.float_bytes, "baseline_parameter_bytes": self.baseline_bytes,
            "fixed_parameter_bytes": self.fixed_bytes, "search_parameter_budget": self.space.budget,
            "device": str(mx.default_device()), "tokenizer": type(self.tokenizer).__name__,
            "token_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in self.data.items()}})
        log("initialized", genes=len(self.names), baseline_bytes=self.baseline_bytes)

    def guard(self):
        if time.time() > self.deadline:
            raise TimeoutError("Authorized 12-hour runtime limit reached; checkpoints retained")
        if self.mx.get_active_memory() > 38 * 1024 ** 3:
            raise MemoryError("MLX active memory exceeded 38 GiB")

    def disk_guard(self):
        size = int(subprocess.check_output(["du", "-sk", str(ROOT)], text=True).split()[0]) * 1024
        if size > self.p["max_disk_gb"] * 10 ** 9:
            raise RuntimeError("Experiment disk limit exceeded")

    def module_bytes(self, module):
        return sum(a.nbytes for _, a in self.flatten(module.parameters()))

    def model_bytes(self):
        return self.module_bytes(self.model)

    def apply(self, g):
        if g is None:
            self.model.update_modules(self.unflatten(list(self.original.items())))
        else:
            if len(g) != len(self.names) or self.space.cost(g) > self.space.budget:
                raise ValueError("Candidate violates shape or byte budget")
            selected = dict(zip(self.names, g))
            self.model.update_modules(self.unflatten([(n, variants[selected.get(n, 4)]) for n, variants in self.bank.items()]))
        self.mx.eval(self.model.parameters())

    def get(self, key):
        row = self.db.execute("SELECT value FROM evaluations WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO evaluations VALUES (?, ?)", (key, json.dumps(value, allow_nan=False)))
        self.db.commit()

    def evaluate(self, g, split):
        key = f"nll:{split}:{ident(g)}"
        old = self.get(key)
        if old is not None:
            return old
        self.guard()
        self.apply(g)
        mx, nn = self.mx, self.nn
        started = time.perf_counter()
        losses = []
        for block in self.data[split]:
            self.guard()
            batch = mx.array(block[None, :])
            logits = self.model(batch[:, :-1])
            loss = nn.losses.cross_entropy(logits.astype(mx.float32), batch[:, 1:], reduction="mean")
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite loss")
            losses.append(value)
            del logits, loss, batch
        mean = statistics.mean(losses)
        result = {"id": ident(g), "genome": list(g) if g is not None else None,
                  "split": split, "nll": mean, "perplexity": math.exp(mean),
                  "block_nll": losses, "tokens": self.data[split].shape[0] * self.p["sequence_length"],
                  "seconds": time.perf_counter() - started,
                  "parameter_bytes": self.model_bytes(), "process_peak_mlx_bytes": mx.get_peak_memory()}
        self.put(key, result)
        log("evaluation", id=ident(g), split=split, nll=mean, seconds=result["seconds"])
        return result

    def task(self, g):
        key = f"arc:{ident(g)}"
        old = self.get(key)
        if old is not None:
            return old
        self.apply(g)
        results = []
        mx, nn = self.mx, self.nn
        for row in self.arc:
            self.guard()
            prefix = self.tokenizer.encode("Question: " + row["question"] + "\nAnswer:", add_special_tokens=False)
            scores = []
            for answer in row["choices"]["text"]:
                suffix = self.tokenizer.encode(" " + answer, add_special_tokens=False)
                ids = mx.array([prefix + suffix])
                logits = self.model(ids[:, :-1])[:, len(prefix) - 1:, :]
                loss = nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, len(prefix):], reduction="mean").item()
                scores.append(loss)
                del logits, ids
            index = min(range(len(scores)), key=lambda i: scores[i])
            prediction = row["choices"]["label"][index]
            results.append({"id": row["id"], "predicted": prediction, "answer": row["answerKey"],
                            "correct": prediction == row["answerKey"], "scores": scores})
        result = {"accuracy": statistics.mean(r["correct"] for r in results), "n": len(results), "items": results}
        self.put(key, result)
        return result

    def search_run(self, method, seed):
        state_path = ROOT / f"search_{method}_{seed}.json"
        prior = read_json(state_path, {})
        if prior.get("complete"):
            return prior
        rng = random.Random(seed)
        population = self.space.initial(seed, self.p["population"])
        seen, history = set(), []
        start = time.perf_counter()
        def evaluate(g):
            score = self.evaluate(g, "train")
            seen.add(g)
            history.append({"id": ident(g), "genome": list(g), "nll": score["nll"],
                            "parameter_bytes": score["parameter_bytes"]})
            state = {"method": method, "seed": seed, "complete": False, "history": history}
            write_json(state_path, state)
            self.run.update(status="searching", method=method, seed=seed, evaluated=len(history), updated=time.time())
            write_json(ROOT / "run.json", self.run)
            return score["nll"]
        losses = [evaluate(g) for g in population]
        while len(history) < self.p["evaluations_per_method_seed"]:
            self.guard()
            elite = min(range(len(losses)), key=lambda i: losses[i])
            next_pop, next_losses = [population[elite]], [losses[elite]]
            pool = (alternis_pool(losses, rng) if method == "alternis" else
                    tournament_pool(losses, rng) if method == "tournament" else None)
            attempts = 0
            while len(next_pop) < self.p["population"] and len(history) < self.p["evaluations_per_method_seed"]:
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("Search failed to produce a new feasible candidate")
                if method == "random":
                    child = self.space.random(rng)
                else:
                    a, b = rng.sample(pool, 2)
                    child = self.space.child(population[a], population[b], rng)
                if child in seen:
                    continue
                next_losses.append(evaluate(child))
                next_pop.append(child)
            population, losses = next_pop, next_losses
        state = {"method": method, "seed": seed, "complete": True, "history": history,
                 "seconds_this_invocation": time.perf_counter() - start}
        write_json(state_path, state)
        log("search_complete", method=method, seed=seed, evaluated=len(history))
        return state

    def smoke(self):
        mx, nn = self.mx, self.nn
        self.apply(None)
        block = mx.array(self.data["train"][:1])
        reference_logits = self.model(block[:, :-1])
        reference_loss = nn.losses.cross_entropy(reference_logits.astype(mx.float32), block[:, 1:], reduction="mean").item()
        del reference_logits
        self.apply(self.base)
        first = self.model(block[:, :-1])
        loss1 = nn.losses.cross_entropy(first.astype(mx.float32), block[:, 1:], reduction="mean").item()
        sampled = first[0, -1, :128].astype(mx.float32)
        mx.eval(sampled)
        del first
        self.apply(None)
        self.apply(self.base)
        second = self.model(block[:, :-1])
        loss2 = nn.losses.cross_entropy(second.astype(mx.float32), block[:, 1:], reduction="mean").item()
        if abs(loss1 - loss2) > 1e-6:
            raise AssertionError("Candidate switching changes deterministic loss")
        del second
        # Compare cached per-module conversion with the library's standard converter.
        from mlx_lm.utils import quantize_model
        self.apply(None)
        self.model, _ = quantize_model(self.model, self.config, 64, 4)
        standard = self.model(block[:, :-1])
        delta = mx.max(mx.abs(standard[0, -1, :128].astype(mx.float32) - sampled)).item()
        if delta > .001:
            raise AssertionError(f"Baseline differs from standard MLX converter: {delta}")
        del standard
        self.apply(self.base)
        result = {"reference_first_block_nll": reference_loss, "four_bit_first_block_nll": loss1,
                  "repeat_delta": abs(loss1 - loss2), "standard_converter_max_logit_delta": delta,
                  "genes": len(self.names), "baseline_parameter_bytes": self.baseline_bytes}
        write_json(ROOT / "smoke.json", result)
        pilot = [self.base] + self.space.initial(123, 5)[1:]
        timings = []
        for g in pilot:
            timings.append(self.evaluate(g, "train")["seconds"])
        result["pilot_seconds"] = timings
        result["median_candidate_seconds"] = statistics.median(timings)
        result["upper_search_hours_without_cache"] = statistics.median(timings) * 540 / 3600
        write_json(ROOT / "smoke.json", result)
        self.export(self.base, ROOT / "models/pilot_4bit")
        self.check_export(ROOT / "models/pilot_4bit", self.base)
        self.disk_guard()
        log("smoke_complete", **result)
        print("PASS — MLX baseline, candidate switching, pilot evaluations and export/reload verified.", flush=True)

    def export(self, g, target):
        from mlx_lm.utils import save
        self.apply(g)
        cfg = copy.deepcopy(self.config)
        q = {"group_size": 64, "bits": 4, "mode": "affine"}
        q.update({name: {"group_size": 64, "bits": int(bits), "mode": "affine"}
                  for name, bits in zip(self.names, g)})
        cfg["quantization"] = q
        cfg["quantization_config"] = q
        target.mkdir(parents=True, exist_ok=True)
        save(target, self.inputs["model_path"], self.model, self.tokenizer, cfg, donate_model=False)
        write_json(target / "experiment_genome.json", {"id": ident(g), "names": self.names, "bits": list(g)})

    def check_export(self, target, g):
        from mlx_lm import load
        mx = self.mx
        self.apply(g)
        x = mx.array(self.data["train"][:1, :65])
        a = self.model(x)[:, -1, :].astype(mx.float32)
        mx.eval(a)
        reloaded, _ = load(str(target))
        b = reloaded(x)[:, -1, :].astype(mx.float32)
        delta = mx.max(mx.abs(a - b)).item()
        if delta > .01:
            raise AssertionError(f"Export/reload logit mismatch: {delta}")
        write_json(target / "reload_check.json", {"max_logit_delta": delta, "pass": True})
        del reloaded, a, b
        gc.collect()
        mx.clear_cache()

    def full(self):
        self.disk_guard()
        if not (ROOT / "smoke.json").exists() or not (ROOT / "models/pilot_4bit/reload_check.json").exists():
            self.smoke()
        # Interleave method order across seeds to reduce systematic thermal ordering.
        states = []
        for index, seed in enumerate(self.p["seeds"]):
            methods = self.p["methods"][index:] + self.p["methods"][:index]
            for method in methods:
                states.append(self.search_run(method, seed))
        self.run.update(status="validation", updated=time.time())
        write_json(ROOT / "run.json", self.run)
        finalists = []
        for state in states:
            shortlist = sorted(state["history"], key=lambda x: x["nll"])[:3]
            candidates = {tuple(x["genome"]) for x in shortlist} | {self.base}
            scored = [(self.evaluate(g, "validation")["nll"], g) for g in sorted(candidates)]
            loss, winner = min(scored)
            finalists.append({"method": state["method"], "seed": state["seed"], "id": ident(winner),
                              "genome": list(winner), "validation_nll": loss})
        # Freeze all choices before reading test scores.
        write_json(ROOT / "finalists.json", finalists)
        best = min(finalists, key=lambda r: (r["validation_nll"], r["method"], r["seed"]))
        write_json(ROOT / "deployment_choice.json", best)
        self.run.update(status="held_out_evaluation", updated=time.time())
        write_json(ROOT / "run.json", self.run)
        outcomes = []
        configs = [("higher_precision", None, None), ("ordinary_4bit", None, self.base)] + [
            (r["method"], r["seed"], tuple(r["genome"])) for r in finalists]
        for method, seed, g in configs:
            result = self.evaluate(g, "test")
            task = self.task(g)
            outcomes.append({"method": method, "seed": seed, "id": ident(g), "genome": list(g) if g else None,
                             "test_nll": result["nll"], "perplexity": result["perplexity"],
                             "parameter_bytes": result["parameter_bytes"], "arc_accuracy": task["accuracy"]})
            write_json(ROOT / "outcomes.json", outcomes)
        winner = tuple(best["genome"])
        self.export(winner, ROOT / "models/selected")
        self.check_export(ROOT / "models/selected", winner)
        self.disk_guard()
        self.run.update(status="accuracy_complete", updated=time.time())
        write_json(ROOT / "run.json", self.run)
        write_report()
        print("PASS — all searches and held-out accuracy evaluations complete.", flush=True)

def write_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = read_json(ROOT / "protocol.json")
    run = read_json(ROOT / "run.json", {})
    outcomes = read_json(ROOT / "outcomes.json", [])
    speed = read_json(ROOT / "benchmarks.json", [])
    states = [read_json(path) for path in sorted(ROOT.glob("search_*.json"))]
    conclusion = "Experiment incomplete; no conclusion about Alternis is justified yet."
    if len(outcomes) == 11:
        by = {(r["method"], r["seed"]): r for r in outcomes}
        wins = {m: sum(by[("alternis", s)]["test_nll"] < by[(m, s)]["test_nll"] for s in p["seeds"])
                for m in ("random", "tournament")}
        if all(v == len(p["seeds"]) for v in wins.values()):
            conclusion = "Alternis had lower held-out loss than both searches in all three paired seeds. This is encouraging exploratory evidence for this setup, not proof of general superiority."
        else:
            conclusion = (f"Alternis did not consistently outperform both competing searches: lower test loss than random in {wins['random']}/3 seeds and tournament in {wins['tournament']}/3. "
                          "This experiment does not establish an Alternis-specific quantization benefit.")
        if all(by[("alternis", s)]["id"] == by[("ordinary_4bit", None)]["id"] for s in p["seeds"]):
            conclusion += " Every Alternis run selected the ordinary four-bit baseline during validation."
    if states:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = {"random": "#697386", "tournament": "#cf7a25", "alternis": "#1968bf"}
        for state in states:
            history = state.get("history", [])
            best, curve = math.inf, []
            for row in history:
                best = min(best, row["nll"])
                curve.append(best)
            ax.plot(range(1, len(curve) + 1), curve, color=colors[state["method"]], alpha=.65,
                    label=f"{state['method']} seed {state['seed']}")
        ax.set(xlabel="Candidate scores supplied to each run", ylabel="Best search negative log likelihood", title="Search progress (lower is better)")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(ROOT / "search_progress.png", dpi=160)
        plt.close(fig)
    rows = "".join(f"<tr><td>{r['method']}</td><td>{r['seed'] if r['seed'] is not None else '—'}</td>"
                   f"<td>{r['test_nll']:.5f}</td><td>{r['perplexity']:.3f}</td><td>{r['arc_accuracy']:.1%}</td>"
                   f"<td>{r['parameter_bytes']/1e9:.3f}</td></tr>" for r in outcomes)
    speedrows = "".join(f"<tr><td>{html.escape(r['label'])}</td><td>{r['median_prompt_tps']:.1f}</td>"
                       f"<td>{r['median_generation_tps']:.1f}</td><td>{r['peak_mlx_bytes']/1e9:.3f}</td>"
                       f"<td>{r['parameter_bytes']/1e9:.3f}</td></tr>" for r in speed)
    evaluation_count = 0
    if (ROOT / "results.sqlite").exists():
        con = sqlite3.connect(ROOT / "results.sqlite")
        evaluation_count = con.execute("SELECT count(*) FROM evaluations WHERE key LIKE 'nll:train:%'").fetchone()[0]
        con.close()
    summary = {"status": run.get("status"), "conclusion": conclusion, "logical_search_evaluations": sum(len(s.get("history", [])) for s in states),
               "unique_search_evaluations_including_pilot": evaluation_count, "outcomes": outcomes, "benchmarks": speed}
    write_json(ROOT / "summary.json", summary)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Alternis quantization experiment</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#192436}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #dce2e8;text-align:left;padding:9px}}.verdict{{padding:20px;background:#edf3f9;border-left:4px solid #1968bf}}img{{max-width:100%}}code{{font-size:13px}}</style></head><body>
<h1>Does Alternis improve Qwen3.5-2B quantization?</h1><p>Status: <strong>{html.escape(run.get('status','unknown'))}</strong></p>
<p class="verdict">{html.escape(conclusion)}</p>
<h2>Held-out results</h2><table><tr><th>Method</th><th>Seed</th><th>Test NLL ↓</th><th>Perplexity ↓</th><th>ARC-Easy accuracy ↑</th><th>Parameter GB</th></tr>{rows}</table>
<p>WikiText-2: 65,536 scored test tokens in independent 512-token contexts. ARC-Easy: 64 fixed public test questions, answer likelihood normalized by answer length; these are exploratory task scores, not official benchmark leaderboard results. Higher precision uses the original checkpoint tensor dtypes.</p>
<h2>Search</h2><img src="search_progress.png" alt="Best search loss by evaluation"><p>Three methods × three seeds × 60 distinct candidate scores = 540 logical evaluations. Identical configurations share cached deterministic results. Unique search objective evaluations including pilot: {evaluation_count}.</p>
<p>Each GA starts with the same 12 configurations, uses three selected parents, two-point crossover, one elite, mutation probability 1/number of genes, and the same byte-budget repair. Alternis orders candidates worst,best,second-worst,second-best, then selects a valid random center and its neighbors. Random search shares the initial population and uses the same candidate generator and repair. The paper used population 30; this feasibility study uses 12.</p>
<p>Search uses 8,192 tokens from the training split. Each run's three best search candidates and the uniform four-bit baseline are compared on 32,768 validation tokens. Finalists and the export choice are frozen before test evaluation. The exported model is selected by validation loss, never test loss.</p>
<h2>Inference measurements</h2><table><tr><th>Configuration</th><th>Prompt tokens/s</th><th>Generation tokens/s</th><th>Peak MLX GB</th><th>Parameter GB</th></tr>{speedrows}</table>
<p>Fresh process for each configuration, one warmup then three timed deterministic generations on a common prompt. Peak memory includes runtime allocations but excludes the search model bank; MLX peak memory is not total system memory. Results can vary with temperature and other Mac activity.</p>
<h2>Limits and interpretation</h2><ul><li>Three seeds and 64 task questions provide limited statistical power. Search-set improvements alone do not establish better quantization.</li><li>The search is restricted to affine 3/4/6-bit linear matrices inside transformer layers; embeddings and other supported matrices stay at four bits. This does not test every possible quantizer or use of Alternis.</li><li>All candidates fit the uniform four-bit parameter-byte budget including scales and biases. Actual bytes may be slightly below the limit. Speed and runtime memory are measured separately.</li><li>The parent-selection algorithm does not guarantee better offspring or preserve useful genetic diversity. Explicit elitism is shared by both GAs.</li><li>AWQ/GPTQ and MLX sensitivity-based quantization are relevant future baselines; this run isolates the selection-method comparison and does not claim state-of-the-art quantization.</li></ul>
<h2>Reproducibility</h2><p>See <a href="protocol.json">protocol</a>, <a href="inputs.json">pinned inputs and versions</a>, <a href="space.json">search space and token hashes</a>, <a href="summary.json">machine-readable summary</a>, and <a href="events.jsonl">event log</a>. Every objective result is checkpointed in results.sqlite. Search records include every candidate's complete bit allocation.</p>
<p>Paper: <a href="file:///Users/smitanand/Downloads/anand-2015-ijca-907067.pdf">Anand, Afreen &amp; Yazdani (2015), A Novel and Efficient Selection Method in Genetic Algorithm</a>, §§3.5–4. Comparison tools: <a href="https://github.com/ml-explore/mlx-lm">MLX-LM</a>.</p>
</body></html>"""
    (ROOT / "report.html").write_text(doc)

def benchmark(label, model_path, genome=None):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import quantize_model
    from mlx.utils import tree_flatten
    model, tokenizer, config = load(model_path, return_config=True)
    if genome is not None:
        space = read_json(ROOT / "space.json")
        selected = dict(zip(space["names"], genome))
        def predicate(path, module):
            if path in selected:
                return {"bits": int(selected[path]), "group_size": 64, "mode": "affine"}
            return True
        model, _ = quantize_model(model, config, 64, 4, quant_predicate=predicate)
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Explain how a computer stores numbers using bits, and give examples of the tradeoff between precision and memory."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    results = []
    for i in range(4):
        mx.reset_peak_memory()
        final = None
        for final in stream_generate(model, tokenizer, prompt, max_tokens=128, sampler=make_sampler(temp=0)):
            pass
        if final is None:
            raise RuntimeError("No generation response")
        if i:
            results.append({"prompt_tps": final.prompt_tps, "generation_tps": final.generation_tps,
                            "tokens": final.generation_tokens, "peak_mlx_bytes": mx.get_peak_memory()})
    row = {"label": label, "median_prompt_tps": statistics.median(r["prompt_tps"] for r in results),
           "median_generation_tps": statistics.median(r["generation_tps"] for r in results),
           "peak_mlx_bytes": max(r["peak_mlx_bytes"] for r in results),
           "parameter_bytes": sum(x.nbytes for _, x in tree_flatten(model.parameters())), "trials": results}
    out = read_json(ROOT / "benchmarks.json", [])
    out = [r for r in out if r["label"] != label] + [row]
    write_json(ROOT / "benchmarks.json", out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "run", "report", "benchmark", "finish"])
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.command == "report":
        write_report()
        return
    if args.command == "benchmark":
        inputs = read_json(ROOT / "inputs.json")
        if args.label == "higher_precision":
            benchmark(args.label, inputs["model_path"])
        elif args.label == "ordinary_4bit":
            benchmark(args.label, str(ROOT / "models/pilot_4bit"))
        else:
            choice = next(r for r in read_json(ROOT / "finalists.json") if f"{r['method']}_{r['seed']}" == args.label)
            benchmark(args.label, inputs["model_path"], choice["genome"])
        return
    if args.command == "finish":
        run = read_json(ROOT / "run.json")
        run.update(status="complete", completed=time.time())
        write_json(ROOT / "run.json", run)
        write_report()
        print("PASS — experiment complete; report and verified model saved.")
        return
    runner = Runner()
    if args.command == "smoke":
        runner.smoke()
    else:
        runner.full()

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        state = read_json(ROOT / "run.json", {})
        state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed",
                     error=f"{type(exc).__name__}: {exc}", updated=time.time())
        write_json(ROOT / "run.json", state)
        log("failure", error=state["error"])
        traceback.print_exc()
        try:
            write_report()
        except Exception:
            pass
        sys.exit(1)
