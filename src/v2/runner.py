"""Expanded, checkpointed benchmark. V1 artifacts are never modified."""
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
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE))
import experiment as v1
v1.ROOT = ROOT
os.environ.update(MPLCONFIGDIR=str(ROOT / "cache/matplotlib"), HF_HUB_OFFLINE="1")
from operators import InformedSpace
read = v1.read_json
write = v1.write_json

def ident(g):
    if g is None:
        return "higher_precision"
    if isinstance(g, str):
        return g
    return hashlib.sha256(bytes(g)).hexdigest()[:20]
v1.ident = ident

def status(stage, **extra):
    state = read(ROOT / "run.json", {"started": time.time()})
    state.pop("error", None)
    state.update(status=stage, updated=time.time(), **extra)
    write(ROOT / "run.json", state)

def guard():
    state = read(ROOT / "run.json")
    if time.time() - state["started"] > read(ROOT / "protocol.json")["max_runtime_hours"] * 3600:
        raise TimeoutError("Expanded experiment reached its configured wall-clock limit")

class Runner(v1.Runner):
    def __init__(self):
        super().__init__()
        self.run.pop("error", None)
        for name, original in self.original.items():
            for bits in (2, 3, 4, 6, 8):
                if bits not in self.bank[name]:
                    self.bank[name][bits] = original.to_quantized(group_size=64, bits=bits)
                    self.mx.eval(self.bank[name][bits].parameters())
        self.space = InformedSpace([{b: self.module_bytes(self.bank[n][b]) for b in (2, 4, 8)} for n in self.names])
        self.fixed_bytes = self.baseline_bytes - self.space.budget
        self.calibration = self.np.load(ROOT / "data/calibration_tokens.npy")
        self.data["sensitivity"] = self.calibration[:4]
        sensitivity = read(ROOT / "sensitivity.json")
        if sensitivity and sensitivity.get("complete"):
            self.space.informed = tuple(sensitivity["genome"])
        write(ROOT / "space.json", {"names": self.names, "costs": self.space.costs,
            "float_parameter_bytes": self.float_bytes, "baseline_parameter_bytes": self.baseline_bytes,
            "fixed_parameter_bytes": self.fixed_bytes, "search_parameter_budget": self.space.budget,
            "token_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in self.data.items()}})

    def apply(self, g):
        if g is None:
            self.model.update_modules(self.unflatten(list(self.original.items())))
        else:
            if isinstance(g, str):
                if not g.startswith("uniform_"):
                    raise ValueError(g)
                bits = int(g.split("_")[1])
                modules = [(n, variants[bits]) for n, variants in self.bank.items()]
            else:
                if not getattr(self, "probing", False) and self.space.cost(g) > self.space.budget:
                    raise ValueError("Candidate exceeds byte budget")
                chosen = dict(zip(self.names, g))
                modules = [(n, variants[chosen.get(n, 4)]) for n, variants in self.bank.items()]
            self.model.update_modules(self.unflatten(modules))
        self.mx.eval(self.model.parameters())

    def evaluate(self, g, split):
        # V1 expects iterable integer genomes; strings designate uniform baselines here.
        if isinstance(g, str):
            key = f"nll:{split}:{g}"
            cached = self.get(key)
            if cached:
                return cached
            self.apply(g)
            return score_model(self, g, split)
        return super().evaluate(g, split)

    def pilot(self):
        status("pilot")
        base = self.evaluate(self.base, "train")
        times = [base["seconds"]]
        for g in self.space.initial(299, 5)[1:]:
            times.append(self.evaluate(g, "train")["seconds"])
        median = statistics.median(times)
        # Reproduction fixes the published allowance; timing never changes it.
        budget = self.p["evaluations_per_method_seed"]
        if budget != 90:
            raise RuntimeError("Published V2 protocol requires exactly 90 candidates")
        write(ROOT / "pilot.json", {"median_candidate_seconds": median, "budget_per_run": budget,
                                   "projected_uncached_search_hours": budget * 30 * median / 3600})
        # Smoke-check every intended bit width before committing to the search.
        x = self.mx.array(self.data["train"][:1, :-1])
        for bits in (2, 3, 4, 6, 8):
            self.apply(f"uniform_{bits}")
            value = self.model(x)[0, -1, :].astype(self.mx.float32)
            if not self.mx.all(self.mx.isfinite(value)).item():
                raise RuntimeError(f"Nonfinite {bits}-bit logits")
        self.export(self.base, ROOT / "models/pilot_4bit")
        self.check_export(ROOT / "models/pilot_4bit", self.base)
        write(ROOT / "pilot_verified.json", {"pass": True, "widths": [2, 3, 4, 6, 8]})
        print("PASS — expanded pilot, five bit widths and export/reload verified.")

    def sensitivity(self):
        status("sensitivity")
        if self.space.informed is not None:
            return
        reference = self.evaluate(self.base, "sensitivity")["nll"]
        deltas = []
        self.probing = True
        for i, name in enumerate(self.names):
            d = {4: 0.0}
            for bits in (2, 8):
                g = list(self.base)
                g[i] = bits
                d[bits] = self.evaluate(tuple(g), "sensitivity")["nll"] - reference
            deltas.append(d)
            status("sensitivity", matrix=i + 1, matrices=len(self.names))
        self.probing = False
        genome = self.space.sensitivity_allocation(deltas)
        result = self.evaluate(genome, "train")
        write(ROOT / "sensitivity.json", {"complete": True, "names": self.names, "deltas": deltas,
            "genome": list(genome), "reference_nll": reference, "search_nll": result["nll"],
            "parameter_bytes": result["parameter_bytes"], "method": "greedy local-loss sensitivity"})
        self.space.informed = genome

    def searches(self):
        if self.space.informed is None:
            raise RuntimeError("Sensitivity initialization missing")
        for index, seed in enumerate(self.p["seeds"]):
            offset = index % 3
            for method in self.p["methods"][offset:] + self.p["methods"][:offset]:
                self.search_run(method, seed)

    def freeze(self):
        if (ROOT / "finalists.json").exists():
            return
        status("validation")
        finalists = []
        for method in self.p["methods"]:
            for seed in self.p["seeds"]:
                state = read(ROOT / f"search_{method}_{seed}.json")
                if not state or not state["complete"]:
                    raise RuntimeError("Cannot select finalists before all searches finish")
                shortlist = sorted(state["history"], key=lambda x: x["nll"])[:3]
                genomes = {tuple(x["genome"]) for x in shortlist} | {self.base, self.space.informed}
                loss, g = min((self.evaluate(g, "validation")["nll"], g) for g in sorted(genomes))
                finalists.append({"label": f"{method}_{seed}", "method": method, "seed": seed,
                                  "genome": list(g), "id": ident(g), "validation_nll": loss})
        write(ROOT / "finalists.json", finalists)
        selected = min((r for r in finalists if r["method"] == "alternis"), key=lambda r: r["validation_nll"])
        write(ROOT / "alternis_choice.json", selected)

def score_model(evaluator, key, split):
    mx, nn = evaluator.mx, evaluator.nn
    losses = []
    started = time.perf_counter()
    for block in evaluator.data[split]:
        evaluator.guard()
        x = mx.array(block[None, :])
        logits = evaluator.model(x[:, :-1])
        value = nn.losses.cross_entropy(logits.astype(mx.float32), x[:, 1:], reduction="mean").item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite evaluation loss")
        losses.append(value)
        del logits, x
    result = {"id": key, "split": split, "nll": statistics.mean(losses), "perplexity": math.exp(statistics.mean(losses)),
              "block_nll": losses, "tokens": len(losses) * 512, "seconds": time.perf_counter() - started,
              "parameter_bytes": evaluator.model_bytes()}
    evaluator.put(f"nll:{split}:{key}", result)
    v1.log("evaluation", id=key, split=split, nll=result["nll"], seconds=result["seconds"])
    return result

class External(v1.Runner):
    def __init__(self, label, control=False, variant=None):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.utils import quantize_model
        self.mx, self.nn, self.np, self.flatten = mx, nn, np, tree_flatten
        self.label, self.control = label, control
        self.cache_key = ("uniform_4" if variant is not None and set(variant) == {4}
                          else ident(variant) if variant is not None else label)
        self.folder = ROOT / "awq_control" if control else ROOT
        self.p = read(ROOT / "protocol.json")
        self.inputs = read(ROOT / "inputs.json")
        self.deadline = read(ROOT / "run.json")["started"] + self.p["max_runtime_hours"] * 3600
        self.db = sqlite3.connect(self.folder / "results.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS evaluations (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.data = {s: np.load(self.folder / "data" / f"{s}_tokens.npy") for s in ["train", "validation", "test"]}
        self.arc = read(self.folder / "data/arc_easy.json")
        path = self.inputs["buddy_model_path"] if control else self.inputs["model_path"]
        if label in ("gptq_4", "combined_gptq", "awq_4"):
            path = str(self.folder / "models" / label)
        self.model, self.tokenizer, self.config = load(path, return_config=True)
        self.model.eval()
        if label.startswith("uniform_"):
            self.model, self.config = quantize_model(self.model, self.config, 64, int(label.split("_")[1]))
        elif variant is not None:
            names = read(ROOT / "space.json")["names"]
            chosen = dict(zip(names, variant))
            self.model, self.config = quantize_model(self.model, self.config, 64, 4,
                quant_predicate=lambda name, module: {"bits": chosen.get(name, 4), "group_size": 64, "mode": "affine"})
        mx.eval(self.model.parameters())

    def apply(self, _):
        pass

    def evaluate(self, _, split):
        cached = self.get(f"nll:{split}:{self.cache_key}")
        result = cached or score_model(self, self.cache_key, split)
        self.put(f"nll:{split}:{self.label}", result)
        return result

    def benchmark(self):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": "Explain how a computer stores numbers using bits, and give examples of the tradeoff between precision and memory."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.mx.clear_cache()
        gc.collect()
        trials = []
        for i in range(4):
            self.mx.reset_peak_memory()
            response = None
            for response in stream_generate(self.model, self.tokenizer, prompt, max_tokens=128, sampler=make_sampler(temp=0)):
                pass
            if response is None:
                raise RuntimeError("Generation failed")
            if i:
                trials.append({"prompt_tps": response.prompt_tps, "generation_tps": response.generation_tps,
                               "peak_mlx_bytes": self.mx.get_peak_memory(), "tokens": response.generation_tokens})
        return {"median_prompt_tps": statistics.median(x["prompt_tps"] for x in trials),
                "median_generation_tps": statistics.median(x["generation_tps"] for x in trials),
                "peak_mlx_bytes": max(x["peak_mlx_bytes"] for x in trials), "trials": trials}

def quantizer(label, control=False, pilot=False):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.utils import save
    from mlx.utils import tree_flatten
    from gptq_mixed import quantize_mixed
    inputs = read(ROOT / "inputs.json")
    folder = ROOT / "awq_control" if control else ROOT
    model_path = inputs["buddy_model_path"] if control else inputs["model_path"]
    target = folder / "models" / (label + "_pilot" if pilot else label)
    if (target / "verified.json").exists():
        return
    model, tokenizer, config = load(model_path, return_config=True)
    model.eval()
    data = np.load(folder / "data/calibration_tokens.npy")
    started = time.perf_counter()
    status("quantizing", quantizer=label, control=control)
    if label == "awq_4":
        from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS, awq_quantize, update_config
        awq_quantize(model, mx.array(data[:8 if pilot else 128, :-1]), AWQ_MODEL_CONFIGS[config["model_type"]],
                     bits=4, group_size=64, embed_bits=4, embed_group_size=64, n_grid=20)
        config = update_config(model, config)
    else:
        allocation = {}
        if label == "combined_gptq":
            allocation = dict(zip(read(ROOT / "space.json")["names"], read(ROOT / "alternis_choice.json")["genome"]))
        model, qconfig = quantize_mixed(model, data[:8] if pilot else data, allocation,
            progress=lambda i, n: status("quantizing", quantizer=label, matrix=i, matrices=n, control=control),
            guard=guard, limit=4 if pilot else None)
        config["quantization"] = qconfig
    config["quantization_config"] = config["quantization"]
    target.mkdir(parents=True, exist_ok=True)
    save(target, model_path, model, tokenizer, config, donate_model=False)
    x = mx.array(data[:1, :64])
    before = model(x)[:, -1, :].astype(mx.float32)
    mx.eval(before)
    reloaded, _ = load(str(target))
    after = reloaded(x)[:, -1, :].astype(mx.float32)
    delta = mx.max(mx.abs(before - after)).item()
    if not math.isfinite(delta) or delta > .01:
        raise RuntimeError(f"Quantizer export mismatch: {delta}")
    write(target / "verified.json", {"pass": True, "max_logit_delta": delta,
        "seconds": time.perf_counter() - started, "pilot": pilot,
        "parameter_bytes": sum(a.nbytes for _, a in tree_flatten(model.parameters()))})

def evaluate(label, control=False, validation_only=False):
    folder = ROOT / "awq_control" if control else ROOT
    variant, meta = None, {"label": label, "method": label, "seed": None}
    if label == "loss_sensitivity":
        variant = read(ROOT / "sensitivity.json")["genome"]
    elif label.startswith(("alternis_", "tournament_", "random_")):
        meta = next(x for x in read(ROOT / "finalists.json") if x["label"] == label)
        variant = meta["genome"]
    e = External(label, control, variant)
    val = e.evaluate(None, "validation")
    if validation_only:
        return
    if not (ROOT / "choices_frozen.json").exists():
        raise RuntimeError("Test evaluation attempted before choices were frozen")
    result = e.evaluate(None, "test")
    task = e.get(f"arc:{e.cache_key}") or e.task(e.cache_key)
    speed = e.benchmark()
    z, n, accuracy = 1.959963984540054, task["n"], task["accuracy"]
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
    row = {**meta, "validation_nll": val["nll"], "test_nll": result["nll"],
           "perplexity": result["perplexity"], "arc_accuracy": task["accuracy"], "arc_n": task["n"],
           "arc_wilson_95": [center-half, center+half],
           "parameter_bytes": result["parameter_bytes"], **speed}
    outcomes = read(folder / "outcomes.json", [])
    write(folder / "outcomes.json", [x for x in outcomes if x["label"] != label] + [row])
    status("held_out_evaluation", label=label, control=control)

def freeze_choices():
    if (ROOT / "choices_frozen.json").exists():
        return
    con = sqlite3.connect(ROOT / "results.sqlite")
    space = read(ROOT / "space.json")
    eligible = []
    for label in ["uniform_4", "loss_sensitivity", "gptq_4", "combined_gptq"]:
        row = con.execute("SELECT value FROM evaluations WHERE key=?", (f"nll:validation:{label}",)).fetchone()
        if not row:
            raise RuntimeError(f"Missing baseline validation: {label}")
        value = json.loads(row[0])
        if value["parameter_bytes"] <= space["baseline_parameter_bytes"]:
            eligible.append({"label": label, "validation_nll": value["nll"]})
    eligible += read(ROOT / "finalists.json")
    selected = min(eligible, key=lambda x: x["validation_nll"])
    write(ROOT / "choices_frozen.json", {"frozen": time.time(), "selected": selected,
          "eligible": eligible, "rule": "lowest validation NLL within uniform-four-bit parameter budget"})

def export_selected():
    from mlx_lm.utils import save
    from mlx_lm import load
    import mlx.core as mx
    choice = read(ROOT / "choices_frozen.json")["selected"]
    label = choice["label"]
    genome = choice.get("genome")
    if label == "loss_sensitivity":
        genome = read(ROOT / "sensitivity.json")["genome"]
    e = External(label, variant=genome)
    target = ROOT / "models/selected"
    save(target, read(ROOT / "inputs.json")["model_path"], e.model, e.tokenizer, e.config, donate_model=False)
    x = mx.array(e.data["train"][:1, :64])
    a = e.model(x)[:, -1, :].astype(mx.float32)
    mx.eval(a)
    other, _ = load(str(target))
    delta = mx.max(mx.abs(a - other(x)[:, -1, :].astype(mx.float32))).item()
    if not math.isfinite(delta) or delta > .01:
        raise RuntimeError("Selected model reload failed")
    write(target / "verified.json", {"pass": True, "max_logit_delta": delta, "selected": choice})

def report():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read(ROOT / "outcomes.json", [])
    buddy = read(ROOT / "awq_control/outcomes.json", [])
    protocol = read(ROOT / "protocol.json")
    state = read(ROOT / "run.json", {})
    comparisons = {}
    aggregates = {}
    by = {r["label"]: r for r in rows}
    for method in ["random", "tournament"]:
        if all(f"{m}_{s}" in by for m in ["alternis", method] for s in protocol["seeds"]):
            deltas = np.array([by[f"alternis_{s}"]["test_nll"] - by[f"{method}_{s}"]["test_nll"] for s in protocol["seeds"]])
            boot = np.random.default_rng(811).choice(deltas, size=(10000, len(deltas)), replace=True).mean(1)
            comparisons[method] = {"mean_paired_nll_difference": float(deltas.mean()),
                "bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(), "wins": int((deltas < 0).sum()),
                "ties": int((deltas == 0).sum()), "seeds": len(deltas)}
    for method in ["random", "tournament", "alternis"]:
        group = [r for r in rows if r.get("method") == method]
        if group:
            aggregates[method] = {"seeds": len(group), "mean_perplexity": statistics.mean(r["perplexity"] for r in group),
                "sd_perplexity": statistics.stdev(r["perplexity"] for r in group) if len(group) > 1 else 0,
                "mean_arc_accuracy": statistics.mean(r["arc_accuracy"] for r in group)}
    def table(items):
        body = "".join(f"<tr><td>{html.escape(r['label'])}</td><td>{r['perplexity']:.3f}</td><td>{r['arc_accuracy']:.1%} [{r['arc_wilson_95'][0]:.1%}, {r['arc_wilson_95'][1]:.1%}]</td><td>{r['parameter_bytes']/1e9:.3f}</td><td>{r['median_generation_tps']:.1f}</td><td>{r['peak_mlx_bytes']/1e9:.3f}</td></tr>" for r in items)
        return "<table><tr><th>Method</th><th>Perplexity ↓</th><th>ARC accuracy ↑</th><th>Weight GB</th><th>Decode tok/s</th><th>Peak MLX GB</th></tr>" + body + "</table>"
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        baselines = [r for r in rows if r.get("seed") is None]
        for r in baselines:
            ax.scatter(r["parameter_bytes"] / 1e9, r["perplexity"], s=50)
            ax.annotate(r["label"], (r["parameter_bytes"] / 1e9, r["perplexity"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Parameter storage (GB)", ylabel="Held-out perplexity (lower is better)", title="Qwen3.5-2B accuracy–storage tradeoff")
        fig.tight_layout()
        fig.savefig(ROOT / "tradeoff.png", dpi=160)
        plt.close(fig)
    findings = "Results are incomplete; no final conclusion is available."
    if len(comparisons) == 2:
        findings = " ".join(f"Against {m}, Alternis won {c['wins']}/{c['seeds']} seeds, tied {c['ties']}, with mean paired NLL difference {c['mean_paired_nll_difference']:+.5f} and bootstrap 95% interval [{c['bootstrap_95_interval'][0]:+.5f}, {c['bootstrap_95_interval'][1]:+.5f}]." for m, c in comparisons.items())
    write(ROOT / "summary.json", {"status": state.get("status"), "findings": findings, "paired_comparisons": comparisons, "aggregates": aggregates,
                                  "outcomes": rows, "awq_control": buddy, "selected": read(ROOT / "choices_frozen.json", {}).get("selected")})
    aggregate_text = " ".join(f"{m}: mean perplexity {a['mean_perplexity']:.3f}, standard deviation {a['sd_perplexity']:.3f}, mean task accuracy {a['mean_arc_accuracy']:.1%} across {a['seeds']} seeds." for m, a in aggregates.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Expanded Alternis benchmark</title><style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#192436}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.finding{{padding:18px;background:#edf3f9}}</style></head><body>
<h1>Expanded Alternis quantization benchmark</h1><p>Status: <strong>{html.escape(state.get('status','unknown'))}</strong></p><p class="finding">{html.escape(findings)}</p>
<h2>Qwen3.5-2B results</h2><p>{html.escape(aggregate_text)}</p>{table(rows)}<img src="tradeoff.png" alt="Accuracy and weight-storage tradeoff">
<h2>AWQ compatibility control: Qwen3-1.7B</h2>{table(buddy)}<p>Compare these techniques within Qwen3-1.7B only. Differences between models are not evidence that one quantizer is better. AWQ is not adapted to Qwen3.5 in this study.</p>
<h2>What changed</h2><p>Ten paired seeds, population 30, shared sensitivity-informed initialization, {protocol['evaluations_per_method_seed']} candidate scores per run, and 16,384 search tokens. Calibration mixes WikiText-2 training and AG News training. Validation uses 65,536 tokens from separate AG News training records; testing uses 131,072 tokens from AG News's official test split. All 400 ARC-Easy questions exclude the previous 64.</p>
<p>Allocation candidates use 2/4/8 bits because MLX GPTQ supports those widths. They fit the actual ordinary-four-bit parameter budget, including scales and biases. Uniform 3/6/8-bit baselines have different sizes and show a tradeoff curve; they are not equal-budget competitors. All methods use group size 64. Allocation methods keep embeddings at four bits; uniform baselines apply their named width to all supported matrices.</p>
<p>Our sensitivity baseline measures individual matrices' 2- and 8-bit calibration-loss changes around the four-bit model, then uses a greedy additive allocation. It is a simple local-loss heuristic, not MLX's gradient sensitivity algorithm. Both GAs and random search receive exactly the same seeded initial population. Crossover, mutation, elite retention and byte-budget repair are shared.</p>
<p>GPTQ uses 65,536 calibration tokens. The local adaptation adds per-matrix bit choices, float32 Hessians and dead-column handling to MLX-LM's GPTQ algorithm. The combined method applies GPTQ to the best validation-selected Alternis allocation; it does not rerun the search inside GPTQ and has only one selected allocation. AWQ uses the same calibration text and four-bit/group-64 settings on its supported control model.</p>
<h2>Statistical interpretation</h2><p>Paired intervals resample ten seed differences, not individual tokens. They are exploratory, unadjusted for multiple comparisons, and do not establish general superiority across models or datasets. Search methods that select the same configuration are ties. ARC-Easy accuracy is measured on 400 fixed questions; brackets show Wilson 95% intervals for individual accuracies, not paired significance tests. These are length-normalized answer-likelihood scores, not official leaderboard results.</p>
<p>Finalists, quantizer configurations and the best-under-budget export are frozen using validation before reading test scores. No test-driven tuning is performed. V1 and V2 change several factors simultaneously, so their difference cannot be attributed to a single improvement.</p>
<h2>Reproducibility</h2><p><a href="protocol.json">Protocol</a> · <a href="inputs.json">Pinned data and models</a> · <a href="summary.json">Results JSON</a> · <a href="choices_frozen.json">Frozen selection</a> · <a href="models/selected/verified.json">Export verification</a>. Intermediate scores are checkpointed in results.sqlite. Speed measurements use one warmup and three timed generations in a fresh process per configuration; peak MLX memory is not total system memory.</p></body></html>"""
    (ROOT / "report.html").write_text(doc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "sensitivity", "search", "freeze", "quantize", "evaluate", "freeze_choices", "export", "report"])
    parser.add_argument("--label")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.command in ["pilot", "sensitivity", "search", "freeze"]:
        r = Runner()
        getattr(r, {"search": "searches"}.get(args.command, args.command))()
    elif args.command == "quantize":
        quantizer(args.label, args.control, args.pilot)
    elif args.command == "evaluate":
        evaluate(args.label, args.control, args.validation_only)
    elif args.command == "freeze_choices":
        freeze_choices()
    elif args.command == "export":
        export_selected()
    else:
        report()

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        state = read(ROOT / "run.json", {"started": time.time()})
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated=time.time())
        write(ROOT / "run.json", state)
        traceback.print_exc()
        sys.exit(1)
