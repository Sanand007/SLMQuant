"""Uncapped 30-generation, 500-distinct-candidate Alternis comparison."""

import hashlib
from pathlib import Path
import random
import time
from types import MethodType

import wide_search_study as previous


STUDY = previous.V3 / "studies/wide30_mlx_affine_qwen3_1_7b"
previous.STUDY = STUDY
shared = previous.shared
shared.STUDY = STUDY
base_load_runner = shared.load_runner


def no_resource_cap(_started):
    """The user explicitly removed the former v3 time and storage caps."""


def source_hashes():
    files = previous.source_hashes()
    path = Path(__file__)
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def search_run(self, method, seed):
    from search import alternis_pool, tournament_pool

    path = STUDY / f"search_{method}_{seed}.json"
    prior = shared.read(path, {})
    if prior.get("complete"):
        return prior
    saved = prior.get("history", [])
    rng = random.Random(seed)
    population = self.space.initial(seed, self.p["population"])
    if len(population) != 18:
        raise RuntimeError("Frozen initial population must contain 18 distinct candidates")
    seen = set()
    history = []
    started = time.perf_counter()

    def evaluate(genome, generation):
        index = len(history)
        if index < len(saved):
            row = saved[index]
            if tuple(row["genome"]) != tuple(genome) or row["generation"] != generation:
                raise RuntimeError("Candidate checkpoint diverged from deterministic replay")
        else:
            result = self.evaluate(genome, "train")
            row = {
                "id": self.ident(genome),
                "genome": list(genome),
                "generation": generation,
                "nll": result["nll"],
                "parameter_bytes": result["parameter_bytes"],
            }
        seen.add(tuple(genome))
        history.append(row)
        if index >= len(saved):
            shared.write(path, {"method": method, "seed": seed, "complete": False,
                                "generation": generation, "history": history})
            self.run.update(status="searching", method=method, seed=seed,
                            evaluated=len(history), generation=generation, updated=time.time())
            shared.write(STUDY / "run.json", self.run)
        return row["nll"]

    losses = [evaluate(genome, 0) for genome in population]
    metrics = []

    def record(generation, genomes, scores):
        pairs = [(a, b) for i, a in enumerate(genomes) for b in genomes[i + 1:]]
        diversity = (sum(sum(x != y for x, y in zip(a, b)) for a, b in pairs) / len(pairs)
                     if pairs else 0.0)
        metrics.append({"generation": generation, "population": len(genomes),
                        "evaluated": len(history), "best_train_nll": min(scores),
                        "mean_pairwise_hamming": diversity})
        shared.write(STUDY / f"search_{method}_{seed}_generations.json", metrics)

    record(0, population, losses)
    for generation in range(1, 31):
        elite = min(range(len(losses)), key=lambda index: losses[index])
        next_population = [population[elite]]
        next_losses = [losses[elite]]
        pool = (alternis_pool(losses, rng) if method == "alternis" else
                tournament_pool(losses, rng) if method == "tournament" else None)
        offspring = 17 if generation in (15, 30) else 16
        attempts = 0
        while len(next_population) < offspring + 1:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Search failed to produce a distinct feasible candidate")
            if method == "random":
                child = self.space.random(rng)
            else:
                a, b = rng.sample(pool, 2)
                child = self.space.child(population[a], population[b], rng)
            if child in seen:
                continue
            next_losses.append(evaluate(child, generation))
            next_population.append(child)
        population, losses = next_population, next_losses
        record(generation, population, losses)
    if len(history) != 500 or len(seen) != 500 or len(saved) > len(history):
        raise RuntimeError("Thirty generations did not yield exactly 500 distinct scores")
    state = {"method": method, "seed": seed, "complete": True,
             "generation": 30, "history": history,
             "seconds_this_invocation": time.perf_counter() - started}
    shared.write(path, state)
    return state


def load_runner(started):
    runner, legacy = base_load_runner(started)
    runner.ident = legacy.ident
    runner.search_run = MethodType(search_run, runner)
    return runner, legacy


shared.prepare = previous.prepare
shared.source_hashes = source_hashes
shared.load_runner = load_runner
shared.limits = no_resource_cap

if __name__ == "__main__":
    shared.main()
