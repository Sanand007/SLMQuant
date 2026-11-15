"""Budget-constrained operators with shared informed initialization."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import Space

class InformedSpace(Space):
    bits = (2, 4, 8)
    informed = None

    def repair(self, genome, rng):
        g = list(genome)
        cost = self.cost(g)
        while cost > self.budget:
            i = rng.choice([i for i, b in enumerate(g) if b > 2])
            old = g[i]
            g[i] = self.bits[self.bits.index(old) - 1]
            cost += self.costs[i][g[i]] - self.costs[i][old]
        while True:
            choices = [i for i, b in enumerate(g) if b < 8 and
                       cost + self.costs[i][self.bits[self.bits.index(b) + 1]] - self.costs[i][b] <= self.budget]
            if not choices:
                return tuple(g)
            i = rng.choice(choices)
            old = g[i]
            g[i] = self.bits[self.bits.index(old) + 1]
            cost += self.costs[i][g[i]] - self.costs[i][old]

    def random(self, rng):
        return self.repair(rng.choices(self.bits, weights=(.15, .80, .05), k=self.n), rng)

    def child(self, a, b, rng):
        start, end = sorted(rng.sample(range(self.n + 1), 2))
        g = list(a[:start] + b[start:end] + a[end:])
        for i in range(self.n):
            if rng.random() < 1 / self.n:
                g[i] = rng.choice([x for x in self.bits if x != g[i]])
        return self.repair(g, rng)

    def initial(self, seed, count=30):
        rng = random.Random(seed + 1000)
        population = [tuple([4] * self.n)]
        if self.informed is not None and self.informed not in population:
            population.append(self.informed)
        while len(population) < count:
            if self.informed is not None and len(population) < count // 2:
                g = list(self.informed)
                for i in rng.sample(range(self.n), rng.randint(1, max(2, self.n // 16))):
                    g[i] = rng.choice(self.bits)
                g = self.repair(g, rng)
            else:
                g = self.random(rng)
            if g not in population:
                population.append(g)
        return population

    def sensitivity_allocation(self, deltas):
        # Predicted loss is additive; probe interactions are deliberately not assumed known.
        g = [2] * self.n
        cost = self.cost(g)
        while True:
            choices = []
            for i, b in enumerate(g):
                for new in self.bits:
                    if new <= b:
                        continue
                    extra = self.costs[i][new] - self.costs[i][b]
                    gain = deltas[i][b] - deltas[i][new]
                    if cost + extra <= self.budget and gain > 0:
                        choices.append((gain / extra, gain, -i, new, extra))
            if not choices:
                break
            _, _, neg_i, new, extra = max(choices)
            g[-neg_i] = new
            cost += extra
        return tuple(g)
