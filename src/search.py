"""Pure, reproducible search operators; loss is minimized."""
import random

BITS = (3, 4, 6)

def alternis_order(losses):
    ranked = sorted(range(len(losses)), key=lambda i: (losses[i], i))
    order = []
    lo, hi = 0, len(ranked) - 1
    while lo <= hi:
        order.append(ranked[hi])
        hi -= 1
        if lo <= hi:
            order.append(ranked[lo])
            lo += 1
    return order

def alternis_pool(losses, rng, size=3):
    order = alternis_order(losses)
    if not 1 <= size <= len(order):
        raise ValueError("Invalid parent pool size")
    # Zero-based bounds matching the paper's contiguous neighbor selection.
    left, right = (size - 1) // 2, size // 2
    center = rng.randrange(left, len(order) - right)
    indices = [center]
    for offset in range(1, max(left, right) + 1):
        if offset <= right:
            indices.append(center + offset)
        if offset <= left:
            indices.append(center - offset)
    return [order[i] for i in indices]

def tournament_pool(losses, rng, size=3):
    return [min(rng.sample(range(len(losses)), 2), key=lambda i: (losses[i], i))
            for _ in range(size)]

class Space:
    def __init__(self, costs):
        self.costs = [{int(k): int(v) for k, v in c.items()} for c in costs]
        self.n = len(costs)
        self.budget = sum(c[4] for c in self.costs)

    def cost(self, genome):
        return sum(c[b] for c, b in zip(self.costs, genome))

    def repair(self, genome, rng):
        g = list(genome)
        cost = self.cost(g)
        while cost > self.budget:
            choices = [i for i, b in enumerate(g) if b > 3]
            if not choices:
                raise ValueError("Budget infeasible")
            i = rng.choice(choices)
            old = g[i]
            g[i] = BITS[BITS.index(old) - 1]
            cost += self.costs[i][g[i]] - self.costs[i][old]
        # Spend available bytes without crossing the common baseline budget.
        while True:
            choices = [i for i, b in enumerate(g) if b < 6 and
                       cost + self.costs[i][BITS[BITS.index(b) + 1]] - self.costs[i][b] <= self.budget]
            if not choices:
                break
            i = rng.choice(choices)
            old = g[i]
            g[i] = BITS[BITS.index(old) + 1]
            cost += self.costs[i][g[i]] - self.costs[i][old]
        return tuple(g)

    def random(self, rng):
        return self.repair(rng.choices(BITS, weights=(.20, .65, .15), k=self.n), rng)

    def child(self, a, b, rng):
        start, end = sorted(rng.sample(range(self.n + 1), 2))
        g = list(a[:start] + b[start:end] + a[end:])
        for i in range(self.n):
            if rng.random() < 1 / self.n:
                g[i] = rng.choice([x for x in BITS if x != g[i]])
        return self.repair(g, rng)

    def initial(self, seed, count=12):
        rng = random.Random(seed + 1000)
        population = [tuple([4] * self.n)]
        while len(population) < count:
            g = self.random(rng)
            if g not in population:
                population.append(g)
        return population
