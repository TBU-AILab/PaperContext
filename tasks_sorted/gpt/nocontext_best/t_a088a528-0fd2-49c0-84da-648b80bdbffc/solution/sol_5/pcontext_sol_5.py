import time
import random
import math

def run(func, dim, bounds, max_time):
    start = time.time()

    # Fix: correct list comprehension syntax
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]

    def clip(x):
        # keep x within bounds
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    # Initialize with a random point
    best_x = [random.uniform(lows[i], highs[i]) for i in range(dim)]
    best = float(func(best_x))

    # Simple time-bounded stochastic search with shrinking step size
    step = [0.2 * (highs[i] - lows[i]) if highs[i] > lows[i] else 1.0 for i in range(dim)]

    while time.time() - start < max_time:
        # propose a local perturbation
        cand = [best_x[i] + random.gauss(0.0, step[i]) for i in range(dim)]
        cand = clip(cand)

        fitness = float(func(cand))
        if fitness <= best:
            best = fitness
            best_x = cand
            # small intensification
            step = [s * 0.99 for s in step]
        else:
            # occasional exploration
            if random.random() < 0.05:
                cand = [random.uniform(lows[i], highs[i]) for i in range(dim)]
                fitness = float(func(cand))
                if fitness < best:
                    best = fitness
                    best_x = cand

        # prevent step from collapsing to zero
        step = [max(s, 1e-12 * (highs[i] - lows[i] if highs[i] > lows[i] else 1.0)) for i, s in enumerate(step)]

    # return fitness of the best found solution
    return best
