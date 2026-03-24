import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Differential Evolution (DE)
    with occasional random injections and a best-so-far return.

    Args:
        func: callable(params:list[float]) -> float
        dim: int
        bounds: list of (low, high) pairs, length == dim
        max_time: seconds (int/float)

    Returns:
        best: float, best (minimum) fitness found within time limit
    """
    t0 = time.time()

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # If bounds are degenerate in any dimension, handle gracefully
    for i in range(dim):
        if spans[i] < 0:
            raise ValueError("Each bound must be (low<=high).")
        if spans[i] == 0:
            # fixed dimension; still fine
            pass

    # --- DE parameters (chosen to be robust across problems) ---
    # Keep population modest for speed; scale with dimension.
    pop_size = max(8, min(40, 10 + 5 * dim))
    F_base = 0.6      # mutation factor baseline
    CR_base = 0.9     # crossover prob baseline
    inject_rate = 0.05  # random re-seeding to avoid stagnation

    # --- initialization ---
    pop = [rand_vec() for _ in range(pop_size)]
    fit = []
    best = float("inf")
    best_x = None

    for x in pop:
        f = func(x)
        fit.append(f)
        if f < best:
            best = f
            best_x = x[:]

    # --- main loop ---
    gen = 0
    while True:
        if time.time() - t0 >= max_time:
            return best

        gen += 1

        # Mild parameter jitter over time to improve exploration/exploitation balance
        # (kept deterministic-ish; no external libs).
        F = min(0.95, max(0.2, F_base + 0.15 * (random.random() - 0.5)))
        CR = min(0.98, max(0.1, CR_base + 0.20 * (random.random() - 0.5)))

        # Iterate individuals
        for i in range(pop_size):
            if time.time() - t0 >= max_time:
                return best

            # Random injection (restart) with small probability
            if random.random() < inject_rate:
                trial = rand_vec()
                ftrial = func(trial)
                if ftrial <= fit[i]:
                    pop[i] = trial
                    fit[i] = ftrial
                    if ftrial < best:
                        best = ftrial
                        best_x = trial[:]
                continue

            # Choose r1, r2, r3 distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = random.choice(idxs); idxs.remove(r1)
            r2 = random.choice(idxs); idxs.remove(r2)
            r3 = random.choice(idxs)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            target = pop[i]

            # DE/rand/1 mutation
            mutant = [0.0] * dim
            for d in range(dim):
                mutant[d] = x1[d] + F * (x2[d] - x3[d])

            # Binomial crossover
            jrand = random.randrange(dim)
            trial = [0.0] * dim
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    trial[d] = clip(mutant[d], d)
                else:
                    trial[d] = target[d]

            # Selection
            ftrial = func(trial)
            if ftrial <= fit[i]:
                pop[i] = trial
                fit[i] = ftrial
                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]

        # If close to time limit, return promptly
        if time.time() - t0 >= max_time:
            return best
