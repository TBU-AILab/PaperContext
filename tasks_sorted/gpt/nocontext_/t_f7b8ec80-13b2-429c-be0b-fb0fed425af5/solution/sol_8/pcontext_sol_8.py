import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained (no external libs) optimizer:
    - Global exploration with Differential Evolution style "rand/1/bin"
    - Gradual parameter annealing
    - Occasional local search (coordinate perturbation) around the current best

    Returns:
        best (float): best (minimum) objective value found within max_time seconds.
    """

    # -------------------- helpers --------------------
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def reflect_into_bounds(x):
        # Reflect to keep points inside bounds (more stable than hard clamp for DE)
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect repeatedly if far out
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                if v > hi:
                    v = hi - (v - hi)
            y[i] = v
        return y

    def eval_f(x):
        # func expects an array-like of floats
        try:
            v = float(func(x))
        except Exception:
            v = float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # -------------------- initialization --------------------
    start = time.time()
    deadline = start + max_time

    # population size heuristic (keep small for speed)
    pop_size = max(8, min(40, 10 + 5 * dim))

    pop = [rand_vec() for _ in range(pop_size)]
    fit = [eval_f(ind) for ind in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # DE parameters (will be annealed)
    F0, CR0 = 0.8, 0.9
    F_min, CR_min = 0.35, 0.5

    # local search step sizes relative to range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # avoid zero ranges
    ranges = [r if r > 0 else 1.0 for r in ranges]

    # -------------------- main loop --------------------
    it = 0
    while time.time() < deadline:
        it += 1

        # anneal parameters over time
        t = (time.time() - start) / max(1e-12, max_time)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        F = F0 * (1.0 - t) + F_min * t
        CR = CR0 * (1.0 - t) + CR_min * t

        # ---- Differential Evolution generation ----
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose a, b, c distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]

            # binomial crossover
            j_rand = random.randrange(dim)
            u = pop[i][:]
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    u[j] = v[j]

            u = reflect_into_bounds(u)
            fu = eval_f(u)

            # selection
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]

        # ---- occasional local search around best ----
        # probability increases as time progresses
        if time.time() < deadline and random.random() < (0.15 + 0.35 * t):
            x = best_x[:]
            fx = best
            # step scale shrinks with time
            base_scale = (0.20 * (1.0 - t) + 0.02 * t)

            # coordinate perturbations
            for _ in range(2 * dim):
                if time.time() >= deadline:
                    return best

                j = random.randrange(dim)
                step = base_scale * ranges[j]
                # random signed step + small gaussian-like noise via sum of uniforms
                noise = (random.random() + random.random() + random.random() - 1.5)
                cand = x[:]
                cand[j] = clamp(cand[j] + step * (1.0 if random.random() < 0.5 else -1.0) + 0.25 * step * noise,
                                bounds[j][0], bounds[j][1])
                fc = eval_f(cand)

                if fc <= fx:
                    x, fx = cand, fc
                    if fc < best:
                        best, best_x = fc, cand[:]

            # inject improved best back into population (replace worst)
            worst_idx = max(range(pop_size), key=lambda k: fit[k])
            pop[worst_idx] = best_x[:]
            fit[worst_idx] = best

        # ---- small random restart if stagnating (cheap diversity) ----
        if time.time() < deadline and (it % 25 == 0):
            # replace a few worst individuals with random samples
            k = max(1, pop_size // 8)
            worst = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)[:k]
            for wi in worst:
                if time.time() >= deadline:
                    return best
                pop[wi] = rand_vec()
                fit[wi] = eval_f(pop[wi])
                if fit[wi] < best:
                    best = fit[wi]
                    best_x = pop[wi][:]

    return best
