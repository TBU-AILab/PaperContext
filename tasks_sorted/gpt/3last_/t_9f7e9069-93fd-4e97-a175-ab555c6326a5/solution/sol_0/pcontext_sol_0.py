import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization using a self-contained Differential Evolution (DE)
    with occasional random restarts and bound handling.

    Returns:
        best (float): fitness of the best solution found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ----- helpers -----
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    # Defensive: if any dimension has zero span, keep it fixed
    fixed = [span[i] == 0.0 for i in range(dim)]

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
            else:
                x[i] = lo[i] + random.random() * span[i]
        return x

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]

    def safe_eval(x):
        # func is expected to accept an array-like; we pass a plain list.
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ----- DE parameters (adaptive-ish via mild randomness) -----
    # Population size: small enough for speed, scales with dimension.
    pop_size = max(12, min(60, 10 * dim))
    # Base mutation/crossover ranges
    F_min, F_max = 0.45, 0.95
    CR_min, CR_max = 0.2, 0.95

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best = min(fit) if fit else float("inf")
    best_idx = fit.index(best) if fit else 0
    best_x = pop[best_idx][:] if pop else rand_vec()

    # Restart controls
    no_improve = 0
    restart_after = max(30, 10 * dim)  # generations without improvement before partial restart

    gen = 0
    while time.time() < deadline:
        gen += 1

        # Mildly randomize F and CR per generation (helps robustness)
        F = F_min + (F_max - F_min) * random.random()
        CR = CR_min + (CR_max - CR_min) * random.random()

        improved_this_gen = False

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # Choose r1, r2, r3 distinct and not equal to i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)

            x1, x2, x3 = pop[r1], pop[r2], pop[r3]

            # Mutation: v = x1 + F*(x2 - x3)
            v = [0.0] * dim
            for d in range(dim):
                if fixed[d]:
                    v[d] = lo[d]
                else:
                    v[d] = x1[d] + F * (x2[d] - x3[d])

            # Crossover (binomial)
            u = pop[i][:]  # start from target
            jrand = random.randrange(dim)  # ensure at least one dimension from mutant
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            clip_inplace(u)

            fu = safe_eval(u)
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best:
                    best = fu
                    best_x = u[:]
                    improved_this_gen = True

        if improved_this_gen:
            no_improve = 0
        else:
            no_improve += 1

        # Partial random restart if stagnating: keep elites, re-sample the rest
        if no_improve >= restart_after and time.time() < deadline:
            no_improve = 0
            # Keep top k
            k = max(2, pop_size // 5)
            order = sorted(range(pop_size), key=lambda idx: fit[idx])
            elites = [pop[idx] for idx in order[:k]]
            elites_fit = [fit[idx] for idx in order[:k]]

            pop = elites[:]
            fit = elites_fit[:]
            while len(pop) < pop_size:
                x = rand_vec()
                pop.append(x)
                fit.append(safe_eval(x))

            # Refresh best trackers
            best = min(best, min(fit))
            best_idx = fit.index(min(fit))
            if fit[best_idx] <= best:
                best_x = pop[best_idx][:]

    return best
