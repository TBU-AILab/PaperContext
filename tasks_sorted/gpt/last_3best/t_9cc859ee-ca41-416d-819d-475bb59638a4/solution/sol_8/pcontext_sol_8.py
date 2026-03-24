import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-limited minimization (black-box) using a simple Differential Evolution + local refinement.
    Self-contained: no external libraries required.
    Returns: best (float) = minimum fitness found within max_time seconds.
    """

    # --- helpers ---
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def ensure_in_bounds(vec):
        return [clip(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(vec):
        # func expects "array-like"; list is fine in typical harnesses
        return float(func(vec))

    # --- time control ---
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- initialization ---
    best = float("inf")
    best_x = None

    # Population size (fixed line that had the syntax error in your previous code)
    # The buggy line had one extra ')'. This is the corrected version:
    base_lambda = max(
        10,
        min(
            50 + 4 * dim,
            8 + 3 * dim + int(3 * math.log(dim + 1.0))
        )
    )
    pop_size = int(base_lambda)

    # DE parameters
    F = 0.7
    CR = 0.9

    pop = []
    fit = []
    for _ in range(pop_size):
        x = rand_vec()
        f = eval_f(x)
        pop.append(x)
        fit.append(f)
        if f < best:
            best = f
            best_x = x[:]

    # --- main loop ---
    while time.time() < deadline:
        # Differential Evolution generation
        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # choose a,b,c distinct and != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation
            mutant = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # crossover
            j_rand = random.randrange(dim)
            trial = []
            for d in range(dim):
                if random.random() < CR or d == j_rand:
                    trial.append(mutant[d])
                else:
                    trial.append(pop[i][d])

            trial = ensure_in_bounds(trial)
            f_trial = eval_f(trial)

            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        # Lightweight local refinement around best (coordinate-wise jitter)
        if best_x is not None and time.time() < deadline:
            x = best_x[:]
            # step size proportional to range
            for d in range(dim):
                if time.time() >= deadline:
                    return best
                lo, hi = bounds[d]
                span = hi - lo
                # try a few perturbations per dimension
                for _ in range(2):
                    if time.time() >= deadline:
                        return best
                    step = (random.random() * 0.2 - 0.1) * span
                    cand = x[:]
                    cand[d] = clip(cand[d] + step, lo, hi)
                    f_cand = eval_f(cand)
                    if f_cand < best:
                        best = f_cand
                        best_x = cand
                        x = cand

    return best
