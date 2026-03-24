import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Differential Evolution (DE)
    with occasional random restarts.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/sequence of length dim
    dim : int
    bounds : list of (low, high)
    max_time : int or float
        time budget in seconds

    Returns
    -------
    best : float
        best (minimum) fitness value found within the time budget
    """

    # --- helpers ---
    def clip(x):
        for i in range(dim):
            lo, hi = bounds[i]
            if x[i] < lo: x[i] = lo
            elif x[i] > hi: x[i] = hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Robust evaluation: if func errors or returns non-finite, treat as very bad
    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            # handle NaN/inf
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
                return float(v)
            # try cast
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # --- algorithm parameters (kept simple & adaptive) ---
    # Population size: small enough for speed, big enough for exploration
    pop_size = max(8, min(30, 10 + 2 * dim))
    # DE parameters; will be jittered a bit over time for robustness
    F_base = 0.7   # differential weight
    CR_base = 0.9  # crossover probability

    start = time.perf_counter()
    deadline = start + float(max_time)

    # --- initialize population ---
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_val = fit[best_idx]
    best_x = pop[best_idx][:]

    # restart control
    no_improve = 0
    restart_after = max(20, 10 * dim)  # generations without improvement

    gen = 0
    while True:
        if time.perf_counter() >= deadline:
            return best_val

        gen += 1
        improved_this_gen = False

        # mild parameter jitter to avoid stagnation
        F = min(1.0, max(0.1, random.gauss(F_base, 0.08)))
        CR = min(1.0, max(0.0, random.gauss(CR_base, 0.08)))

        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return best_val

            # choose 3 distinct indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation: v = xa + F*(xb - xc)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]
            v = clip(v)

            # binomial crossover
            trial = pop[i][:]
            j_rand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # selection
            f_trial = safe_eval(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial

                if f_trial < best_val:
                    best_val = f_trial
                    best_x = trial[:]
                    improved_this_gen = True

        if improved_this_gen:
            no_improve = 0
        else:
            no_improve += 1

        # --- occasional random restart around the best to escape local minima ---
        if no_improve >= restart_after:
            no_improve = 0

            # keep a few best, reinitialize the rest (some globally, some locally)
            order = sorted(range(pop_size), key=lambda k: fit[k])
            keep = max(2, pop_size // 4)
            survivors = [pop[k][:] for k in order[:keep]]
            survivors_fit = [fit[k] for k in order[:keep]]

            # local perturbation scale decreases with time (annealing-like)
            t = (time.perf_counter() - start) / max(1e-9, (deadline - start))
            # scale in [~0.5 .. ~0.05] of range width
            base_scale = 0.5 * (1.0 - t) + 0.05 * t

            new_pop = survivors[:]
            new_fit = survivors_fit[:]
            while len(new_pop) < pop_size and time.perf_counter() < deadline:
                if random.random() < 0.6:
                    # local sample around best_x
                    x = []
                    for d in range(dim):
                        lo, hi = bounds[d]
                        width = hi - lo
                        # gaussian around best_x with clipped bounds
                        val = random.gauss(best_x[d], base_scale * width)
                        if val < lo: val = lo
                        if val > hi: val = hi
                        x.append(val)
                else:
                    # global sample
                    x = rand_vec()

                f = safe_eval(x)
                new_pop.append(x)
                new_fit.append(f)
                if f < best_val:
                    best_val = f
                    best_x = x[:]

            pop, fit = new_pop, new_fit

    # unreachable
    # return best_val
