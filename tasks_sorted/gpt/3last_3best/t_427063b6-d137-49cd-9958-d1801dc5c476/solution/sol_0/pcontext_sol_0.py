import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a lightweight Differential Evolution (DE)
    with occasional random restarts.

    - No external libraries required.
    - Works with func(params)->float where params is a list/sequence of length dim.
    - Respects bounds [(low, high), ...].
    Returns: best (float) best fitness found within max_time seconds.
    """
    # ------------------------ helpers ------------------------
    def clip(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def ensure_bounds(vec):
        return [clip(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Evaluate with basic safety: if func misbehaves, treat as bad candidate
    def safe_eval(vec):
        try:
            v = func(vec)
            # handle NaN / inf
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                if v != v or v == float("inf") or v == float("-inf"):
                    return float("inf")
                return float(v)
            return float("inf")
        except Exception:
            return float("inf")

    # ------------------------ initialization ------------------------
    t0 = time.time()
    deadline = t0 + float(max_time)

    # If time is extremely small, do at least one evaluation
    best = float("inf")
    best_x = None

    # Population sizing (small, time-friendly)
    pop_size = max(8, min(30, 10 + 2 * dim))
    # DE parameters (will jitter a bit)
    F_base = 0.7
    CR_base = 0.9

    # initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    # update best
    for i in range(pop_size):
        if fit[i] < best:
            best = fit[i]
            best_x = pop[i][:]

    # ------------------------ main loop ------------------------
    gen = 0
    no_improve_gens = 0

    while time.time() < deadline:
        gen += 1
        improved_this_gen = False

        # Jitter parameters slightly each generation for robustness
        F = clip(random.gauss(F_base, 0.15), 0.2, 0.95)
        CR = clip(random.gauss(CR_base, 0.10), 0.1, 0.99)

        for i in range(pop_size):
            # Time check inside loop to respect max_time tightly
            if time.time() >= deadline:
                return best

            # Choose 3 distinct indices != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            xa, xb, xc = pop[a], pop[b], pop[c]

            # Mutation: v = xa + F*(xb-xc)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]

            # Crossover (binomial) to form trial
            jrand = random.randrange(dim)
            trial = []
            for j in range(dim):
                if random.random() < CR or j == jrand:
                    trial.append(v[j])
                else:
                    trial.append(pop[i][j])

            trial = ensure_bounds(trial)

            # Selection
            f_trial = safe_eval(trial)
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                improved_this_gen = True
                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        # Simple stagnation handling: partial random restart
        if improved_this_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # If stuck, re-seed a fraction of worst individuals
        if no_improve_gens >= 10:
            no_improve_gens = 0
            # replace worst 30% with random samples (keep best)
            k = max(1, pop_size * 3 // 10)
            # indices sorted by fitness descending (worst first)
            worst = sorted(range(pop_size), key=lambda idx: fit[idx], reverse=True)
            for idx in worst[:k]:
                if time.time() >= deadline:
                    return best
                # keep the current global best safe
                if best_x is not None and pop[idx] == best_x:
                    continue
                x = rand_vec()
                fx = safe_eval(x)
                pop[idx] = x
                fit[idx] = fx
                if fx < best:
                    best = fx
                    best_x = x[:]

    return best
