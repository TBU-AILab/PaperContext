import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: Differential Evolution + periodic local refinement.

    - No external dependencies.
    - Works with any callable func(params)->float, where params is a list/array-like of length dim.
    - Respects per-dimension bounds.
    - Returns: best (float) = lowest fitness value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def clip_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def safe_eval(x):
        # Robust against occasional numerical issues inside func.
        try:
            y = func(x)
            if y is None:
                return float("inf")
            # Handle NaN / inf
            if isinstance(y, float):
                if math.isnan(y) or math.isinf(y):
                    return float("inf")
            return float(y)
        except Exception:
            return float("inf")

    # ---------- initialization ----------
    # Population size: small but decent for time-bounded scenarios
    pop_size = max(10, 6 * dim)
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # Differential Evolution parameters (adaptive-ish)
    F = 0.6          # mutation factor
    CR = 0.9         # crossover rate
    stagnation = 0

    # Local search step sizes (as fraction of range)
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_sigma = [0.15 * (r if r > 0 else 1.0) for r in ranges]

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1
        improved_any = False

        # A light adaptation to encourage exploration when stuck
        if stagnation > 15:
            F = min(0.9, 0.5 + random.random() * 0.4)  # 0.5..0.9
            CR = 0.5 + random.random() * 0.5           # 0.5..1.0
        else:
            # gentle random walk around defaults
            F = clamp(F + (random.random() - 0.5) * 0.05, 0.35, 0.9)
            CR = clamp(CR + (random.random() - 0.5) * 0.05, 0.1, 1.0)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            # Choose three distinct indices a,b,c != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]
            # DE/rand/1 mutation: v = xa + F*(xb-xc)
            v = [xa[d] + F * (xb[d] - xc[d]) for d in range(dim)]

            # Binomial crossover
            j_rand = random.randrange(dim)
            trial = []
            for d in range(dim):
                if random.random() < CR or d == j_rand:
                    trial.append(v[d])
                else:
                    trial.append(pop[i][d])
            trial = clip_vec(trial)

            f_trial = safe_eval(trial)

            # Selection
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                improved_any = True

                if f_trial < best:
                    best = f_trial
                    best_x = trial[:]

        if improved_any:
            stagnation = 0
        else:
            stagnation += 1

        # ---------- periodic local refinement around current best ----------
        # Use a shrinking Gaussian step; do a few attempts per refinement.
        if gen % 5 == 0:
            # shrink over time and with stagnation
            time_frac = (time.time() - t0) / max(1e-9, max_time)
            shrink = max(0.02, 1.0 - 0.85 * time_frac)
            # if stuck, briefly increase local probing a bit
            bump = 1.0 + 0.25 * min(10, stagnation)

            sigmas = [s * shrink * bump for s in base_sigma]

            # number of local tries scales mildly with dimension
            local_tries = max(5, 2 * dim)

            for _ in range(local_tries):
                if time.time() >= deadline:
                    return best

                cand = []
                for d in range(dim):
                    # Box-Muller gaussian
                    u1 = max(1e-12, random.random())
                    u2 = random.random()
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    cand.append(best_x[d] + sigmas[d] * z)

                cand = clip_vec(cand)
                f_cand = safe_eval(cand)
                if f_cand < best:
                    best = f_cand
                    best_x = cand[:]
                    # also inject into population by replacing worst
                    worst_idx = max(range(pop_size), key=lambda k: fit[k])
                    pop[worst_idx] = cand[:]
                    fit[worst_idx] = f_cand

    return best
