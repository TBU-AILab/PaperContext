import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved derivative-free minimizer (time-bounded): Adaptive DE + local polish.

    Core idea:
      1) Differential Evolution (DE/rand/1/bin) with:
         - small population (fast), adaptive F/CR (jitter),
         - occasional reinitialization of worst individuals (anti-stagnation),
         - bound handling by reflection (better than hard clamp).
      2) Periodic local search on the current best (coordinate + small Gaussian),
         to accelerate final convergence.

    Uses only Python standard library. Returns best fitness found.
    """
    t0 = time.time()
    if dim <= 0:
        # Degenerate; just evaluate empty vector if allowed
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    # avoid zero spans
    spans = [s if s > 0.0 else 1.0 for s in spans]

    def time_up():
        return (time.time() - t0) >= max_time

    def reflect_into_bounds(x):
        # Reflect (with wrap of multiple reflections) then clamp to be safe.
        y = list(x)
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            if lo == hi:
                y[i] = lo
                continue
            v = y[i]
            # reflect repeatedly if far outside
            # map into [lo, hi] by reflection
            width = hi - lo
            # shift to [0, width]
            v = v - lo
            # reflect using modulo on 2*width
            m = v % (2.0 * width)
            if m <= width:
                v2 = m
            else:
                v2 = 2.0 * width - m
            y[i] = lo + v2
            # numerical safety
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # --- Population size: small but not tiny ---
    # rule-of-thumb: 6..30 depending on dim
    NP = max(8, min(30, 6 + 2 * dim))

    # Initialize population
    pop = [rand_point() for _ in range(NP)]
    fit = [eval_f(ind) for ind in pop]
    best_idx = min(range(NP), key=lambda i: fit[i])
    best_x = pop[best_idx][:]
    best = fit[best_idx]

    # DE control parameters (base)
    F_base = 0.65
    CR_base = 0.9

    # Stagnation / refresh
    last_best_time = time.time()
    refresh_patience = max(0.12 * max_time, 0.6)  # seconds w/o improving best
    refresh_frac = 0.25  # replace this fraction of worst individuals

    # Local search settings
    next_local_time = time.time() + max(0.15, 0.10 * max_time)
    coord_step0 = 0.08  # fraction of span
    gauss_sigma0 = 0.06  # fraction of span
    local_tries_per_call = max(10, 3 * dim)

    # Helper: local improvement around a point (budget aware)
    def local_polish(x0, f0):
        x = x0[:]
        fx = f0
        # start steps relative to span
        steps = [coord_step0 * spans[i] for i in range(dim)]
        # do a few rounds; coordinate best-improvement + gaussian probes
        for _ in range(local_tries_per_call):
            if time_up():
                break

            # 50% coordinate move, 50% gaussian move
            if random.random() < 0.5:
                i = random.randrange(dim)
                s = steps[i]
                if s <= 0.0:
                    continue
                # try +/- step
                cand1 = x[:]
                cand1[i] += s
                cand1 = reflect_into_bounds(cand1)
                f1 = eval_f(cand1)
                if time_up():
                    return x, fx
                cand2 = x[:]
                cand2[i] -= s
                cand2 = reflect_into_bounds(cand2)
                f2 = eval_f(cand2)

                if f1 <= fx or f2 <= fx:
                    if f1 <= f2:
                        x, fx = cand1, f1
                    else:
                        x, fx = cand2, f2
                    steps[i] = min(0.25 * spans[i], steps[i] * 1.15)
                else:
                    steps[i] = max(1e-12 * spans[i], steps[i] * 0.65)
            else:
                # small gaussian perturbation in all dims
                y = x[:]
                for i in range(dim):
                    y[i] += random.gauss(0.0, gauss_sigma0 * spans[i])
                y = reflect_into_bounds(y)
                fy = eval_f(y)
                if fy <= fx:
                    x, fx = y, fy

        return x, fx

    # Main loop
    gens = 0
    while not time_up():
        gens += 1

        # Periodic local search on best
        if time.time() >= next_local_time and not time_up():
            bx, bf = local_polish(best_x, best)
            if bf < best:
                best, best_x = bf, bx[:]
                last_best_time = time.time()
            # schedule next local search (more frequent later)
            remaining = max(0.0, max_time - (time.time() - t0))
            next_local_time = time.time() + max(0.12, 0.08 * remaining)

        # Refresh worst if stagnating
        if (time.time() - last_best_time) >= refresh_patience and not time_up():
            k = max(1, int(refresh_frac * NP))
            # indices of worst k
            worst = sorted(range(NP), key=lambda i: fit[i], reverse=True)[:k]
            for idx in worst:
                # half random, half around best (heavy-tailed-ish)
                if random.random() < 0.5:
                    pop[idx] = rand_point()
                else:
                    y = best_x[:]
                    for j in range(dim):
                        # Cauchy-like perturbation
                        u = random.random()
                        c = math.tan(math.pi * (u - 0.5))
                        y[j] += 0.10 * spans[j] * c
                    pop[idx] = reflect_into_bounds(y)
                fit[idx] = eval_f(pop[idx])
                if fit[idx] < best:
                    best = fit[idx]
                    best_x = pop[idx][:]
                    last_best_time = time.time()
            # reset patience timer
            last_best_time = time.time()

        # One DE generation
        for i in range(NP):
            if time_up():
                break

            # choose r1,r2,r3 distinct and != i
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)

            # adaptive jitter for F and CR (keeps diversity)
            F = F_base + random.uniform(-0.25, 0.25)
            if F < 0.1:
                F = 0.1
            elif F > 0.95:
                F = 0.95

            CR = CR_base + random.uniform(-0.15, 0.15)
            if CR < 0.05:
                CR = 0.05
            elif CR > 0.98:
                CR = 0.98

            x_i = pop[i]
            a = pop[r1]
            b = pop[r2]
            c = pop[r3]

            # mutation: v = a + F*(b-c)
            v = [a[d] + F * (b[d] - c[d]) for d in range(dim)]

            # crossover: binomial
            jrand = random.randrange(dim)
            u = x_i[:]
            for d in range(dim):
                if d == jrand or random.random() < CR:
                    u[d] = v[d]

            u = reflect_into_bounds(u)
            fu = eval_f(u)

            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu

                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_best_time = time.time()

        # slight parameter drift: if improving often, reduce exploration a bit
        # (kept very mild to avoid overfitting)
        if gens % 12 == 0:
            # move bases slowly
            F_base = 0.55 + 0.25 * random.random()
            CR_base = 0.75 + 0.20 * random.random()

    return best
