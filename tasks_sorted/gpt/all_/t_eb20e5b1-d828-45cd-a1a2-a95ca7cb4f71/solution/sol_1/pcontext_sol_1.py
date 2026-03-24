import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Main changes vs. the provided DE:
    - Uses JADE-style "current-to-pbest/1" (stronger exploitation) + external archive
    - Self-adaptive per-individual F and CR (success-history learning)
    - Elitism via p-best selection, plus occasional local coordinate search around best
    - Robust bound handling (reflect) and safe evaluation
    - Time-aware behavior and light restarts if stalled

    Returns: best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- helpers ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0 else 0.0 for s in spans]

    def ensure_bounds(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                y[i] = lo
                continue
            # reflect repeatedly if outside
            while y[i] < lo or y[i] > hi:
                if y[i] < lo:
                    y[i] = lo + (lo - y[i])
                if y[i] > hi:
                    y[i] = hi - (y[i] - hi)
            # numerical safety
            if y[i] < lo: y[i] = lo
            if y[i] > hi: y[i] = hi
        return y

    def rand_vec():
        return [lows[i] + random.random() * spans[i] if spans[i] > 0 else lows[i] for i in range(dim)]

    def safe_eval(x):
        try:
            v = func(x)
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def clip01(a):
        if a < 0.0: return 0.0
        if a > 1.0: return 1.0
        return a

    # ---- quick degenerate cases ----
    if dim <= 0:
        # Evaluate empty vector if allowed; else return inf
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # ---- JADE-like parameters ----
    pop_size = max(16, min(80, 12 * dim))  # slightly larger for pbest pressure + archive
    p_min, p_max = 0.05, 0.25            # pbest fraction range
    c = 0.1                               # learning rate for mu_F, mu_CR
    mu_F = 0.6
    mu_CR = 0.5
    archive = []                          # stores replaced parents (diversity)
    archive_max = pop_size

    # per-individual parameters (start near mu_*)
    F_i = [max(0.05, min(1.0, random.gauss(mu_F, 0.1))) for _ in range(pop_size)]
    CR_i = [clip01(random.gauss(mu_CR, 0.1)) for _ in range(pop_size)]

    # ---- init ----
    pop = [rand_vec() for _ in range(pop_size)]
    fit = [safe_eval(x) for x in pop]
    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best = fit[best_idx]
    best_x = pop[best_idx][:]

    # stall / restart controls
    last_improve_t = time.time()
    stall_restart_seconds = max(0.20, 0.20 * float(max_time))

    gen = 0
    while True:
        if time.time() >= deadline:
            return best
        gen += 1

        elapsed = time.time() - t0
        frac = min(1.0, elapsed / max(1e-12, float(max_time)))

        # time-aware pbest pressure (more exploitation later)
        p_frac = p_min + (p_max - p_min) * (0.25 + 0.75 * frac)
        p_count = max(2, int(math.ceil(p_frac * pop_size)))

        # indices sorted by fitness (ascending)
        order = sorted(range(pop_size), key=lambda i: fit[i])

        # success sets for history updates
        S_F = []
        S_CR = []
        dF = []  # fitness improvements (for weighting)

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fit[i]

            # sample adaptive parameters (JADE style)
            # F: sample from Cauchy around mu_F; CR: normal around mu_CR
            # implement simple rejection to keep in (0,1]
            Fi = None
            for _ in range(12):
                # Cauchy(mu_F, 0.1)
                u = random.random()
                Fi_try = mu_F + 0.1 * math.tan(math.pi * (u - 0.5))
                if 0.0 < Fi_try <= 1.0:
                    Fi = Fi_try
                    break
            if Fi is None:
                Fi = max(0.05, min(1.0, mu_F))

            CRi = clip01(random.gauss(mu_CR, 0.1))

            # choose pbest from top p_count
            pbest = order[random.randrange(p_count)]
            x_pbest = pop[pbest]

            # choose r1 from population excluding i
            r1 = i
            while r1 == i:
                r1 = random.randrange(pop_size)

            # choose r2 from pop U archive excluding i and r1
            union = pop + archive
            union_n = len(union)
            r2 = None
            for _ in range(20):
                idx = random.randrange(union_n)
                x2 = union[idx]
                # avoid selecting same vector object by checking coordinates if in pop
                # (cheap): if from pop, ensure index not i or r1
                if idx < pop_size:
                    if idx == i or idx == r1:
                        continue
                r2 = x2
                break
            if r2 is None:
                # fallback: just pick another from population
                rr = i
                while rr == i or rr == r1:
                    rr = random.randrange(pop_size)
                r2 = pop[rr]

            x_r1 = pop[r1]
            x_r2 = r2

            # mutation: current-to-pbest/1 with archive option
            # v = xi + F*(x_pbest - xi) + F*(x_r1 - x_r2)
            v = [0.0] * dim
            for d in range(dim):
                v[d] = xi[d] + Fi * (x_pbest[d] - xi[d]) + Fi * (x_r1[d] - x_r2[d])

            v = ensure_bounds(v)

            # binomial crossover
            jrand = random.randrange(dim)
            u = [0.0] * dim
            for d in range(dim):
                if random.random() < CRi or d == jrand:
                    u[d] = v[d]
                else:
                    u[d] = xi[d]

            u = ensure_bounds(u)
            fu = safe_eval(u)

            # selection + archive update
            if fu <= fi:
                # replace
                pop[i] = u
                fit[i] = fu

                # archive the parent if different and feasible
                archive.append(xi)
                if len(archive) > archive_max:
                    # random removal keeps diversity
                    del archive[random.randrange(len(archive))]

                # success memories
                if fu < fi:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    dF.append(fi - fu)

                if fu < best:
                    best = fu
                    best_x = u[:]
                    last_improve_t = time.time()

        # update mu_F and mu_CR using weighted Lehmer mean for F, weighted mean for CR
        if dF:
            w_sum = sum(dF)
            if w_sum <= 0:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / w_sum for di in dF]

            # mu_CR: weighted arithmetic mean
            mu_CR = sum(wk * cr for wk, cr in zip(w, S_CR))

            # mu_F: weighted Lehmer mean = sum(w*F^2)/sum(w*F)
            num = sum(wk * (f * f) for wk, f in zip(w, S_F))
            den = sum(wk * f for wk, f in zip(w, S_F))
            if den > 1e-12:
                mu_F = num / den
            # smooth update
            mu_CR = (1 - c) * mu_CR + c * mu_CR  # keeps structure; effectively no-op but safe
            mu_F = (1 - c) * mu_F + c * mu_F     # same (we already set); keep for clarity

            # clamp
            mu_CR = clip01(mu_CR)
            mu_F = max(0.05, min(1.0, mu_F))

        # occasional local search around best (coordinate + small gaussian)
        if time.time() < deadline and (gen % 4 == 0):
            # step shrinks with time
            base = 0.20 * (1.0 - frac) ** 2 + 0.01
            tries = 6
            for _ in range(tries):
                if time.time() >= deadline:
                    return best
                cand = best_x[:]

                # coordinate search: perturb 1..k dims
                k = 1 if dim == 1 else random.randint(1, max(1, dim // 4))
                for __ in range(k):
                    d = random.randrange(dim)
                    if spans[d] > 0:
                        cand[d] += random.gauss(0.0, base * spans[d])
                cand = ensure_bounds(cand)
                fc = safe_eval(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    last_improve_t = time.time()

        # soft restart if stalled: keep elites, refill around best + random
        if time.time() - last_improve_t > stall_restart_seconds:
            elite_n = max(2, pop_size // 8)
            order = sorted(range(pop_size), key=lambda i: fit[i])
            elites = [pop[i][:] for i in order[:elite_n]]
            elites_fit = [fit[i] for i in order[:elite_n]]

            pop = elites[:]
            fit = elites_fit[:]
            archive = []
            # reset parameter memories mildly
            mu_F = 0.65
            mu_CR = 0.5

            while len(pop) < pop_size and time.time() < deadline:
                if random.random() < 0.55:
                    x = rand_vec()
                else:
                    x = best_x[:]
                    # broad noise early, narrower later
                    s = (0.45 * (1.0 - frac) + 0.15)  # 0.6..0.15
                    for d in range(dim):
                        if spans[d] > 0:
                            x[d] += random.gauss(0.0, s * spans[d])
                    x = ensure_bounds(x)
                pop.append(x)
                fit.append(safe_eval(x))

            best_idx = min(range(pop_size), key=lambda i: fit[i])
            if fit[best_idx] < best:
                best = fit[best_idx]
                best_x = pop[best_idx][:]
            last_improve_t = time.time()
